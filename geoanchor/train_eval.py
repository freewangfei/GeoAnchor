from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .data import PorosityNPZDataset, collate_batch
from .metrics import regression_metrics, round_metrics
from .model import GeoAnchorNet, count_parameters


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    epochs: int = 30
    batch_size: int = 4
    lr: float = 3e-4
    seed: int = 20260422
    device: str = "cuda"
    calibration_mode: str = "none"
    use_prior_condition: bool = True
    anchor_band_strength: float = 1.0
    curve_gate_strength: float = 1.0
    curve_gate_mode: str = "column"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    err = torch.abs(pred - target) * mask
    if weight is not None:
        err = err * weight
    return err.sum() / (mask.sum() + 1e-6)


def smoothness_loss(pred: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
    boundary = structure[:, 2:3]
    fault = structure[:, 3:4]
    smooth_region = torch.clamp(1.0 - 0.7 * boundary - 0.8 * fault, min=0.0)
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dz = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    return (dx * smooth_region[:, :, :, 1:]).mean() + 0.5 * (dz * smooth_region[:, :, 1:, :]).mean()


def component_target_distribution(structure: torch.Tensor, anchor_mask: torch.Tensor, n_components: int) -> torch.Tensor:
    boundary = torch.clamp(structure[:, 2:3] + structure[:, 3:4], 0.0, 1.0)
    anchor_local = F.avg_pool2d(anchor_mask, kernel_size=(9, 5), stride=1, padding=(4, 2))
    anchor_local = torch.clamp(6.0 * anchor_local, 0.0, 1.0)
    anchor_column = (anchor_mask.sum(dim=2, keepdim=True) > 0.0).float().expand_as(anchor_mask)
    curve_zone = torch.maximum(anchor_column, anchor_local)
    smooth = (1.0 - boundary) * (1.0 - anchor_local)
    detail = boundary * (1.0 - 0.5 * anchor_local)
    well_local = anchor_local * (1.0 - 0.5 * boundary)
    anchor_curve = curve_zone * (0.35 + 0.65 * (1.0 - boundary))
    target = torch.cat([smooth, detail, well_local, anchor_curve], dim=1)
    if target.shape[1] != n_components:
        target = torch.full(
            (structure.shape[0], n_components, structure.shape[2], structure.shape[3]),
            1.0 / float(n_components),
            device=structure.device,
            dtype=structure.dtype,
        )
    return torch.clamp(target, min=1e-6) / torch.clamp(target.sum(dim=1, keepdim=True), min=1e-6)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def make_model(dataset: PorosityNPZDataset, cfg: TrainConfig) -> GeoAnchorNet:
    return GeoAnchorNet(
        structure_ch=dataset.structure.shape[1],
        use_prior_condition=cfg.use_prior_condition,
        anchor_band_strength=cfg.anchor_band_strength,
        curve_gate_strength=cfg.curve_gate_strength,
        curve_gate_mode=cfg.curve_gate_mode,
    )


def forward_batch(model: GeoAnchorNet, batch: dict[str, object], device: torch.device):
    anchor_mask = batch.get("supervised_mask")
    return model(
        batch["seismic"].to(device),
        batch["structure"].to(device),
        batch["prior"].to(device),
        batch["anchor"].to(device),
        anchor_mask.to(device) if anchor_mask is not None else None,
    )


def evaluate(model: GeoAnchorNet, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    preds, trues, priors, masks, domains = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = forward_batch(model, batch, device)
            preds.append(out.porosity.cpu().numpy())
            trues.append(batch["porosity"].numpy())
            priors.append(batch["prior"].numpy())
            domains.extend(batch["domain"])
            if "eval_mask" in batch:
                masks.append(batch["eval_mask"].numpy())
            elif "supervised_mask" in batch:
                masks.append(batch["supervised_mask"].numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    prior = np.concatenate(priors, axis=0)
    mask = np.concatenate(masks, axis=0) if masks else np.ones_like(y_pred, dtype=np.float32)
    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "prior": prior,
        "mask": mask,
        "domains": np.asarray(domains),
        "metrics": round_metrics(regression_metrics(y_true, y_pred, mask)),
    }


def _poly_features(pred: np.ndarray, prior: np.ndarray, degree: int) -> np.ndarray:
    pred = pred.reshape(-1).astype(np.float32)
    prior = prior.reshape(-1).astype(np.float32)
    cols = [np.ones_like(pred), pred, prior]
    if degree >= 2:
        cols.extend([pred**2, pred * prior, prior**2])
    if degree >= 3:
        cols.extend([pred**3, (pred**2) * prior, pred * (prior**2), prior**3])
    return np.column_stack(cols)


def fit_calibration(mode: str, y_true: np.ndarray, y_pred: np.ndarray, prior: np.ndarray, mask: np.ndarray) -> dict[str, object] | None:
    if mode == "none":
        return None
    if mode == "quadratic_prior":
        degree, reg, relative = 2, 1e-4, False
    elif mode == "cubic_prior_relative":
        degree, reg, relative = 3, 1e-7, True
    else:
        raise ValueError(f"unsupported calibration_mode: {mode}")
    valid = mask.astype(bool)
    x = _poly_features(y_pred[valid], prior[valid], degree)
    y = y_true[valid].astype(np.float32).reshape(-1)
    if relative:
        weights = 1.0 / np.clip(np.abs(y), 0.02, None) ** 2.0
        x = x * np.sqrt(weights)[:, None]
        y = y * np.sqrt(weights)
    coef = np.linalg.solve(x.T @ x + reg * np.eye(x.shape[1], dtype=np.float32), x.T @ y)
    return {"mode": mode, "degree": degree, "coef": coef.astype(np.float32).tolist()}


def apply_calibration(calibration: dict[str, object] | None, y_pred: np.ndarray, prior: np.ndarray) -> np.ndarray:
    if calibration is None:
        return y_pred
    coef = np.asarray(calibration["coef"], dtype=np.float32)
    features = _poly_features(y_pred, prior, int(calibration["degree"]))
    calibrated = features @ coef
    return np.clip(calibrated.reshape(y_pred.shape), 0.0, 0.5)


def train_model(cfg: TrainConfig) -> dict[str, object]:
    seed_everything(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)

    train_ds = PorosityNPZDataset(Path(cfg.data_root) / "train.npz")
    val_ds = PorosityNPZDataset(Path(cfg.data_root) / "val.npz")
    test_ds = PorosityNPZDataset(Path(cfg.data_root) / "test.npz")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)

    model = make_model(train_ds, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_state = None
    best_rmse = float("inf")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            target = batch["porosity"].to(device)
            structure = batch["structure"].to(device)
            mask = batch["supervised_mask"].to(device) if "supervised_mask" in batch else torch.ones_like(target)
            out = forward_batch(model, batch, device)
            aux = out.aux

            sup = masked_l1(out.porosity, target, mask)
            trend_loss = masked_l1(aux["base_trend"], target, mask)
            prop_loss = masked_l1(aux["anchor_curve"], target, mask)
            residual_sparse = (torch.abs(aux["gain"] * aux["residual"]) * mask).sum() / (mask.sum() + 1e-6)
            target_weights = component_target_distribution(structure, aux["anchor_mask"], aux["weights"].shape[1])
            weight_loss = masked_l1(aux["weights"], target_weights, mask)
            anchor_loss = masked_l1(out.porosity, target, mask)
            smooth_loss = smoothness_loss(out.porosity, structure)

            loss = (
                sup
                + 0.05 * trend_loss
                + 0.10 * prop_loss
                + 0.08 * residual_sparse
                + 0.08 * weight_loss
                + 0.16 * anchor_loss
                + 0.04 * smooth_loss
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_result = evaluate(model, val_loader, device)
        val_metrics = val_result["metrics"]
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 6), **val_metrics})
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"epoch={epoch:03d} loss={np.mean(losses):.6f} val_rmse={val_metrics['rmse']:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    val_result = evaluate(model, val_loader, device)
    calibration = fit_calibration(
        cfg.calibration_mode,
        val_result["y_true"],
        val_result["y_pred"],
        val_result["prior"],
        val_result["mask"],
    )
    test_result = evaluate(model, test_loader, device)
    y_pred = apply_calibration(calibration, test_result["y_pred"], test_result["prior"])
    metrics = round_metrics(regression_metrics(test_result["y_true"], y_pred, test_result["mask"]))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "calibration": calibration,
            "n_params": count_parameters(model),
        },
        out_dir / "geoanchor_net.pt",
    )
    np.savez_compressed(
        out_dir / "predictions.npz",
        y_true=test_result["y_true"],
        y_pred=y_pred,
        y_pred_raw=test_result["y_pred"],
        prior=test_result["prior"],
        eval_mask=test_result["mask"],
        domains=test_result["domains"],
    )
    result = {
        "model": "GeoAnchor-Net",
        "n_params": count_parameters(model),
        "metrics": metrics,
        "calibration": calibration,
        "config": asdict(cfg),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def test_model(checkpoint: str | Path, data_root: str | Path, out_dir: str | Path, batch_size: int = 4, device: str = "cuda") -> dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_device = resolve_device(device)
    test_ds = PorosityNPZDataset(Path(data_root) / "test.npz")
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    ckpt = torch.load(checkpoint, map_location=torch_device)
    cfg_dict = ckpt.get("config", {})
    cfg = TrainConfig(data_root=str(data_root), out_dir=str(out_dir), **{k: v for k, v in cfg_dict.items() if k in TrainConfig.__dataclass_fields__ and k not in {"data_root", "out_dir"}})
    model = make_model(test_ds, cfg).to(torch_device)
    model.load_state_dict(ckpt["state_dict"])
    result = evaluate(model, loader, torch_device)
    y_pred = apply_calibration(ckpt.get("calibration"), result["y_pred"], result["prior"])
    metrics = round_metrics(regression_metrics(result["y_true"], y_pred, result["mask"]))
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        y_true=result["y_true"],
        y_pred=y_pred,
        y_pred_raw=result["y_pred"],
        prior=result["prior"],
        eval_mask=result["mask"],
        domains=result["domains"],
    )
    payload = {"model": "GeoAnchor-Net", "metrics": metrics, "checkpoint": str(checkpoint)}
    (out_dir / "test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
