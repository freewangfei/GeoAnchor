from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


_PRIOR_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def _build_rgt_prior_lut(train_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    cache_key = str(train_npz.resolve())
    if cache_key in _PRIOR_CACHE:
        return _PRIOR_CACHE[cache_key]

    train = load_npz(train_npz)
    rgt = train["structure"][:, 0].reshape(-1)
    porosity = train["porosity"][:, 0].reshape(-1)
    bins = np.linspace(0.0, 1.0, 129, dtype=np.float32)
    bin_id = np.digitize(rgt, bins) - 1
    lut = np.zeros(len(bins) - 1, dtype=np.float32)
    global_mean = float(porosity.mean())
    for idx in range(len(lut)):
        mask = bin_id == idx
        lut[idx] = float(porosity[mask].mean()) if mask.any() else global_mean
    _PRIOR_CACHE[cache_key] = (bins, lut)
    return bins, lut


def synthesize_rgt_prior(split_path: Path, structure: np.ndarray) -> np.ndarray:
    bins, lut = _build_rgt_prior_lut(split_path.parent / "train.npz")
    rgt = structure[:, 0]
    bin_id = np.clip(np.digitize(rgt, bins) - 1, 0, len(lut) - 1)
    prior = lut[bin_id][:, None].astype(np.float32)
    return np.clip(prior, 0.0, 0.5)


class PorosityNPZDataset(Dataset):
    def __init__(self, split_path: str | Path):
        self.split_path = Path(split_path)
        data = load_npz(self.split_path)
        self.seismic = data["seismic"].astype(np.float32)
        self.structure = data["structure"].astype(np.float32)
        self.porosity = data["porosity"].astype(np.float32)
        self.domains = data["domains"].astype(str) if "domains" in data else np.array(["default"] * len(self.seismic))
        self.supervised_mask = data["supervised_mask"].astype(np.float32) if "supervised_mask" in data else None
        self.eval_mask = data["eval_mask"].astype(np.float32) if "eval_mask" in data else None
        self.prior = data["prior"].astype(np.float32) if "prior" in data else synthesize_rgt_prior(self.split_path, self.structure)
        self.anchor = self.porosity * self.supervised_mask if self.supervised_mask is not None else np.zeros_like(self.porosity)

    def __len__(self) -> int:
        return int(self.seismic.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        item: dict[str, object] = {
            "seismic": self.seismic[index],
            "structure": self.structure[index],
            "porosity": self.porosity[index],
            "prior": self.prior[index],
            "anchor": self.anchor[index],
            "domain": self.domains[index],
            "index": index,
        }
        if self.supervised_mask is not None:
            item["supervised_mask"] = self.supervised_mask[index]
        if self.eval_mask is not None:
            item["eval_mask"] = self.eval_mask[index]
        return item


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {
        "seismic": torch.from_numpy(np.stack([x["seismic"] for x in batch])),
        "structure": torch.from_numpy(np.stack([x["structure"] for x in batch])),
        "porosity": torch.from_numpy(np.stack([x["porosity"] for x in batch])),
        "prior": torch.from_numpy(np.stack([x["prior"] for x in batch])),
        "anchor": torch.from_numpy(np.stack([x["anchor"] for x in batch])),
        "domain": [str(x["domain"]) for x in batch],
        "index": [int(x["index"]) for x in batch],
    }
    if "supervised_mask" in batch[0]:
        out["supervised_mask"] = torch.from_numpy(np.stack([x["supervised_mask"] for x in batch]))
    if "eval_mask" in batch[0]:
        out["eval_mask"] = torch.from_numpy(np.stack([x["eval_mask"] for x in batch]))
    return out
