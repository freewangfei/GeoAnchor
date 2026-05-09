from __future__ import annotations

import json
from pathlib import Path

from geoanchor import TrainConfig, train_model


EXPERIMENTS = [
    {
        "name": "openporobench_s",
        "data_root": "data/openporobench_s",
        "out_dir": "outputs/openporobench_s",
        "epochs": 44,
        "batch_size": 8,
        "lr": 1.2e-4,
        "seed": 20260446,
        "calibration_mode": "quadratic_prior",
        "use_prior_condition": False,
        "anchor_band_strength": 0.65,
        "curve_gate_strength": 0.65,
    },
    {
        "name": "f3_demo_2023",
        "data_root": "data/external/f3_demo_2023/openporo_npz",
        "out_dir": "outputs/f3_demo_2023",
        "epochs": 20,
        "batch_size": 2,
        "lr": 7e-4,
        "seed": 20260431,
        "calibration_mode": "none",
        "use_prior_condition": True,
        "anchor_band_strength": 1.0,
        "curve_gate_strength": 1.0,
    },
    {
        "name": "seis2rock_dense",
        "data_root": "data/external/seis2rock_dense_openporo",
        "out_dir": "outputs/seis2rock_dense",
        "epochs": 26,
        "batch_size": 4,
        "lr": 3e-4,
        "seed": 20260547,
        "calibration_mode": "cubic_prior_relative",
        "use_prior_condition": False,
        "anchor_band_strength": 0.65,
        "curve_gate_strength": 0.65,
    },
    {
        "name": "seis2rock_aux",
        "data_root": "data/external/seis2rock_aux_openporo",
        "out_dir": "outputs/seis2rock_aux",
        "epochs": 24,
        "batch_size": 4,
        "lr": 2e-4,
        "seed": 20260727,
        "calibration_mode": "cubic_prior_relative",
        "use_prior_condition": True,
        "anchor_band_strength": 1.0,
        "curve_gate_strength": 1.0,
    },
]


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = []
    for item in EXPERIMENTS:
        cfg = TrainConfig(
            data_root=str(root / item["data_root"]),
            out_dir=str(root / item["out_dir"]),
            epochs=item["epochs"],
            batch_size=item["batch_size"],
            lr=item["lr"],
            seed=item["seed"],
            calibration_mode=item["calibration_mode"],
            use_prior_condition=item["use_prior_condition"],
            anchor_band_strength=item["anchor_band_strength"],
            curve_gate_strength=item["curve_gate_strength"],
        )
        result = train_model(cfg)
        rows.append({"dataset": item["name"], **result["metrics"]})
    out_path = root / "outputs" / "reproduction_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
