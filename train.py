from __future__ import annotations

import argparse
import json

from geoanchor import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GeoAnchor-Net.")
    parser.add_argument("--data-root", required=True, help="Directory containing train.npz, val.npz, and test.npz.")
    parser.add_argument("--out-dir", required=True, help="Directory for checkpoint and metrics.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--calibration-mode", default="none", choices=["none", "quadratic_prior", "cubic_prior_relative"])
    parser.add_argument("--disable-prior-condition", action="store_true")
    parser.add_argument("--anchor-band-strength", type=float, default=1.0)
    parser.add_argument("--curve-gate-strength", type=float, default=1.0)
    parser.add_argument("--curve-gate-mode", default="column", choices=["column", "soft_column", "local", "none"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        calibration_mode=args.calibration_mode,
        use_prior_condition=not args.disable_prior_condition,
        anchor_band_strength=args.anchor_band_strength,
        curve_gate_strength=args.curve_gate_strength,
        curve_gate_mode=args.curve_gate_mode,
    )
    result = train_model(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
