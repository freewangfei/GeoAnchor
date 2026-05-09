from __future__ import annotations

import argparse
import json

from geoanchor import test_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GeoAnchor-Net checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to geoanchor_net.pt.")
    parser.add_argument("--data-root", required=True, help="Directory containing test.npz.")
    parser.add_argument("--out-dir", required=True, help="Directory for test metrics and predictions.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = test_model(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
