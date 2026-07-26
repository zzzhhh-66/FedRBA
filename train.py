"""Command-line entry point for one FedRBA-Lite experiment."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.federated.fedrba import SUPPORTED_ALGORITHMS
from src.federated.server import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=sorted(SUPPORTED_ALGORITHMS),
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted YAML overrides, for example federated.rounds=2.",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint path.")
    parser.add_argument(
        "--device",
        default=None,
        help="Override runtime.device (auto, cpu, cuda, cuda:0, ...).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing result directory for this dataset/algorithm/seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override)
    run_experiment(
        config,
        algorithm=args.algorithm,
        resume=args.resume,
        device_override=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

