"""Run leakage-safe checkpoint and operating-point analysis for one credit run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_credit_run  # noqa: E402
from src.logging.experiment_logger import json_safe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-root", default="operating_point_outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--fpr-targets",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.10],
    )
    parser.add_argument(
        "--recall-targets",
        nargs="+",
        type=float,
        default=[0.50, 0.70, 0.80],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_credit_run(
        args.run_dir,
        args.output_root,
        device_name=args.device,
        fpr_targets=args.fpr_targets,
        recall_targets=args.recall_targets,
        overwrite=bool(args.overwrite),
    )
    run = result["run"]
    selected = result["selected_checkpoint"]
    threshold = result["balanced_accuracy_operating_point"]
    print(
        f"completed dataset={run['dataset']} method={run['method']} "
        f"scenario={run['scenario']} seed={run['seed']}"
    )
    print(
        f"selected_round={selected['round']} "
        f"validation_pr_auc={selected['validation_pr_auc']:.6f} "
        f"threshold={threshold['threshold_selected_on_validation']:.8f}"
    )
    print(
        json.dumps(
            json_safe(
                {
                    "test_ranking_metrics": result["test_ranking_metrics"],
                    "test_balanced_operating_point": threshold["test"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
