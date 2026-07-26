"""Evaluate a checkpoint or summarize completed multi-seed experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.data import build_dataset
from src.federated.server import resolve_device
from src.logging.experiment_logger import json_safe
from src.metrics.classification import evaluate_model
from src.models import MLPClassifier


SUMMARY_FIELDS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "minority_recall",
    "roc_auc",
    "pr_auc",
]


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate_checkpoint(
    config_path: str, checkpoint_path: str, device_name: str
) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"])
    bundle = build_dataset(config["dataset"], seed=seed)
    checkpoint = _torch_load(Path(checkpoint_path))
    model = MLPClassifier(
        bundle.input_dim,
        config["model"]["hidden_dims"],
        bundle.num_classes,
        float(config["model"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    metrics, _, _ = evaluate_model(
        model,
        bundle.test_dataset,
        device=resolve_device(device_name),
        batch_size=int(config["federated"]["batch_size"]),
        num_classes=bundle.num_classes,
        minority_class=bundle.minority_class,
        num_workers=int(config.get("runtime", {}).get("num_workers", 0)),
        pin_memory=bool(config.get("runtime", {}).get("pin_memory", False)),
    )
    print(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2))
    return metrics


def summarize(root: str, dataset: str) -> list[dict[str, Any]]:
    dataset_root = Path(root) / dataset
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in dataset_root.glob("**/seed_*/final_metrics.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        key = (
            str(value.get("method", value["algorithm"])),
            str(value.get("scenario", "default")),
        )
        grouped.setdefault(key, []).append(value)
    if not grouped:
        raise FileNotFoundError(f"No final_metrics.json files under {dataset_root}.")
    rows: list[dict[str, Any]] = []
    for (method, scenario), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "method": method,
            "scenario": scenario,
            "runs": len(values),
        }
        for field in SUMMARY_FIELDS:
            array = np.asarray([item[field] for item in values], dtype=np.float64)
            row[f"{field}_mean"] = float(np.nanmean(array))
            row[f"{field}_std"] = (
                float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0
            )
            row[field] = f"{row[f'{field}_mean']:.4f}±{row[f'{field}_std']:.4f}"
        rows.append(row)
    csv_path = dataset_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (dataset_root / "summary.json").write_text(
        json.dumps(json_safe(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_method_comparisons(dataset_root)
    print(f"Wrote {csv_path}")
    return rows


def _plot_method_comparisons(dataset_root: Path) -> None:
    """Plot mean±std convergence curves across completed seeds."""

    grouped: dict[str, dict[str, list[list[dict[str, str]]]]] = {}
    for metrics_path in dataset_root.glob("**/seed_*/metrics.csv"):
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        if records:
            scenario = metrics_path.parents[1].name
            method = metrics_path.parents[2].name
            grouped.setdefault(scenario, {}).setdefault(method, []).append(records)
    for scenario, methods in grouped.items():
        for field, ylabel, filename in (
        ("accuracy", "Accuracy", "comparison_accuracy_vs_round.png"),
        ("macro_f1", "Macro-F1", "comparison_macro_f1_vs_round.png"),
        (
            "minority_recall",
            "Minority / worst-class recall",
            "comparison_minority_recall_vs_round.png",
        ),
        ):
            figure, axis = plt.subplots(figsize=(7, 4.8))
            for method, runs in sorted(methods.items()):
                common_length = min(len(run) for run in runs)
                values = np.asarray(
                    [
                        [float(record[field]) for record in run[:common_length]]
                        for run in runs
                    ],
                    dtype=np.float64,
                )
                rounds = np.asarray(
                    [int(record["round"]) for record in runs[0][:common_length]]
                )
                mean = values.mean(axis=0)
                std = (
                    values.std(axis=0, ddof=1)
                    if len(values) > 1
                    else np.zeros_like(mean)
                )
                axis.plot(rounds, mean, label=method)
                axis.fill_between(rounds, mean - std, mean + std, alpha=0.16)
            axis.set_xlabel("Communication Round")
            axis.set_ylabel(ylabel)
            axis.set_title(scenario)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            figure.tight_layout()
            stem = Path(filename).stem
            figure.savefig(
                dataset_root / f"{stem}_{scenario}.png", dpi=180
            )
            plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint")
    mode.add_argument("--summarize-root")
    parser.add_argument("--config")
    parser.add_argument("--dataset")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint:
        if not args.config:
            raise SystemExit("--config is required with --checkpoint.")
        evaluate_checkpoint(args.config, args.checkpoint, args.device)
    else:
        if not args.dataset:
            raise SystemExit("--dataset is required with --summarize-root.")
        summarize(args.summarize_root, args.dataset)


if __name__ == "__main__":
    main()
