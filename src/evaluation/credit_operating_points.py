"""Validation-only checkpoint and operating-point selection for credit runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Subset

from src.config import load_config
from src.data import build_dataset
from src.federated.server import resolve_device
from src.logging.experiment_logger import json_safe
from src.metrics.classification import evaluate_model
from src.metrics.operating_points import (
    binary_metrics_at_threshold,
    select_balanced_accuracy_threshold,
    select_thresholds_at_fpr,
    select_thresholds_at_recall,
)
from src.models import MLPClassifier


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_round(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("round_"):
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return int(stem.split("_", maxsplit=1)[1])


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_validation_indices(run_dir: Path, train_size: int) -> list[int]:
    partition_path = run_dir / "client_partitions.json"
    if not partition_path.is_file():
        raise FileNotFoundError(f"Missing partition audit file: {partition_path}")
    value = json.loads(partition_path.read_text(encoding="utf-8"))
    indices = sorted(
        int(index)
        for client in value.values()
        for index in client["validation"]
    )
    if not indices:
        raise ValueError("The federated validation union is empty.")
    if len(indices) != len(set(indices)):
        raise ValueError("Validation indices contain duplicates.")
    if indices[0] < 0 or indices[-1] >= train_size:
        raise ValueError("Validation indices are outside the training dataset.")
    return indices


def _model_from_config(config: dict[str, Any], input_dim: int) -> MLPClassifier:
    return MLPClassifier(
        input_dim=input_dim,
        hidden_dims=config["model"]["hidden_dims"],
        num_classes=int(config["dataset"]["num_classes"]),
        dropout=float(config["model"].get("dropout", 0.2)),
    )


def _evaluate(
    model: torch.nn.Module,
    dataset: Any,
    *,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    runtime = config.get("runtime", {})
    return evaluate_model(
        model,
        dataset,
        device=device,
        batch_size=int(config["federated"]["batch_size"]),
        num_classes=int(config["dataset"]["num_classes"]),
        minority_class=int(config["dataset"]["minority_class"]),
        num_workers=int(runtime.get("num_workers", 0)),
        pin_memory=bool(runtime.get("pin_memory", False)),
    )


def _test_at_validation_thresholds(
    validation_points: Iterable[dict[str, Any]],
    test_targets: np.ndarray,
    test_scores: np.ndarray,
    *,
    target_key: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for validation in validation_points:
        test = binary_metrics_at_threshold(
            test_targets, test_scores, validation["threshold"]
        )
        output.append(
            {
                target_key: validation[target_key],
                "threshold_selected_on_validation": validation["threshold"],
                "selection_rule": validation["selection_rule"],
                "validation": validation,
                "test": test,
            }
        )
    return output


def evaluate_credit_run(
    run_dir: str | Path,
    output_root: str | Path,
    *,
    device_name: str = "auto",
    fpr_targets: Iterable[float] = (0.01, 0.05, 0.10),
    recall_targets: Iterable[float] = (0.50, 0.70, 0.80),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate one completed binary-credit run without test-set selection."""

    source = Path(run_dir)
    config_path = source / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved run config: {config_path}")
    config = load_config(config_path)
    dataset_name = str(config["dataset"]["name"])
    if dataset_name not in {"default_credit", "give_me_some_credit"}:
        raise ValueError("Operating-point analysis is restricted to credit datasets.")
    if int(config["dataset"]["num_classes"]) != 2:
        raise ValueError("Operating-point analysis requires a binary dataset.")
    seed = int(config["experiment"]["seed"])
    method = source.parents[1].name
    scenario = source.parents[0].name
    directory_dataset = source.parents[2].name
    if directory_dataset != dataset_name:
        raise ValueError("Run-directory dataset and config dataset do not match.")
    if source.name != f"seed_{seed}":
        raise ValueError("Run-directory seed and config seed do not match.")

    destination = (
        Path(output_root) / dataset_name / method / scenario / source.name
    )
    result_path = destination / "operating_point_metrics.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(
            f"Result already exists: {result_path}. Use --overwrite explicitly."
        )
    destination.mkdir(parents=True, exist_ok=True)

    bundle = build_dataset(config["dataset"], seed=seed)
    if bundle.num_classes != 2 or bundle.minority_class != 1:
        raise ValueError("Credit protocol requires positive/minority class id 1.")
    validation_indices = _load_validation_indices(source, len(bundle.train_dataset))
    validation_dataset = Subset(bundle.train_dataset, validation_indices)
    validation_targets_expected = bundle.train_targets[
        np.asarray(validation_indices, dtype=np.int64)
    ]
    if len(np.unique(validation_targets_expected)) != 2:
        raise ValueError("Both classes must be present in validation data.")

    checkpoint_paths = sorted(
        (source / "checkpoints").glob("round_*.pt"),
        key=_checkpoint_round,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No periodic checkpoints under {source}.")
    device = resolve_device(device_name)
    model = _model_from_config(config, bundle.input_dim)
    candidate_rows: list[dict[str, Any]] = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = _torch_load(checkpoint_path)
        file_round = _checkpoint_round(checkpoint_path)
        if int(checkpoint["round"]) != file_round:
            raise ValueError(
                f"Checkpoint round mismatch in {checkpoint_path}: "
                f"payload={checkpoint['round']}, filename={file_round}."
            )
        if checkpoint.get("dataset_name") not in (None, dataset_name):
            raise ValueError(
                f"Checkpoint dataset mismatch in {checkpoint_path}."
            )
        if checkpoint.get("algorithm") not in (None, method):
            raise ValueError(
                f"Checkpoint algorithm mismatch in {checkpoint_path}."
            )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        validation_metrics, _, validation_targets = _evaluate(
            model, validation_dataset, config=config, device=device
        )
        if not np.array_equal(validation_targets, validation_targets_expected):
            raise RuntimeError("Validation target reconstruction is inconsistent.")
        candidate_rows.append(
            {
                "round": int(checkpoint["round"]),
                "checkpoint": str(checkpoint_path),
                "validation_loss": validation_metrics["test_loss"],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_pr_auc": validation_metrics["pr_auc"],
            }
        )

    selected_row = min(
        candidate_rows,
        key=lambda item: (-item["validation_pr_auc"], item["round"]),
    )
    selected_checkpoint = _torch_load(Path(selected_row["checkpoint"]))
    model.load_state_dict(selected_checkpoint["model_state"], strict=True)
    validation_metrics, validation_probabilities, validation_targets = _evaluate(
        model, validation_dataset, config=config, device=device
    )

    # Test data is first touched only after all checkpoint selection is complete.
    test_metrics_argmax, test_probabilities, test_targets = _evaluate(
        model, bundle.test_dataset, config=config, device=device
    )
    validation_scores = validation_probabilities[:, 1]
    test_scores = test_probabilities[:, 1]

    validation_balanced = select_balanced_accuracy_threshold(
        validation_targets, validation_scores
    )
    test_balanced = binary_metrics_at_threshold(
        test_targets, test_scores, validation_balanced["threshold"]
    )
    validation_fpr_points = select_thresholds_at_fpr(
        validation_targets, validation_scores, fpr_targets
    )
    validation_recall_points = select_thresholds_at_recall(
        validation_targets, validation_scores, recall_targets
    )

    result = {
        "protocol": {
            "checkpoint_selection_source": (
                "union of per-client validation subsets from training data"
            ),
            "checkpoint_selection_metric": "validation_pr_auc",
            "checkpoint_tie_break": "earliest checkpoint round",
            "threshold_selection_source": "same training-data validation union",
            "primary_threshold_rule": (
                "maximize validation balanced_accuracy; ties minimize validation "
                "FPR then maximize threshold"
            ),
            "test_selection_prohibited": True,
            "test_evaluations_after_selection": 1,
            "candidate_checkpoints": "saved periodic checkpoints only",
            "validation_reuse_disclosure": (
                "FedRBA uses aggregate client validation statistics as its "
                "blindspot control signal; no validation example is used for "
                "gradient updates, and every method uses the identical validation "
                "union for this post-training protocol."
            ),
        },
        "run": {
            "source_run_dir": str(source),
            "dataset": dataset_name,
            "method": method,
            "scenario": scenario,
            "seed": seed,
        },
        "sample_counts": {
            "validation": int(len(validation_targets)),
            "validation_positive": int(validation_targets.sum()),
            "test": int(len(test_targets)),
            "test_positive": int(test_targets.sum()),
        },
        "split_audit": {
            "validation_indices_sha256": _array_sha256(
                np.asarray(validation_indices, dtype=np.int64)
            ),
            "validation_targets_sha256": _array_sha256(validation_targets),
            "test_targets_sha256": _array_sha256(test_targets),
        },
        "checkpoint_candidates": candidate_rows,
        "selected_checkpoint": selected_row,
        "validation_ranking_metrics": {
            "roc_auc": validation_metrics["roc_auc"],
            "pr_auc": validation_metrics["pr_auc"],
        },
        "test_ranking_metrics": {
            "roc_auc": test_metrics_argmax["roc_auc"],
            "pr_auc": test_metrics_argmax["pr_auc"],
        },
        "argmax_test": test_metrics_argmax,
        "balanced_accuracy_operating_point": {
            "threshold_selected_on_validation": validation_balanced["threshold"],
            "selection_rule": validation_balanced["selection_rule"],
            "validation": validation_balanced,
            "test": test_balanced,
        },
        "fixed_fpr_operating_points": _test_at_validation_thresholds(
            validation_fpr_points,
            test_targets,
            test_scores,
            target_key="target_fpr",
        ),
        "fixed_recall_operating_points": _test_at_validation_thresholds(
            validation_recall_points,
            test_targets,
            test_scores,
            target_key="target_recall",
        ),
    }
    result_path.write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    candidate_path = destination / "checkpoint_validation_scores.csv"
    with candidate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    return result
