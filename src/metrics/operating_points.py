"""Auditable binary threshold selection without test-set tuning."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_binary_inputs(
    targets: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    probability = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.shape != probability.shape:
        raise ValueError("targets and scores must be non-empty arrays of equal size.")
    if not np.all(np.isfinite(probability)):
        raise ValueError("scores contain NaN or Inf.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("targets must contain binary labels 0 and 1 only.")
    if len(np.unique(y)) != 2:
        raise ValueError("both binary classes must be present.")
    return y, probability


def binary_metrics_at_threshold(
    targets: ArrayLike,
    scores: ArrayLike,
    threshold: float,
) -> dict[str, Any]:
    """Return deterministic binary metrics for ``score >= threshold``."""

    y, probability = _validate_binary_inputs(targets, scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    predicted = probability >= float(threshold)
    positive = y == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & negative))
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(1.0 - specificity),
        "fnr": float(1.0 - recall),
        "f1": float(f1),
        "predicted_positive_rate": float((tp + fp) / len(y)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _threshold_sweep(
    targets: ArrayLike, scores: ArrayLike
) -> list[dict[str, Any]]:
    """Evaluate all distinct score thresholds plus the all-negative endpoint."""

    y, probability = _validate_binary_inputs(targets, scores)
    order = np.argsort(-probability, kind="mergesort")
    sorted_scores = probability[order]
    sorted_targets = y[order]
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    cumulative_positive = np.cumsum(sorted_targets)
    records = [
        binary_metrics_at_threshold(
            y, probability, float(np.nextafter(sorted_scores[0], np.inf))
        )
    ]
    for end in group_ends:
        threshold = float(sorted_scores[int(end)])
        included = int(end) + 1
        tp = int(cumulative_positive[int(end)])
        fp = included - tp
        positives = int(cumulative_positive[-1])
        negatives = len(y) - positives
        fn = positives - tp
        tn = negatives - fp
        recall = tp / positives
        specificity = tn / negatives
        precision = tp / max(1, tp + fp)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        records.append(
            {
                "threshold": threshold,
                "accuracy": float((tp + tn) / len(y)),
                "balanced_accuracy": float((recall + specificity) / 2.0),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "fpr": float(1.0 - specificity),
                "fnr": float(1.0 - recall),
                "f1": float(f1),
                "predicted_positive_rate": float((tp + fp) / len(y)),
                "confusion_matrix": [[tn, fp], [fn, tp]],
            }
        )
    return records


def select_balanced_accuracy_threshold(
    targets: ArrayLike, scores: ArrayLike
) -> dict[str, Any]:
    """Select a threshold using validation balanced accuracy only.

    Ties prefer the lower false-positive rate and then the higher threshold.
    """

    records = _threshold_sweep(targets, scores)
    best = max(
        records,
        key=lambda item: (
            item["balanced_accuracy"],
            -item["fpr"],
            item["threshold"],
        ),
    )
    return {
        "selection_rule": (
            "maximize validation balanced_accuracy; ties minimize validation "
            "FPR then maximize threshold"
        ),
        **best,
    }


def _validate_targets(values: Iterable[float], *, name: str) -> list[float]:
    targets = [float(value) for value in values]
    if not targets:
        raise ValueError(f"{name} cannot be empty.")
    if any(not np.isfinite(value) or not 0.0 < value < 1.0 for value in targets):
        raise ValueError(f"{name} values must lie strictly in (0, 1).")
    if len(set(targets)) != len(targets):
        raise ValueError(f"{name} values must be unique.")
    return sorted(targets)


def select_thresholds_at_fpr(
    targets: ArrayLike,
    scores: ArrayLike,
    fpr_targets: Iterable[float],
) -> list[dict[str, Any]]:
    """Select validation thresholds under fixed validation-FPR constraints."""

    requested = _validate_targets(fpr_targets, name="fpr_targets")
    records = _threshold_sweep(targets, scores)
    output: list[dict[str, Any]] = []
    for target in requested:
        feasible = [item for item in records if item["fpr"] <= target + 1e-12]
        best = max(
            feasible,
            key=lambda item: (
                item["recall"],
                item["precision"],
                -item["fpr"],
                item["threshold"],
            ),
        )
        output.append(
            {
                "target_fpr": target,
                "selection_rule": (
                    "among validation thresholds with FPR <= target, maximize "
                    "validation recall; ties maximize precision, minimize FPR, "
                    "then maximize threshold"
                ),
                **best,
            }
        )
    return output


def select_thresholds_at_recall(
    targets: ArrayLike,
    scores: ArrayLike,
    recall_targets: Iterable[float],
) -> list[dict[str, Any]]:
    """Select validation thresholds under fixed validation-recall constraints."""

    requested = _validate_targets(recall_targets, name="recall_targets")
    records = _threshold_sweep(targets, scores)
    output: list[dict[str, Any]] = []
    for target in requested:
        feasible = [item for item in records if item["recall"] >= target - 1e-12]
        best = max(
            feasible,
            key=lambda item: (
                item["precision"],
                -item["fpr"],
                item["threshold"],
            ),
        )
        output.append(
            {
                "target_recall": target,
                "selection_rule": (
                    "among validation thresholds with recall >= target, maximize "
                    "validation precision; ties minimize FPR then maximize threshold"
                ),
                **best,
            }
        )
    return output
