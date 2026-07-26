"""Classification evaluation with explicit undefined-metric warnings."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader


def _binary_roc_auc(
    targets: NDArray[np.int64], scores: NDArray[np.float64]
) -> float:
    positives = targets == 1
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if n_positive == 0 or n_negative == 0:
        warnings.warn("ROC-AUC is undefined because one class is absent.", stacklevel=2)
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[positives].sum())
    return (
        rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def _average_precision(
    targets: NDArray[np.int64], scores: NDArray[np.float64]
) -> float:
    n_positive = int(targets.sum())
    if n_positive == 0:
        warnings.warn(
            "PR-AUC is undefined because the positive class is absent.", stacklevel=2
        )
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order]
    true_positives = np.cumsum(sorted_targets)
    precision = true_positives / np.arange(1, len(targets) + 1)
    return float(precision[sorted_targets == 1].sum() / n_positive)


def classification_metrics(
    targets: NDArray[np.integer],
    probabilities: NDArray[np.floating],
    *,
    num_classes: int,
    minority_class: int | None,
) -> dict[str, Any]:
    """Compute accuracy, class metrics, macro scores, ROC-AUC, and PR-AUC."""

    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(y), num_classes):
        raise ValueError(
            f"Expected probability shape {(len(y), num_classes)}, "
            f"got {probabilities.shape}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise FloatingPointError("Probabilities contain NaN or Inf.")
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_count > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    valid_classes = support > 0
    if not np.all(valid_classes):
        warnings.warn(
            "Some classes are absent in evaluation targets; macro metrics use "
            "only observed classes.",
            stacklevel=2,
        )
    macro_f1 = float(f1[valid_classes].mean())
    balanced_accuracy = float(recall[valid_classes].mean())

    roc_values: list[float] = []
    pr_values: list[float] = []
    for class_id in range(num_classes):
        binary = (y == class_id).astype(np.int64)
        roc_values.append(_binary_roc_auc(binary, probabilities[:, class_id]))
        pr_values.append(_average_precision(binary, probabilities[:, class_id]))
    if num_classes == 2:
        roc_auc = float(roc_values[1])
        pr_auc = float(pr_values[1])
    else:
        finite_roc = [value for value in roc_values if np.isfinite(value)]
        finite_pr = [value for value in pr_values if np.isfinite(value)]
        roc_auc = float(np.mean(finite_roc)) if finite_roc else float("nan")
        pr_auc = float(np.mean(finite_pr)) if finite_pr else float("nan")
    if minority_class is None:
        minority_recall = float(recall[valid_classes].min())
        minority_f1 = float(f1[valid_classes].min())
    else:
        minority_recall = float(recall[minority_class])
        minority_f1 = float(f1[minority_class])
    return {
        "accuracy": float((predictions == y).mean()),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall_per_class": recall.tolist(),
        "precision_per_class": precision.tolist(),
        "f1_per_class": f1.tolist(),
        "minority_recall": minority_recall,
        "minority_f1": minority_f1,
        "confusion_matrix": confusion.astype(int).tolist(),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int,
    num_classes: int,
    minority_class: int | None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[dict[str, Any], NDArray[np.float64], NDArray[np.int64]]:
    """Evaluate a model and move it back to CPU before returning."""

    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    all_probabilities: list[NDArray[np.float64]] = []
    all_targets: list[NDArray[np.int64]] = []
    for features, labels in loader:
        features = features.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)
        logits = model(features)
        total_loss += float(criterion(logits, labels).detach().cpu())
        all_probabilities.append(
            torch.softmax(logits, dim=1).detach().cpu().double().numpy()
        )
        all_targets.append(labels.detach().cpu().numpy().astype(np.int64))
    model.cpu()
    probabilities = np.concatenate(all_probabilities, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = classification_metrics(
        targets,
        probabilities,
        num_classes=num_classes,
        minority_class=minority_class,
    )
    metrics["test_loss"] = total_loss / max(1, len(targets))
    return metrics, probabilities, targets
