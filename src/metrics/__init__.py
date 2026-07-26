"""Classification and class-sensitive metrics."""

from .classification import classification_metrics, evaluate_model
from .fairness import recall_gap
from .operating_points import (
    binary_metrics_at_threshold,
    select_balanced_accuracy_threshold,
    select_thresholds_at_fpr,
    select_thresholds_at_recall,
)

__all__ = [
    "binary_metrics_at_threshold",
    "classification_metrics",
    "evaluate_model",
    "recall_gap",
    "select_balanced_accuracy_threshold",
    "select_thresholds_at_fpr",
    "select_thresholds_at_recall",
]
