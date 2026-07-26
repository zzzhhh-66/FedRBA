"""Dynamic class-blindspot estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class ClassStatistics:
    """Privacy-preserving validation aggregates reported by one client."""

    count: NDArray[np.int64]
    loss_sum: NDArray[np.float64]
    correct: NDArray[np.int64]

    def __post_init__(self) -> None:
        self.count = np.asarray(self.count, dtype=np.int64)
        self.loss_sum = np.asarray(self.loss_sum, dtype=np.float64)
        self.correct = np.asarray(self.correct, dtype=np.int64)
        if not (
            self.count.ndim == self.loss_sum.ndim == self.correct.ndim == 1
            and self.count.shape == self.loss_sum.shape == self.correct.shape
        ):
            raise ValueError("Class-statistic arrays must be same-length 1-D arrays.")
        if np.any(self.count < 0) or np.any(self.correct < 0):
            raise ValueError("Counts and correct predictions must be non-negative.")
        if np.any(self.correct > self.count):
            raise ValueError("Correct predictions cannot exceed class counts.")
        if np.any(~np.isfinite(self.loss_sum)) or np.any(self.loss_sum < 0):
            raise ValueError("Class loss sums must be finite and non-negative.")


def blindspot_class_weights(
    blindspot_weights: NDArray[np.floating], strength: float
) -> NDArray[np.float32]:
    """Convert normalized blindspots to mean-one local loss weights."""

    q = np.asarray(blindspot_weights, dtype=np.float64)
    if q.ndim != 1 or q.size < 2 or np.any(q < 0) or not np.all(np.isfinite(q)):
        raise ValueError("blindspot_weights must be a finite non-negative vector.")
    total = float(q.sum())
    if total <= 0:
        raise ValueError("blindspot_weights must have positive sum.")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must lie in [0, 1].")
    q = q / total
    classes = q.size
    weights = classes * ((1.0 - strength) / classes + strength * q)
    return weights.astype(np.float32)


def update_blindspot_weights(
    current: NDArray[np.floating],
    statistics: Sequence[ClassStatistics],
    *,
    temperature: float,
    ema: float,
    epsilon: float = 1e-12,
) -> tuple[NDArray[np.float64], dict[str, list[float]]]:
    """Update class blindspots from selected clients' validation statistics."""

    q = np.asarray(current, dtype=np.float64)
    if q.ndim != 1 or q.size < 2 or np.any(q < 0) or not np.all(np.isfinite(q)):
        raise ValueError("current blindspot weights are invalid.")
    if not statistics:
        raise ValueError("At least one client statistic is required.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    if not 0.0 <= ema < 1.0:
        raise ValueError("ema must lie in [0, 1).")
    if any(item.count.shape != q.shape for item in statistics):
        raise ValueError("Statistic class count does not match blindspot vector.")

    total_count = np.sum([item.count for item in statistics], axis=0).astype(
        np.float64
    )
    total_correct = np.sum([item.correct for item in statistics], axis=0).astype(
        np.float64
    )
    total_loss = np.sum([item.loss_sum for item in statistics], axis=0).astype(
        np.float64
    )
    accuracy = total_correct / np.maximum(total_count, epsilon)
    error = 1.0 - accuracy
    logits = error / float(temperature)
    logits -= float(np.max(logits))
    raw = np.exp(logits)
    raw /= max(float(raw.sum()), epsilon)
    normalized_current = q / max(float(q.sum()), epsilon)
    updated = ema * normalized_current + (1.0 - ema) * raw
    updated = np.maximum(updated, 0.0)
    updated /= max(float(updated.sum()), epsilon)
    if not np.all(np.isfinite(updated)):
        raise FloatingPointError("Blindspot update produced NaN or Inf.")
    details = {
        "class_count": total_count.tolist(),
        "class_loss": (total_loss / np.maximum(total_count, epsilon)).tolist(),
        "class_accuracy": accuracy.tolist(),
        "class_error": error.tolist(),
        "raw_blindspot": raw.tolist(),
    }
    return updated, details

