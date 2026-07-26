"""Small helpers for class-sensitive reporting."""

from __future__ import annotations

from typing import Sequence


def recall_gap(recall_per_class: Sequence[float]) -> float:
    """Return best-minus-worst class recall."""

    values = [float(value) for value in recall_per_class]
    if not values:
        raise ValueError("recall_per_class cannot be empty.")
    return max(values) - min(values)

