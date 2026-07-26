"""Shared dataset representations without eager PyTorch imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ArrayDataset:
    """A minimal NumPy-backed dataset compatible with ``torch.DataLoader``."""

    def __init__(self, features: NDArray[np.floating], targets: NDArray[np.integer]):
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.int64)
        if x.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {x.shape}.")
        if y.ndim != 1 or len(x) != len(y):
            raise ValueError("targets must be 1-D and match the feature row count.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Preprocessed features contain NaN or Inf.")
        self.features = x
        self.targets = y

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        return torch.from_numpy(self.features[index]), torch.tensor(
            int(self.targets[index]), dtype=torch.long
        )


@dataclass
class DatasetBundle:
    """Uniform result returned by every dataset loader."""

    train_dataset: Any
    test_dataset: Any
    train_targets: NDArray[np.int64]
    test_targets: NDArray[np.int64]
    input_dim: int
    num_classes: int
    minority_class: int | None
    feature_names: list[str] | None
    class_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.train_targets = np.asarray(self.train_targets, dtype=np.int64)
        self.test_targets = np.asarray(self.test_targets, dtype=np.int64)
        if len(self.train_dataset) != len(self.train_targets):
            raise ValueError("train_dataset and train_targets lengths differ.")
        if len(self.test_dataset) != len(self.test_targets):
            raise ValueError("test_dataset and test_targets lengths differ.")
        if self.input_dim <= 0 or self.num_classes < 2:
            raise ValueError("input_dim must be positive and num_classes >= 2.")
        allowed = set(range(self.num_classes))
        observed = set(np.unique(np.concatenate([self.train_targets, self.test_targets])))
        if not observed.issubset(allowed):
            raise ValueError(
                f"Targets {sorted(observed)} are outside [0, {self.num_classes - 1}]."
            )

