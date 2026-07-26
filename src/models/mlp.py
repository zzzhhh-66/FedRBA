"""MLP with an explicit backbone/classifier split."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class MLPClassifier(nn.Module):
    """Feature-extracting MLP followed by one class-addressable linear layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least 2.")
        if not hidden_dims or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive integers.")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not 0.0 <= float(dropout) < 1.0
        ):
            raise ValueError("dropout must lie in [0, 1).")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous_dim = hidden_dim

        self.input_dim = input_dim
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous_dim, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Flatten non-batch dimensions and return unnormalized class logits."""

        flattened = inputs.flatten(start_dim=1)
        if flattened.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} flattened features, "
                f"received {flattened.shape[1]}."
            )
        features = self.backbone(flattened)
        return self.classifier(features)

