"""FedProx local regularization."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


def proximal_penalty(
    model: nn.Module, global_parameters: Mapping[str, Tensor]
) -> Tensor:
    """Return ``sum ||local-global||²`` over trainable parameters."""

    penalty: Tensor | None = None
    for name, parameter in model.named_parameters():
        if name not in global_parameters:
            raise KeyError(f"Missing global parameter {name!r}.")
        term = torch.sum((parameter - global_parameters[name]) ** 2)
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        raise ValueError("Model has no trainable parameters.")
    return penalty

