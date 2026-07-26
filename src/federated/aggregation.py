"""Numerically stable FedAvg and FedRBA model aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .client import ClientUpdate


@dataclass
class AggregationReport:
    """Normalized client and class-wise weights for experiment auditing."""

    backbone_weights: dict[int, float]
    classwise_weights: dict[int, list[float]]


def stable_cosine(left: Tensor, right: Tensor, epsilon: float = 1e-12) -> float:
    """Cosine similarity with defined behavior for zero vectors."""

    a = left.detach().reshape(-1).double()
    b = right.detach().reshape(-1).double()
    norm_a = float(torch.linalg.vector_norm(a))
    norm_b = float(torch.linalg.vector_norm(b))
    if norm_a <= epsilon and norm_b <= epsilon:
        return 1.0
    if norm_a <= epsilon or norm_b <= epsilon:
        return 0.0
    value = float(torch.dot(a, b) / (norm_a * norm_b))
    if not np.isfinite(value):
        raise FloatingPointError("Cosine similarity produced NaN or Inf.")
    return float(np.clip(value, -1.0, 1.0))


def _validate_updates(
    global_state: Mapping[str, Tensor], updates: Sequence[ClientUpdate]
) -> None:
    if not updates:
        raise ValueError("At least one client update is required.")
    expected = set(global_state)
    for update in updates:
        if set(update.delta) != expected:
            raise ValueError(f"Client {update.client_id} state keys do not match.")
        if update.train_sample_count <= 0:
            raise ValueError("Client train sample counts must be positive.")
        for key, global_tensor in global_state.items():
            if update.delta[key].shape != global_tensor.shape:
                raise ValueError(
                    f"Client {update.client_id} shape mismatch for {key!r}."
                )


def sample_count_weights(updates: Sequence[ClientUpdate]) -> NDArray[np.float64]:
    """Return normalized FedAvg weights."""

    counts = np.asarray([item.train_sample_count for item in updates], dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("Total client sample count must be positive.")
    return counts / total


def aggregate_fedavg(
    global_state: Mapping[str, Tensor],
    updates: Sequence[ClientUpdate],
    *,
    server_lr: float = 1.0,
) -> tuple[dict[str, Tensor], AggregationReport]:
    """Aggregate every floating state tensor by client train sample count."""

    _validate_updates(global_state, updates)
    weights = sample_count_weights(updates)
    result = {key: tensor.detach().clone() for key, tensor in global_state.items()}
    for key, original in global_state.items():
        if not torch.is_floating_point(original):
            continue
        aggregate = torch.zeros_like(original)
        for weight, update in zip(weights, updates):
            aggregate.add_(update.delta[key].to(aggregate.dtype), alpha=float(weight))
        result[key] = original + float(server_lr) * aggregate
    report = AggregationReport(
        backbone_weights={
            update.client_id: float(weight)
            for update, weight in zip(updates, weights)
        },
        classwise_weights={
            update.client_id: [float(weight)] * int(update.class_train_counts.size)
            for update, weight in zip(updates, weights)
        },
    )
    return result, report


def _flatten_backbone(update: ClientUpdate) -> Tensor:
    tensors = [
        value.reshape(-1)
        for key, value in update.delta.items()
        if not key.startswith("classifier.") and torch.is_floating_point(value)
    ]
    if not tensors:
        raise ValueError("No floating backbone tensors were found.")
    return torch.cat(tensors)


def backbone_aggregation_weights(
    updates: Sequence[ClientUpdate],
    *,
    alignment_min: float,
    alignment_power: float,
    use_alignment: bool,
) -> NDArray[np.float64]:
    """Compute normalized sample-count and direction-aware backbone weights."""

    counts = np.asarray([item.train_sample_count for item in updates], dtype=np.float64)
    vectors = [_flatten_backbone(item) for item in updates]
    reference = torch.zeros_like(vectors[0])
    base = counts / float(counts.sum())
    for weight, vector in zip(base, vectors):
        reference.add_(vector, alpha=float(weight))
    alignments = np.ones(len(updates), dtype=np.float64)
    if use_alignment:
        for index, vector in enumerate(vectors):
            mapped = (stable_cosine(vector, reference) + 1.0) / 2.0
            alignments[index] = np.clip(mapped, alignment_min, 1.0)
    raw = counts * np.power(alignments, float(alignment_power))
    total = float(raw.sum())
    if total <= 0 or not np.all(np.isfinite(raw)):
        raise FloatingPointError("Invalid backbone aggregation weights.")
    return raw / total


def classwise_aggregation_weights(
    updates: Sequence[ClientUpdate],
    *,
    reliability_tau: float,
    alignment_min: float,
    alignment_power: float,
    use_alignment: bool,
) -> NDArray[np.float64]:
    """Return a ``clients × classes`` matrix of normalized reliability weights."""

    if reliability_tau < 0:
        raise ValueError("reliability_tau must be non-negative.")
    num_classes = int(updates[0].class_train_counts.size)
    if any(item.class_train_counts.size != num_classes for item in updates):
        raise ValueError("Client class-count vectors have different lengths.")
    matrix = np.zeros((len(updates), num_classes), dtype=np.float64)
    for class_id in range(num_classes):
        counts = np.asarray(
            [item.class_train_counts[class_id] for item in updates],
            dtype=np.float64,
        )
        if float(counts.sum()) <= 0:
            continue
        vectors = [
            torch.cat(
                [
                    item.delta["classifier.weight"][class_id].reshape(-1),
                    item.delta["classifier.bias"][class_id].reshape(-1),
                ]
            )
            for item in updates
        ]
        reference = torch.zeros_like(vectors[0])
        reference_weights = counts / float(counts.sum())
        for weight, vector in zip(reference_weights, vectors):
            reference.add_(vector, alpha=float(weight))
        alignments = np.ones(len(updates), dtype=np.float64)
        if use_alignment:
            for index, vector in enumerate(vectors):
                if counts[index] <= 0:
                    alignments[index] = 0.0
                else:
                    mapped = (stable_cosine(vector, reference) + 1.0) / 2.0
                    alignments[index] = np.clip(mapped, alignment_min, 1.0)
        sufficiency = counts / (counts + float(reliability_tau))
        reliability = sufficiency * np.power(alignments, float(alignment_power))
        raw = counts * reliability
        raw[counts <= 0] = 0.0
        total = float(raw.sum())
        if total > 0:
            matrix[:, class_id] = raw / total
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("Class-wise weights contain NaN or Inf.")
    return matrix


def aggregate_fedrba(
    global_state: Mapping[str, Tensor],
    updates: Sequence[ClientUpdate],
    blindspot_weights: NDArray[np.floating],
    *,
    server_lr: float,
    reliability_tau: float,
    alignment_min: float,
    alignment_power: float,
    blindspot_update_scale: float,
    blindspot_scale_min: float,
    blindspot_scale_max: float,
    use_classwise_aggregation: bool,
    use_update_alignment: bool,
    use_blindspot_server_scaling: bool,
) -> tuple[dict[str, Tensor], AggregationReport]:
    """Aggregate backbone by client reliability and classifier by class."""

    _validate_updates(global_state, updates)
    if "classifier.weight" not in global_state or "classifier.bias" not in global_state:
        raise KeyError("FedRBA requires classifier.weight and classifier.bias.")
    q = np.asarray(blindspot_weights, dtype=np.float64)
    num_classes = int(global_state["classifier.weight"].shape[0])
    if q.shape != (num_classes,) or np.any(q < 0) or not np.all(np.isfinite(q)):
        raise ValueError("blindspot_weights do not match classifier classes.")
    q = q / float(q.sum())
    backbone_weights = backbone_aggregation_weights(
        updates,
        alignment_min=alignment_min,
        alignment_power=alignment_power,
        use_alignment=use_update_alignment,
    )
    if use_classwise_aggregation:
        class_weights = classwise_aggregation_weights(
            updates,
            reliability_tau=reliability_tau,
            alignment_min=alignment_min,
            alignment_power=alignment_power,
            use_alignment=use_update_alignment,
        )
    else:
        class_weights = np.repeat(
            sample_count_weights(updates)[:, None], num_classes, axis=1
        )

    result = {key: tensor.detach().clone() for key, tensor in global_state.items()}
    for key, original in global_state.items():
        if key.startswith("classifier.") or not torch.is_floating_point(original):
            continue
        aggregate = torch.zeros_like(original)
        for weight, update in zip(backbone_weights, updates):
            aggregate.add_(update.delta[key].to(aggregate.dtype), alpha=float(weight))
        result[key] = original + float(server_lr) * aggregate

    scales = np.ones(num_classes, dtype=np.float64)
    if use_blindspot_server_scaling:
        scales = np.clip(
            1.0 + float(blindspot_update_scale) * (num_classes * q - 1.0),
            float(blindspot_scale_min),
            float(blindspot_scale_max),
        )
    for class_id in range(num_classes):
        weights = class_weights[:, class_id]
        if float(weights.sum()) <= 0:
            continue
        weight_delta = torch.zeros_like(global_state["classifier.weight"][class_id])
        bias_delta = torch.zeros_like(global_state["classifier.bias"][class_id])
        for weight, update in zip(weights, updates):
            weight_delta.add_(
                update.delta["classifier.weight"][class_id].to(weight_delta.dtype),
                alpha=float(weight),
            )
            bias_delta.add_(
                update.delta["classifier.bias"][class_id].to(bias_delta.dtype),
                alpha=float(weight),
            )
        step = float(server_lr) * float(scales[class_id])
        result["classifier.weight"][class_id] = (
            global_state["classifier.weight"][class_id] + step * weight_delta
        )
        result["classifier.bias"][class_id] = (
            global_state["classifier.bias"][class_id] + step * bias_delta
        )
    report = AggregationReport(
        backbone_weights={
            update.client_id: float(weight)
            for update, weight in zip(updates, backbone_weights)
        },
        classwise_weights={
            update.client_id: class_weights[index].astype(float).tolist()
            for index, update in enumerate(updates)
        },
    )
    return result, report

