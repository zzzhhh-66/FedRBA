"""Reproducible federated label-skew partitioning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


ClientPartitions = dict[int, list[int]]
ClientSplits = dict[int, dict[str, list[int]]]


def _as_label_array(labels: ArrayLike) -> NDArray[np.int64]:
    """Validate and normalize a one-dimensional integer label array."""

    array = np.asarray(labels)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array.")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError("labels must contain finite integer-valued entries.")
    return array.astype(np.int64, copy=False)


def _validate_partition_request(
    num_samples: int,
    num_clients: int,
    alpha: float,
    min_client_samples: int,
    seed: int,
) -> None:
    """Validate Dirichlet partition arguments before consuming randomness."""

    for name, value in (
        ("num_clients", num_clients),
        ("min_client_samples", min_client_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError(f"alpha must be finite and positive, got {alpha!r}.")
    required = num_clients * min_client_samples
    if required > num_samples:
        raise ValueError(
            "Infeasible minimum: "
            f"{num_samples} samples cannot give {num_clients} clients at least "
            f"{min_client_samples} samples each (requires {required})."
        )


def _rebalance_minimum(
    client_indices: list[list[int]],
    min_client_samples: int,
    rng: np.random.Generator,
) -> None:
    """Move the fewest possible indexes to satisfy the hard client minimum."""

    deficits = [
        client_id
        for client_id, indices in enumerate(client_indices)
        if len(indices) < min_client_samples
    ]
    for receiver_id in deficits:
        while len(client_indices[receiver_id]) < min_client_samples:
            surplus = np.asarray(
                [
                    max(0, len(indices) - min_client_samples)
                    for indices in client_indices
                ],
                dtype=np.int64,
            )
            donor_id = int(np.argmax(surplus))
            if surplus[donor_id] <= 0:
                raise RuntimeError(
                    "Internal partition error: no donor can satisfy client minimum."
                )
            needed = min_client_samples - len(client_indices[receiver_id])
            transfer_count = min(needed, int(surplus[donor_id]))
            donor = client_indices[donor_id]
            chosen_positions = np.sort(
                rng.choice(len(donor), size=transfer_count, replace=False)
            )
            chosen_set = set(int(position) for position in chosen_positions)
            transferred = [
                sample_index
                for position, sample_index in enumerate(donor)
                if position in chosen_set
            ]
            client_indices[donor_id] = [
                sample_index
                for position, sample_index in enumerate(donor)
                if position not in chosen_set
            ]
            client_indices[receiver_id].extend(transferred)


def _assert_complete_partition(
    partitions: Mapping[int, Sequence[int]], num_samples: int
) -> None:
    """Raise on duplicates, omissions, or out-of-range indexes."""

    flattened = np.asarray(
        [index for values in partitions.values() for index in values],
        dtype=np.int64,
    )
    if flattened.size != num_samples:
        raise RuntimeError(
            f"Partition contains {flattened.size} entries for {num_samples} samples."
        )
    if flattened.size and (
        int(flattened.min()) < 0 or int(flattened.max()) >= num_samples
    ):
        raise RuntimeError("Partition contains an out-of-range sample index.")
    if not np.array_equal(np.sort(flattened), np.arange(num_samples)):
        raise RuntimeError("Partition has duplicate or missing sample indexes.")


def partition_labels_dirichlet(
    labels: ArrayLike,
    *,
    num_clients: int,
    alpha: float,
    min_client_samples: int,
    seed: int,
) -> ClientPartitions:
    """Partition sample indexes using per-class Dirichlet label skew.

    Each class draws client proportions independently. A deterministic minimal
    rebalancing pass then transfers indexes only when required to enforce
    ``min_client_samples``. No sample is duplicated or omitted.
    """

    label_array = _as_label_array(labels)
    _validate_partition_request(
        label_array.size, num_clients, alpha, min_client_samples, seed
    )
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for class_label in np.unique(label_array):
        class_indices = np.flatnonzero(label_array == class_label)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, float(alpha)))
        counts = rng.multinomial(class_indices.size, proportions)
        boundaries = np.cumsum(counts)[:-1]
        class_splits = np.split(class_indices, boundaries)
        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(int(index) for index in split)

    _rebalance_minimum(client_indices, min_client_samples, rng)
    partitions: ClientPartitions = {}
    for client_id, indices in enumerate(client_indices):
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        partitions[client_id] = shuffled.tolist()

    _assert_complete_partition(partitions, label_array.size)
    if any(len(indices) < min_client_samples for indices in partitions.values()):
        raise RuntimeError("Internal partition error: minimum size was not enforced.")
    return partitions


def split_client_indices(
    partitions: Mapping[int, Sequence[int]],
    *,
    validation_fraction: float = 0.2,
    seed: int,
) -> ClientSplits:
    """Split each client's indexes into disjoint train and validation subsets."""

    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ValueError("validation_fraction must lie strictly in (0, 1).")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    result: ClientSplits = {}
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(partitions))
    for child_seed, client_id in zip(child_seeds, sorted(partitions)):
        indexes = np.asarray(partitions[client_id], dtype=np.int64)
        if indexes.size < 2:
            raise ValueError(
                f"Client {client_id} needs at least 2 samples for a local split."
            )
        rng = np.random.default_rng(child_seed)
        shuffled = indexes.copy()
        rng.shuffle(shuffled)
        validation_size = int(round(shuffled.size * float(validation_fraction)))
        validation_size = max(1, min(validation_size, shuffled.size - 1))
        result[client_id] = {
            "train": shuffled[validation_size:].tolist(),
            "validation": shuffled[:validation_size].tolist(),
        }
    return result


def apply_lognormal_quantity_skew(
    partitions: Mapping[int, Sequence[int]],
    *,
    min_client_samples: int,
    sigma: float,
    seed: int,
) -> ClientPartitions:
    """Rebalance an existing label-skew partition to log-normal client sizes.

    Every client first receives the hard minimum. Remaining capacity is drawn
    from normalized log-normal weights, then the fewest necessary sample
    indexes are moved from surplus clients to deficit clients.
    """

    client_ids = sorted(partitions)
    if client_ids != list(range(len(client_ids))):
        raise ValueError("Client ids must be contiguous integers starting at zero.")
    if (
        isinstance(min_client_samples, bool)
        or not isinstance(min_client_samples, int)
        or min_client_samples <= 0
    ):
        raise ValueError("min_client_samples must be a positive integer.")
    if not isinstance(sigma, (int, float)) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive.")
    total_samples = sum(len(partitions[client_id]) for client_id in client_ids)
    _assert_complete_partition(partitions, total_samples)
    required = min_client_samples * len(client_ids)
    if required > total_samples:
        raise ValueError("Quantity-skew minimum is infeasible.")

    rng = np.random.default_rng(seed)
    weights = rng.lognormal(mean=0.0, sigma=float(sigma), size=len(client_ids))
    weights /= float(weights.sum())
    residual = total_samples - required
    target_sizes = (
        np.full(len(client_ids), min_client_samples, dtype=np.int64)
        + rng.multinomial(residual, weights)
    )
    result = {
        client_id: [int(index) for index in partitions[client_id]]
        for client_id in client_ids
    }
    for receiver_id in client_ids:
        while len(result[receiver_id]) < int(target_sizes[receiver_id]):
            surplus = np.asarray(
                [
                    max(0, len(result[donor_id]) - int(target_sizes[donor_id]))
                    for donor_id in client_ids
                ]
            )
            donor_id = int(np.argmax(surplus))
            if surplus[donor_id] <= 0:
                raise RuntimeError("Internal quantity-skew rebalancing failure.")
            transfer_count = min(
                int(target_sizes[receiver_id]) - len(result[receiver_id]),
                int(surplus[donor_id]),
            )
            donor = result[donor_id]
            chosen = set(
                int(position)
                for position in rng.choice(
                    len(donor), size=transfer_count, replace=False
                )
            )
            result[receiver_id].extend(
                value for position, value in enumerate(donor) if position in chosen
            )
            result[donor_id] = [
                value for position, value in enumerate(donor) if position not in chosen
            ]
    for client_id in client_ids:
        values = np.asarray(result[client_id], dtype=np.int64)
        rng.shuffle(values)
        result[client_id] = values.tolist()
    _assert_complete_partition(result, total_samples)
    if [len(result[index]) for index in client_ids] != target_sizes.tolist():
        raise RuntimeError("Quantity-skew targets were not reached exactly.")
    return result


def client_label_histograms(
    labels: ArrayLike,
    partitions: Mapping[int, Sequence[int]],
    *,
    num_classes: int | None = None,
) -> dict[int, list[int]]:
    """Count class samples per client for auditing and JSON serialization."""

    label_array = _as_label_array(labels)
    _assert_complete_partition(partitions, label_array.size)
    inferred_classes = int(label_array.max()) + 1
    width = inferred_classes if num_classes is None else num_classes
    if isinstance(width, bool) or not isinstance(width, int) or width < inferred_classes:
        raise ValueError(
            f"num_classes must be an integer >= {inferred_classes}, got {width!r}."
        )
    return {
        client_id: np.bincount(
            label_array[np.asarray(indexes, dtype=np.int64)], minlength=width
        )
        .astype(int)
        .tolist()
        for client_id, indexes in partitions.items()
    }
