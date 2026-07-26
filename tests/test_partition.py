"""Acceptance tests for reproducible client partitioning."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.fashion_mnist import dataset_targets
from src.data.partition import (
    apply_lognormal_quantity_skew,
    client_label_histograms,
    partition_labels_dirichlet,
    split_client_indices,
)


def _imbalanced_labels() -> np.ndarray:
    return np.concatenate(
        [
            np.full(500, 0, dtype=np.int64),
            np.full(250, 1, dtype=np.int64),
            np.full(150, 2, dtype=np.int64),
            np.full(100, 3, dtype=np.int64),
        ]
    )


def test_dirichlet_partition_is_complete_unique_minimum_and_reproducible() -> None:
    labels = _imbalanced_labels()
    kwargs = {
        "num_clients": 5,
        "alpha": 0.3,
        "min_client_samples": 100,
        "seed": 42,
    }

    first = partition_labels_dirichlet(labels, **kwargs)
    second = partition_labels_dirichlet(labels, **kwargs)

    assert first == second
    assert sorted(first) == list(range(kwargs["num_clients"]))
    assert all(
        len(client_indices) >= kwargs["min_client_samples"]
        for client_indices in first.values()
    )

    flattened = [index for indexes in first.values() for index in indexes]
    assert len(flattened) == len(labels)
    assert len(set(flattened)) == len(labels)
    assert sorted(flattened) == list(range(len(labels)))


def test_different_seed_changes_partition() -> None:
    labels = _imbalanced_labels()
    common = {
        "num_clients": 5,
        "alpha": 0.3,
        "min_client_samples": 100,
    }
    first = partition_labels_dirichlet(labels, seed=7, **common)
    second = partition_labels_dirichlet(labels, seed=8, **common)
    assert first != second


def test_local_split_is_disjoint_complete_and_reproducible() -> None:
    labels = _imbalanced_labels()
    partitions = partition_labels_dirichlet(
        labels,
        num_clients=5,
        alpha=0.1,
        min_client_samples=100,
        seed=123,
    )

    first = split_client_indices(
        partitions, validation_fraction=0.2, seed=123
    )
    second = split_client_indices(
        partitions, validation_fraction=0.2, seed=123
    )
    assert first == second

    for client_id, original in partitions.items():
        train = first[client_id]["train"]
        validation = first[client_id]["validation"]
        assert train
        assert validation
        assert set(train).isdisjoint(validation)
        assert set(train) | set(validation) == set(original)
        assert abs(len(validation) / len(original) - 0.2) <= 1 / len(original)


def test_histograms_account_for_every_label() -> None:
    labels = _imbalanced_labels()
    partitions = partition_labels_dirichlet(
        labels,
        num_clients=5,
        alpha=1.0,
        min_client_samples=100,
        seed=99,
    )
    histograms = client_label_histograms(labels, partitions, num_classes=4)
    total = np.sum(np.asarray(list(histograms.values())), axis=0)
    np.testing.assert_array_equal(total, np.bincount(labels, minlength=4))


def test_infeasible_minimum_is_rejected() -> None:
    labels = np.arange(20) % 2
    with pytest.raises(ValueError, match="Infeasible minimum"):
        partition_labels_dirichlet(
            labels,
            num_clients=5,
            alpha=0.3,
            min_client_samples=5,
            seed=1,
        )


def test_dataset_targets_preserves_subset_index_order() -> None:
    class DummyDataset:
        targets = np.asarray([4, 3, 2, 1, 0], dtype=np.int64)

        def __len__(self) -> int:
            return len(self.targets)

    class DummySubset:
        def __init__(self) -> None:
            self.dataset = DummyDataset()
            self.indices = [4, 1, 3]

        def __len__(self) -> int:
            return len(self.indices)

    np.testing.assert_array_equal(
        dataset_targets(DummySubset()), np.asarray([0, 3, 1])
    )


def test_lognormal_quantity_skew_is_complete_and_reproducible() -> None:
    labels = _imbalanced_labels()
    partitions = partition_labels_dirichlet(
        labels,
        num_clients=5,
        alpha=0.3,
        min_client_samples=100,
        seed=42,
    )
    first = apply_lognormal_quantity_skew(
        partitions, min_client_samples=100, sigma=1.0, seed=9
    )
    second = apply_lognormal_quantity_skew(
        partitions, min_client_samples=100, sigma=1.0, seed=9
    )
    assert first == second
    flattened = [index for values in first.values() for index in values]
    assert sorted(flattened) == list(range(len(labels)))
    assert min(map(len, first.values())) >= 100
    assert len(set(map(len, first.values()))) > 1
