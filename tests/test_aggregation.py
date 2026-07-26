"""Class-wise reliability aggregation edge cases."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.federated.aggregation import (  # noqa: E402
    aggregate_fedrba,
    classwise_aggregation_weights,
    stable_cosine,
)
from src.federated.blindspot import ClassStatistics  # noqa: E402
from src.federated.client import ClientUpdate  # noqa: E402


def _update(client_id: int, counts: list[int], scale: float) -> ClientUpdate:
    stats = ClassStatistics(
        count=np.asarray(counts),
        loss_sum=np.ones(3),
        correct=np.zeros(3, dtype=np.int64),
    )
    return ClientUpdate(
        client_id=client_id,
        delta={
            "backbone.weight": torch.full((2, 2), scale),
            "classifier.weight": torch.full((3, 2), scale),
            "classifier.bias": torch.full((3,), scale),
        },
        train_sample_count=sum(counts),
        class_train_counts=np.asarray(counts, dtype=np.int64),
        validation_statistics=stats,
        train_loss=1.0,
    )


def test_class_weights_sum_to_one_and_absent_clients_are_zero() -> None:
    updates = [_update(0, [10, 0, 0], 0.1), _update(1, [5, 8, 0], 0.2)]
    weights = classwise_aggregation_weights(
        updates,
        reliability_tau=20,
        alignment_min=0.1,
        alignment_power=1,
        use_alignment=True,
    )
    assert np.isclose(weights[:, 0].sum(), 1.0)
    assert np.isclose(weights[:, 1].sum(), 1.0)
    assert weights[0, 1] == 0
    assert np.all(weights[:, 2] == 0)


def test_single_client_class_and_all_absent_class_are_safe() -> None:
    state = {
        "backbone.weight": torch.zeros(2, 2),
        "classifier.weight": torch.zeros(3, 2),
        "classifier.bias": torch.zeros(3),
    }
    updates = [_update(0, [10, 0, 0], 0.1), _update(1, [0, 8, 0], 0.2)]
    result, report = aggregate_fedrba(
        state,
        updates,
        np.full(3, 1 / 3),
        server_lr=1,
        reliability_tau=20,
        alignment_min=0.1,
        alignment_power=1,
        blindspot_update_scale=0.5,
        blindspot_scale_min=0.5,
        blindspot_scale_max=1.5,
        use_classwise_aggregation=True,
        use_update_alignment=True,
        use_blindspot_server_scaling=True,
    )
    assert report.classwise_weights[0][0] == 1.0
    assert report.classwise_weights[1][1] == 1.0
    torch.testing.assert_close(result["classifier.weight"][2], state["classifier.weight"][2])
    torch.testing.assert_close(result["classifier.bias"][2], state["classifier.bias"][2])


def test_stable_cosine_defines_zero_vectors() -> None:
    zero = torch.zeros(4)
    one = torch.ones(4)
    assert stable_cosine(zero, zero) == 1.0
    assert stable_cosine(zero, one) == 0.0

