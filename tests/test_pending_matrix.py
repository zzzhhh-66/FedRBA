"""Pending-matrix result validation tests."""

from __future__ import annotations

import json

from scripts.build_pending_matrix import is_valid_result


def test_valid_completed_result(tmp_path) -> None:
    path = tmp_path / "final_metrics.json"
    path.write_text(
        json.dumps(
            {
                "round": 100,
                "algorithm": "fedavg",
                "dataset": "fashion_mnist",
                "seed": 1,
                "dirichlet_alpha": 0.3,
            }
        ),
        encoding="utf-8",
    )
    assert is_valid_result(
        path,
        expected_round=100,
        algorithm="fedavg",
        dataset="fashion_mnist",
        alpha=0.3,
        seed=1,
    )


def test_wrong_round_is_not_complete(tmp_path) -> None:
    path = tmp_path / "final_metrics.json"
    path.write_text(
        json.dumps(
            {
                "round": 99,
                "algorithm": "fedavg",
                "dataset": "fashion_mnist",
                "seed": 1,
                "dirichlet_alpha": 0.3,
            }
        ),
        encoding="utf-8",
    )
    assert not is_valid_result(
        path,
        expected_round=100,
        algorithm="fedavg",
        dataset="fashion_mnist",
        alpha=0.3,
        seed=1,
    )
