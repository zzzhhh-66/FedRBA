"""Tests for auditable 10/20/30-round sensitivity analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_window_sensitivity import analyze


DATASET_SCENARIOS = {
    "fashion_mnist": ("alpha_0p1", "alpha_0p3", "alpha_1"),
    "default_credit": ("alpha_0p1", "alpha_0p3"),
    "give_me_some_credit": ("alpha_0p3",),
}
ROUNDS = {
    "fashion_mnist": 100,
    "default_credit": 150,
    "give_me_some_credit": 150,
}
METHODS = (
    "fedavg",
    "fedavg_classweight",
    "fedprox",
    "fedrba",
    "fedrba_classagg",
)


def _write_main_matrix(root: Path) -> None:
    for dataset, scenarios in DATASET_SCENARIOS.items():
        for scenario in scenarios:
            for method_id, method in enumerate(METHODS):
                for seed in (1, 2, 3):
                    rounds = np.arange(1, ROUNDS[dataset] + 1)
                    base = 0.5 + method_id * 0.01 + seed * 0.001
                    minority_recall = np.full(len(rounds), base)
                    minority_recall[rounds % 10 == 0] = 0.0
                    frame = pd.DataFrame(
                        {
                            "round": rounds,
                            "accuracy": base + rounds / 100_000,
                            "macro_f1": base + rounds / 110_000,
                            "balanced_accuracy": base + rounds / 120_000,
                            "roc_auc": base + rounds / 130_000,
                            "pr_auc": base + rounds / 140_000,
                            "minority_recall": minority_recall,
                            "minority_f1": base + rounds / 150_000,
                        }
                    )
                    path = (
                        root
                        / dataset
                        / method
                        / scenario
                        / f"seed_{seed}"
                        / "metrics.csv"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(path, index=False)


def test_window_analysis_covers_90_runs_and_three_windows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    destination = tmp_path / "sensitivity"
    _write_main_matrix(root)

    outputs = analyze(root, destination)
    assert all(path.is_file() for path in outputs.values())

    per_run = pd.read_csv(outputs["per_run"])
    aggregate = pd.read_csv(outputs["aggregate"])
    comparison = pd.read_csv(outputs["comparison"])
    gmsc_per_run = pd.read_csv(outputs["gmsc_per_run"])
    gmsc_aggregate = pd.read_csv(outputs["gmsc_aggregate"])

    assert len(per_run) == 270
    assert len(aggregate) == 90
    assert len(comparison) == 30
    assert len(gmsc_per_run) == 2_250
    assert len(gmsc_aggregate) == 750
    assert sorted(per_run["window"].unique()) == [10, 20, 30]
    assert set(aggregate["n_seeds"]) == {3}
    assert np.allclose(per_run["minority_collapse_rate"], 0.1)

    fashion = per_run[
        (per_run["dataset"] == "fashion_mnist")
        & (per_run["method"] == "fedavg")
        & (per_run["scenario"] == "alpha_1")
        & (per_run["seed"] == 1)
    ].set_index("window")
    assert fashion.loc[10, "start_round"] == 91
    assert fashion.loc[20, "start_round"] == 81
    assert fashion.loc[30, "start_round"] == 71


def test_window_analysis_rejects_incomplete_history(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _write_main_matrix(root)
    broken = (
        root
        / "fashion_mnist"
        / "fedavg"
        / "alpha_1"
        / "seed_1"
        / "metrics.csv"
    )
    frame = pd.read_csv(broken).iloc[:-1]
    frame.to_csv(broken, index=False)

    try:
        analyze(root, tmp_path / "sensitivity")
    except ValueError as exc:
        assert "contiguous rounds" in str(exc)
    else:
        raise AssertionError("Incomplete histories must be rejected.")
