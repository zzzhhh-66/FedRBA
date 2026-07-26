"""Tests for the two-alpha Default Credit ablation matrix."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_credit_ablation_matrix import build_commands


def _write_complete_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round"])
        writer.writeheader()
        writer.writerows({"round": value} for value in range(1, 151))


def test_credit_ablation_matrix_contains_18_unique_runs(
    tmp_path: Path,
) -> None:
    commands, expected = build_commands(output_root=tmp_path / "outputs")
    assert expected == 18
    assert len(commands) == 18
    assert len(set(commands)) == 18
    assert sum("partition.dirichlet_alpha=0.1" in item for item in commands) == 9
    assert sum("partition.dirichlet_alpha=0.3" in item for item in commands) == 9
    assert sum("fedrba.use_blindspot_loss=false" in item for item in commands) == 6
    assert (
        sum("fedrba.use_classwise_aggregation=false" in item for item in commands)
        == 6
    )
    assert sum("fedrba.use_update_alignment=false" in item for item in commands) == 6
    for seed in (1, 2, 3):
        assert sum(f"experiment.seed={seed}" in item for item in commands) == 6


def test_credit_ablation_matrix_safely_skips_complete_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    completed = (
        root
        / "default_credit"
        / "ablation_no_blindspot_loss"
        / "alpha_0p1"
        / "seed_1"
        / "metrics.csv"
    )
    _write_complete_metrics(completed)
    commands, expected = build_commands(
        output_root=root,
        skip_completed=True,
    )
    assert expected == 18
    assert len(commands) == 17


def test_credit_ablation_matrix_rejects_partial_run(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    partial = (
        root
        / "default_credit"
        / "ablation_no_update_alignment"
        / "alpha_0p3"
        / "seed_2"
        / "metrics.csv"
    )
    partial.parent.mkdir(parents=True)
    partial.write_text("round\n1\n", encoding="utf-8")
    try:
        build_commands(output_root=root, skip_completed=True)
    except RuntimeError as exc:
        assert "manual inspection" in str(exc)
    else:
        raise AssertionError("Partial ablation outputs must not be skipped.")
