"""Tests for the auditable quantity-skew experiment workflow."""

from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import ConfigError, load_config
from scripts.analyze_quantity_skew import METRICS, METHODS, analyze
from scripts.audit_quantity_skew_partitions import gini, summarize_partitions
from scripts.build_quantity_skew_matrix import build_commands, scenario_name
from scripts.check_quantity_skew_results import check_results
from scripts.pack_quantity_skew_results import build_archive


def test_quantity_skew_config_requires_positive_lognormal_sigma() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "default_credit.yaml"
    valid = load_config(
        config_path,
        [
            "partition.quantity_skew.type=lognormal",
            "partition.quantity_skew.sigma=1.0",
        ],
    )
    assert valid["partition"]["quantity_skew"] == {
        "type": "lognormal",
        "sigma": 1.0,
    }
    try:
        load_config(
            config_path,
            [
                "partition.quantity_skew.type=lognormal",
                "partition.quantity_skew.sigma=0.0",
            ],
        )
    except ConfigError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("Log-normal quantity skew must reject sigma=0.")


def _write_complete_metrics(path: Path, rounds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round"])
        writer.writeheader()
        writer.writerows({"round": value} for value in range(1, rounds + 1))


def test_quantity_skew_matrix_contains_15_full_and_2_smoke_runs(
    tmp_path: Path,
) -> None:
    full, full_expected = build_commands(
        profile="full", output_root=tmp_path / "full"
    )
    smoke, smoke_expected = build_commands(
        profile="smoke", output_root=tmp_path / "smoke"
    )
    assert full_expected == 15
    assert len(full) == 15
    assert len(set(full)) == 15
    assert smoke_expected == 2
    assert len(smoke) == 2
    assert all("partition.quantity_skew.type=lognormal" in item for item in full)
    assert all("partition.quantity_skew.sigma=1" in item for item in full)
    assert all("partition.dirichlet_alpha=0.3" in item for item in full)
    assert sum("--algorithm fedrba " in item for item in full) == 3
    assert sum("--algorithm fedavg " in item for item in full) == 3
    assert all("federated.rounds=2" in item for item in smoke)
    assert any("--algorithm fedavg " in item for item in smoke)
    assert any("--algorithm fedrba " in item for item in smoke)


def test_quantity_skew_matrix_safely_skips_only_complete_runs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    scenario = scenario_name(alpha=0.3, sigma=1.0)
    complete = (
        root
        / "default_credit"
        / "fedavg"
        / scenario
        / "seed_1"
        / "metrics.csv"
    )
    _write_complete_metrics(complete, 150)
    commands, expected = build_commands(
        profile="full", output_root=root, skip_completed=True
    )
    assert expected == 15
    assert len(commands) == 14

    partial = (
        root
        / "default_credit"
        / "fedrba"
        / scenario
        / "seed_2"
        / "metrics.csv"
    )
    _write_complete_metrics(partial, 149)
    try:
        build_commands(
            profile="full", output_root=root, skip_completed=True
        )
    except RuntimeError as exc:
        assert "manual inspection" in str(exc)
    else:
        raise AssertionError("Partial quantity-skew runs must not be skipped.")


def test_quantity_skew_smoke_checker_validates_two_runs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "smoke"
    scenario = scenario_name(alpha=0.3, sigma=1.0, smoke=True)
    partition = json.dumps({"seed": 1, "clients": [[0], [1]]})
    for method in ("fedavg", "fedrba"):
        run_dir = (
            root
            / "default_credit"
            / method
            / scenario
            / "seed_1"
        )
        _write_complete_metrics(run_dir / "metrics.csv", 2)
        (run_dir / "final_metrics.json").write_text(
            "{}", encoding="utf-8"
        )
        (run_dir / "client_partitions.json").write_text(
            partition, encoding="utf-8"
        )
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "partition": {
                        "quantity_skew": {
                            "type": "lognormal",
                            "sigma": 1.0,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
    assert check_results(root, profile="smoke") == {
        "completed": 2,
        "expected": 2,
    }


def test_partition_summary_reports_realized_quantity_skew() -> None:
    labels = np.asarray([0, 1] * 50, dtype=np.int64)
    balanced = {
        0: list(range(0, 25)),
        1: list(range(25, 50)),
        2: list(range(50, 75)),
        3: list(range(75, 100)),
    }
    skewed = {
        0: list(range(0, 10)),
        1: list(range(10, 20)),
        2: list(range(20, 40)),
        3: list(range(40, 100)),
    }
    base_summary, base_clients = summarize_partitions(
        labels, balanced, seed=1, condition="none"
    )
    skew_summary, skew_clients = summarize_partitions(
        labels, skewed, seed=1, condition="lognormal_sigma_1"
    )
    assert len(base_clients) == len(skew_clients) == 4
    assert np.isclose(base_summary["size_cv"], 0.0)
    assert np.isclose(gini([25, 25, 25, 25]), 0.0)
    assert skew_summary["size_cv"] > base_summary["size_cv"]
    assert skew_summary["size_gini"] > base_summary["size_gini"]
    assert skew_summary["size_max_min_ratio"] == 6.0


def _write_analysis_fixture(
    baseline_root: Path, skew_root: Path
) -> None:
    skew_scenario = scenario_name(alpha=0.3, sigma=1.0)
    for condition, root, scenario in (
        ("none", baseline_root, "alpha_0p3"),
        ("lognormal_sigma_1", skew_root, skew_scenario),
    ):
        for method_index, method in enumerate(METHODS):
            for seed in (1, 2, 3):
                run_dir = (
                    root
                    / "default_credit"
                    / method
                    / scenario
                    / f"seed_{seed}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                rounds = np.arange(1, 151, dtype=np.int64)
                base = (
                    0.58
                    + 0.01 * method_index
                    + 0.001 * seed
                    + (0.005 if condition != "none" else 0.0)
                )
                data = {"round": rounds}
                for metric_index, metric in enumerate(METRICS):
                    values = np.full(
                        len(rounds), base + metric_index * 0.001
                    )
                    if metric == "minority_recall":
                        values[rounds % 10 == 0] = 0.0
                    data[metric] = values
                pd.DataFrame(data).to_csv(
                    run_dir / "metrics.csv", index=False
                )
                (run_dir / "final_metrics.json").write_text(
                    "{}", encoding="utf-8"
                )
                quantity = (
                    {"type": "none", "sigma": 0.0}
                    if condition == "none"
                    else {"type": "lognormal", "sigma": 1.0}
                )
                (run_dir / "config.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "partition": {
                                "dirichlet_alpha": 0.3,
                                "quantity_skew": quantity,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "client_partitions.json").write_text(
                    json.dumps(
                        {
                            "condition": condition,
                            "seed": seed,
                            "clients": [10, 20, 30]
                            if condition == "none"
                            else [5, 15, 40],
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )


def test_quantity_skew_analysis_validates_30_paired_runs(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "outputs"
    skew = tmp_path / "quantity_skew_outputs"
    destination = tmp_path / "analysis"
    _write_analysis_fixture(baseline, skew)

    outputs = analyze(baseline, skew, destination)
    assert all(path.is_file() for path in outputs.values())
    per_run = pd.read_csv(outputs["per_run"])
    aggregate = pd.read_csv(outputs["aggregate"])
    effects = pd.read_csv(outputs["condition_effects"])
    comparisons = pd.read_csv(outputs["fedrba_comparisons"])
    assert len(per_run) == 30
    assert len(aggregate) == 10
    assert len(effects) == 40
    assert len(comparisons) == 32
    assert set(per_run["window"]) == {20}
    assert set(aggregate["n_seeds"]) == {3}
    assert np.allclose(per_run["minority_collapse_rate"], 0.1)
    assert not aggregate.isna().any().any()


def test_quantity_skew_archive_excludes_checkpoints(tmp_path: Path) -> None:
    baseline = tmp_path / "outputs"
    skew = tmp_path / "quantity_skew_outputs"
    _write_analysis_fixture(baseline, skew)
    audit_root = tmp_path / "quantity_skew_partition_audit"
    analysis_root = tmp_path / "quantity_skew_analysis"
    audit_root.mkdir()
    analysis_root.mkdir()
    (audit_root / "audit.csv").write_text("ok\n1\n", encoding="utf-8")
    (analysis_root / "aggregate.csv").write_text(
        "ok\n1\n", encoding="utf-8"
    )
    checkpoint = next(skew.glob("**/seed_1"))
    (checkpoint / "checkpoints").mkdir()
    (checkpoint / "checkpoints" / "last.pt").write_bytes(b"large")

    output = build_archive(
        skew,
        audit_root,
        analysis_root,
        tmp_path / "results.tar",
    )
    with tarfile.open(output, mode="r") as archive:
        names = archive.getnames()
    assert any(name.endswith("metrics.csv") for name in names)
    assert any(name.endswith("quantity_skew_analysis/aggregate.csv") for name in names)
    assert all("checkpoints" not in name for name in names)
