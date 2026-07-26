"""Tests for the credit operating-point command matrix."""

from __future__ import annotations

from pathlib import Path

from scripts.build_operating_point_matrix import (
    build_commands,
    discover_run_dirs,
    remove_completed,
)


def test_operating_point_matrix_has_45_unique_commands(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    for dataset, scenarios in {
        "default_credit": ["alpha_0p1", "alpha_0p3"],
        "give_me_some_credit": ["alpha_0p3"],
    }.items():
        for method in [
            "fedavg",
            "fedavg_classweight",
            "fedprox",
            "fedrba",
            "fedrba_classagg",
        ]:
            for scenario in scenarios:
                for seed in (1, 2, 3):
                    run_dir = root / dataset / method / scenario / f"seed_{seed}"
                    checkpoints = run_dir / "checkpoints"
                    checkpoints.mkdir(parents=True)
                    (run_dir / "config.yaml").write_text("config", encoding="utf-8")
                    (run_dir / "client_partitions.json").write_text(
                        "{}", encoding="utf-8"
                    )
                    (checkpoints / "round_0010.pt").write_bytes(b"checkpoint")

    run_dirs = discover_run_dirs(root, [1, 2, 3])
    commands = build_commands(
        run_dirs,
        output_root="operating_point_outputs",
        device="auto",
        fpr_targets=[0.01, 0.05, 0.10],
        recall_targets=[0.50, 0.70, 0.80],
    )
    assert len(run_dirs) == 45
    assert len(commands) == 45
    assert len(set(commands)) == 45
    assert all("--fpr-targets 0.01 0.05 0.1" in value for value in commands)
    assert all("--recall-targets 0.5 0.7 0.8" in value for value in commands)

    completed = (
        tmp_path
        / "operating_point_outputs"
        / "default_credit"
        / "fedavg"
        / "alpha_0p1"
        / "seed_1"
        / "operating_point_metrics.json"
    )
    completed.parent.mkdir(parents=True)
    completed.write_text("{}", encoding="utf-8")
    pending = remove_completed(
        run_dirs, output_root=tmp_path / "operating_point_outputs"
    )
    assert len(pending) == 44
