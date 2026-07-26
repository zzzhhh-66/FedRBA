"""Experiment-matrix coverage and deduplication tests."""

from scripts.build_experiment_matrix import ALGORITHMS, SEEDS, build


def test_full_main_matrix_has_requested_90_jobs() -> None:
    commands = build("full")
    assert len(commands) == 90
    assert len(commands) == len(set(commands))
    assert any(
        "configs/default_credit.yaml" in item
        and "partition.dirichlet_alpha=0.1" in item
        for item in commands
    )


def test_reduced_main_matrix_has_requested_75_jobs() -> None:
    commands = build("reduced")
    assert len(commands) == 75
    assert len(commands) == len(set(commands))


def test_optional_ablations_add_only_nine_jobs() -> None:
    commands = build("full", include_ablations=True)
    assert len(commands) == 99
    assert len(commands) == len(set(commands))
    expected = 3 * len(SEEDS)
    actual = sum("experiment.variant=ablation_" in item for item in commands)
    assert actual == expected


def test_ablation_only_matrix_has_nine_jobs() -> None:
    commands = build("full", only_ablations=True)
    assert len(commands) == 9
    assert all("experiment.variant=ablation_" in item for item in commands)
    assert len(commands) == len(set(commands))


def test_seed_batches_cover_full_matrix_without_overlap() -> None:
    first = build("full", seeds=[1])
    remaining = build("full", seeds=[2, 3])
    assert len(first) == 30
    assert len(remaining) == 60
    assert set(first).isdisjoint(remaining)
    assert set(first) | set(remaining) == set(build("full"))


def test_every_main_cell_has_three_seeds_and_five_algorithms() -> None:
    commands = build("full")
    for algorithm in ALGORITHMS:
        assert sum(f"--algorithm {algorithm} " in item for item in commands) == 18
