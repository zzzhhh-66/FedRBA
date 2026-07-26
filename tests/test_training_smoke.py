"""Two-round CPU smoke test with 1,000 Fashion-shaped synthetic samples."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import load_config  # noqa: E402
from src.data.base import ArrayDataset, DatasetBundle  # noqa: E402
from src.federated.server import run_experiment  # noqa: E402


def make_smoke_bundle() -> DatasetBundle:
    rng = np.random.default_rng(123)
    train_targets = np.arange(1_000, dtype=np.int64) % 10
    test_targets = np.arange(200, dtype=np.int64) % 10
    prototypes = rng.normal(0, 1, size=(10, 784)).astype(np.float32)
    train_features = prototypes[train_targets] + rng.normal(
        0, 0.25, size=(1_000, 784)
    ).astype(np.float32)
    test_features = prototypes[test_targets] + rng.normal(
        0, 0.25, size=(200, 784)
    ).astype(np.float32)
    return DatasetBundle(
        ArrayDataset(train_features, train_targets),
        ArrayDataset(test_features, test_targets),
        train_targets,
        test_targets,
        input_dim=784,
        num_classes=10,
        minority_class=None,
        feature_names=None,
    )


@pytest.mark.parametrize("algorithm", ["fedavg", "fedrba"])
def test_two_round_cpu_training(tmp_path: Path, algorithm: str) -> None:
    config = copy.deepcopy(load_config("configs/fashion_mnist.yaml"))
    config["experiment"]["output_dir"] = str(tmp_path)
    config["experiment"]["seed"] = 7
    config["federated"].update(
        clients_per_round=2, rounds=2, local_epochs=1, batch_size=64
    )
    config["partition"].update(num_clients=5, min_client_samples=100)
    config["model"]["hidden_dims"] = [32, 16]
    config["model"]["dropout"] = 0.0
    config["runtime"].update(device="cpu", num_workers=0, pin_memory=False)
    config["evaluation"]["save_every"] = 1
    result = run_experiment(
        config,
        algorithm=algorithm,
        bundle=make_smoke_bundle(),
        device_override="cpu",
    )
    assert result["round"] == 2
    assert np.isfinite(result["accuracy"])
    assert np.isclose(sum(result["blindspot_weights"]), 1.0)
    run_dir = tmp_path / "fashion_mnist" / algorithm / "alpha_0p3" / "seed_7"
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "plots" / "confusion_matrix.png").is_file()
