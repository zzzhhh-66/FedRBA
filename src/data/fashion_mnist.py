"""Fashion-MNIST loading with deterministic optional subsampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import DatasetBundle


def dataset_targets(dataset: Any) -> np.ndarray:
    """Extract labels in dataset index order, including nested torch subsets."""

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        parent_targets = dataset_targets(dataset.dataset)
        indices = np.asarray(dataset.indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("Subset indices must be one-dimensional.")
        return parent_targets[indices]

    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError(
            "Dataset has no 'targets' attribute; provide a label extraction adapter."
        )
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    array = np.asarray(targets)
    if array.ndim != 1 or array.shape[0] != len(dataset):
        raise ValueError("Dataset targets must be one-dimensional and match its length.")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError("Dataset targets must be finite integer-valued labels.")
    return array.astype(np.int64, copy=False)


def _deterministic_subset(
    dataset: Any, maximum_samples: int | None, seed: int
) -> Any:
    """Return a deterministic torch ``Subset`` when a limit is requested."""

    if maximum_samples is None:
        return dataset
    if (
        isinstance(maximum_samples, bool)
        or not isinstance(maximum_samples, int)
        or maximum_samples <= 0
    ):
        raise ValueError("maximum_samples must be a positive integer or None.")
    if maximum_samples >= len(dataset):
        return dataset

    from torch.utils.data import Subset

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=maximum_samples, replace=False))
    return Subset(dataset, indices.tolist())


def load_fashion_mnist(
    data_dir: str | Path,
    *,
    download: bool = True,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    seed: int = 42,
) -> tuple[Any, Any]:
    """Load normalized Fashion-MNIST train and independent test datasets.

    PyTorch imports are lazy so partition-only tests do not require the complete
    training environment. A clear error is raised when dependencies are absent.
    The test dataset is returned independently and must never be used for
    blindspot estimation.
    """

    try:
        from torchvision import datasets, transforms
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Fashion-MNIST loading requires compatible torch and torchvision "
            "installations. Install requirements.txt in Python 3.11."
        ) from exc

    root = Path(data_dir)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = datasets.FashionMNIST(
        root=str(root), train=True, transform=transform, download=download
    )
    test_dataset = datasets.FashionMNIST(
        root=str(root), train=False, transform=transform, download=download
    )
    train_dataset = _deterministic_subset(train_dataset, max_train_samples, seed)
    test_dataset = _deterministic_subset(test_dataset, max_test_samples, seed + 1)
    return train_dataset, test_dataset


def _apply_long_tail(
    dataset: Any,
    targets: np.ndarray,
    *,
    imbalance_ratio: float,
    seed: int,
) -> tuple[Any, np.ndarray, list[int]]:
    """Downsample training classes exponentially while preserving the test set."""

    if not 0.0 < imbalance_ratio <= 1.0:
        raise ValueError("imbalance_ratio must lie in (0, 1].")
    from torch.utils.data import Subset

    rng = np.random.default_rng(seed)
    classes = np.unique(targets)
    maximum = min(int(np.sum(targets == label)) for label in classes)
    selected: list[int] = []
    for rank, label in enumerate(classes):
        exponent = rank / max(1, len(classes) - 1)
        count = max(1, int(round(maximum * imbalance_ratio**exponent)))
        candidates = np.flatnonzero(targets == label)
        chosen = rng.choice(candidates, size=min(count, len(candidates)), replace=False)
        selected.extend(int(index) for index in chosen)
    rng.shuffle(selected)
    subset = Subset(dataset, selected)
    return subset, targets[np.asarray(selected, dtype=np.int64)], selected


def build_fashion_mnist_bundle(
    config: Mapping[str, Any], *, seed: int
) -> DatasetBundle:
    """Build a Fashion-MNIST bundle with optional training-only long tail."""

    train_dataset, test_dataset = load_fashion_mnist(
        str(config.get("data_dir", "data")),
        download=bool(config.get("download", True)),
        max_train_samples=config.get("max_train_samples"),
        max_test_samples=config.get("max_test_samples"),
        seed=seed,
    )
    train_targets = dataset_targets(train_dataset)
    test_targets = dataset_targets(test_dataset)
    imbalance = config.get("imbalance", {})
    selected_indices: list[int] | None = None
    if isinstance(imbalance, Mapping) and imbalance.get("type", "none") != "none":
        if imbalance.get("type") != "exponential":
            raise ValueError("Fashion-MNIST imbalance.type must be none or exponential.")
        train_dataset, train_targets, selected_indices = _apply_long_tail(
            train_dataset,
            train_targets,
            imbalance_ratio=float(imbalance.get("imbalance_ratio", 0.1)),
            seed=seed,
        )
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    minority = (
        int(np.argmin(np.bincount(train_targets, minlength=10)))
        if selected_indices is not None
        else None
    )
    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_targets=train_targets,
        test_targets=test_targets,
        input_dim=784,
        num_classes=10,
        minority_class=minority,
        feature_names=None,
        class_names=classes,
        metadata={
            "source": "torchvision:FashionMNIST",
            "long_tail_selected_indices": selected_indices,
        },
    )
