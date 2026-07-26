"""Dataset loading and federated partition helpers."""

from typing import Any, Mapping

from .base import ArrayDataset, DatasetBundle
from .default_credit import load_default_credit
from .fashion_mnist import (
    build_fashion_mnist_bundle,
    dataset_targets,
    load_fashion_mnist,
)
from .give_me_some_credit import load_give_me_some_credit
from .partition import (
    apply_lognormal_quantity_skew,
    client_label_histograms,
    partition_labels_dirichlet,
    split_client_indices,
)

__all__ = [
    "ArrayDataset",
    "apply_lognormal_quantity_skew",
    "DatasetBundle",
    "build_dataset",
    "client_label_histograms",
    "dataset_targets",
    "load_fashion_mnist",
    "partition_labels_dirichlet",
    "split_client_indices",
]


def build_dataset(config: Mapping[str, Any], *, seed: int) -> DatasetBundle:
    """Dispatch a dataset configuration to its concrete loader."""

    name = str(config.get("name"))
    if name == "fashion_mnist":
        return build_fashion_mnist_bundle(config, seed=seed)
    if name == "default_credit":
        return load_default_credit(config, seed=seed)
    if name == "give_me_some_credit":
        return load_give_me_some_credit(config, seed=seed)
    raise ValueError(f"Unknown dataset name {name!r}.")
