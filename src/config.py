"""YAML configuration loading, overrides, and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


def _parse_override(override: str) -> tuple[list[str], Any]:
    """Parse one ``dotted.path=value`` command-line override."""

    if "=" not in override:
        raise ConfigError(
            f"Invalid override {override!r}; expected dotted.path=value."
        )
    dotted_key, raw_value = override.split("=", maxsplit=1)
    keys = [part.strip() for part in dotted_key.split(".")]
    if not keys or any(not key for key in keys):
        raise ConfigError(f"Invalid override path in {override!r}.")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML value in override {override!r}.") from exc
    return keys, value


def apply_overrides(
    config: Mapping[str, Any], overrides: Sequence[str] | None = None
) -> dict[str, Any]:
    """Return a deep copy of ``config`` with dotted overrides applied.

    Existing paths must remain mappings while being traversed. The final key may
    be new, which permits adding optional experimental fields from the CLI.
    """

    result: dict[str, Any] = deepcopy(dict(config))
    for override in overrides or ():
        keys, value = _parse_override(override)
        cursor: MutableMapping[str, Any] = result
        for key in keys[:-1]:
            if key not in cursor:
                raise ConfigError(
                    f"Unknown override section {'.'.join(keys[:-1])!r}."
                )
            child = cursor[key]
            if not isinstance(child, MutableMapping):
                raise ConfigError(
                    f"Override path {'.'.join(keys)!r} crosses non-mapping "
                    f"value at {key!r}."
                )
            cursor = child
        cursor[keys[-1]] = value
    return result


def _require_section(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Missing or invalid configuration section {section!r}.")
    return value


def _require_positive_int(section: Mapping[str, Any], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key!r} must be a positive integer, got {value!r}.")
    return value


def _require_positive_number(section: Mapping[str, Any], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{key!r} must be positive, got {value!r}.")
    return float(value)


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate cross-module experiment invariants before expensive work."""

    experiment = _require_section(config, "experiment")
    dataset = _require_section(config, "dataset")
    federated = _require_section(config, "federated")
    partition = _require_section(config, "partition")
    model = _require_section(config, "model")
    optimizer = _require_section(config, "optimizer")
    fedrba = _require_section(config, "fedrba")
    evaluation = _require_section(config, "evaluation")

    seed = experiment.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"experiment.seed must be non-negative, got {seed!r}.")
    dataset_name = dataset.get("name")
    if dataset_name not in {
        "fashion_mnist",
        "default_credit",
        "give_me_some_credit",
    }:
        raise ConfigError(
            "dataset.name must be fashion_mnist, default_credit, or "
            "give_me_some_credit, "
            f"got {dataset_name!r}."
        )
    num_classes = _require_positive_int(dataset, "num_classes")
    if num_classes < 2:
        raise ConfigError("dataset.num_classes must be at least 2.")
    validation_fraction = partition.get("validation_fraction", 0.2)
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ConfigError("partition.validation_fraction must lie strictly in (0, 1).")

    num_clients = _require_positive_int(partition, "num_clients")
    clients_per_round = _require_positive_int(federated, "clients_per_round")
    if clients_per_round > num_clients:
        raise ConfigError("federated.clients_per_round cannot exceed num_clients.")
    for key in ("rounds", "local_epochs", "batch_size"):
        _require_positive_int(federated, key)
    _require_positive_int(partition, "min_client_samples")
    if partition.get("type") != "label_dirichlet":
        raise ConfigError("partition.type must be 'label_dirichlet'.")
    _require_positive_number(partition, "dirichlet_alpha")
    quantity_skew = partition.get("quantity_skew", {"type": "none", "sigma": 0.0})
    if not isinstance(quantity_skew, Mapping):
        raise ConfigError("partition.quantity_skew must be a mapping.")
    quantity_type = quantity_skew.get("type", "none")
    if quantity_type not in {"none", "lognormal"}:
        raise ConfigError(
            "partition.quantity_skew.type must be 'none' or 'lognormal'."
        )
    quantity_sigma = quantity_skew.get("sigma", 0.0)
    if (
        isinstance(quantity_sigma, bool)
        or not isinstance(quantity_sigma, (int, float))
        or float(quantity_sigma) < 0.0
    ):
        raise ConfigError("partition.quantity_skew.sigma must be non-negative.")
    if quantity_type == "lognormal" and float(quantity_sigma) <= 0.0:
        raise ConfigError(
            "partition.quantity_skew.sigma must be positive for lognormal skew."
        )

    input_dim = model.get("input_dim")
    if dataset_name == "fashion_mnist":
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ConfigError("model.input_dim must be positive for Fashion-MNIST.")
    elif input_dim is not None and (
        isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0
    ):
        raise ConfigError("model.input_dim must be null or positive for CSV data.")

    hidden_dims = model.get("hidden_dims")
    if (
        not isinstance(hidden_dims, list)
        or not hidden_dims
        or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in hidden_dims
        )
    ):
        raise ConfigError("model.hidden_dims must be a non-empty list of positive ints.")
    dropout = model.get("dropout", 0.0)
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise ConfigError("model.dropout must lie in [0, 1).")

    if optimizer.get("name") not in {"adam", "sgd"}:
        raise ConfigError("optimizer.name must be 'adam' or 'sgd'.")
    _require_positive_number(optimizer, "lr")
    weight_decay = optimizer.get("weight_decay", 0.0)
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or weight_decay < 0
    ):
        raise ConfigError("optimizer.weight_decay must be non-negative.")

    temperature = fedrba.get("blindspot_temperature")
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ConfigError("fedrba.blindspot_temperature must be positive.")
    ema = fedrba.get("blindspot_ema")
    if not isinstance(ema, (int, float)) or not 0 <= float(ema) < 1:
        raise ConfigError("fedrba.blindspot_ema must lie in [0, 1).")
    alignment_min = fedrba.get("alignment_min")
    if (
        not isinstance(alignment_min, (int, float))
        or not 0 < float(alignment_min) <= 1
    ):
        raise ConfigError("fedrba.alignment_min must lie in (0, 1].")
    for key in (
        "use_blindspot_loss",
        "use_blindspot_server_scaling",
        "use_classwise_aggregation",
        "use_update_alignment",
    ):
        if not isinstance(fedrba.get(key), bool):
            raise ConfigError(f"fedrba.{key} must be boolean.")
    for key in ("eval_every", "save_every"):
        _require_positive_int(evaluation, key)


def load_config(
    path: str | Path, overrides: Sequence[str] | None = None
) -> dict[str, Any]:
    """Load, override, and validate a YAML experiment configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}.") from exc
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Top-level YAML value in {config_path} must be a mapping.")
    config = apply_overrides(loaded, overrides)
    validate_config(config)
    return config


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a resolved configuration as stable, human-readable YAML."""

    validate_config(config)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            dict(config), sort_keys=False, allow_unicode=True, default_flow_style=False
        ),
        encoding="utf-8",
    )
