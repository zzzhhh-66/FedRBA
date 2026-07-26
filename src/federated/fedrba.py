"""Algorithm presets and configurable FedRBA ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SUPPORTED_ALGORITHMS = {
    "fedavg",
    "fedprox",
    "fedavg_classweight",
    "fedrba_classagg",
    "fedrba",
}


@dataclass(frozen=True)
class AlgorithmOptions:
    """Resolved behavior for one named baseline or ablation."""

    name: str
    use_fedprox: bool
    update_blindspot: bool
    use_blindspot_loss: bool
    use_blindspot_server_scaling: bool
    use_classwise_aggregation: bool
    use_update_alignment: bool
    use_rba_aggregation: bool


def resolve_algorithm(
    algorithm: str, fedrba_config: Mapping[str, Any]
) -> AlgorithmOptions:
    """Map a CLI algorithm and YAML switches to unambiguous behavior."""

    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; choose {sorted(SUPPORTED_ALGORITHMS)}."
        )
    if algorithm == "fedavg":
        return AlgorithmOptions(algorithm, False, False, False, False, False, False, False)
    if algorithm == "fedprox":
        return AlgorithmOptions(algorithm, True, False, False, False, False, False, False)
    if algorithm == "fedavg_classweight":
        return AlgorithmOptions(algorithm, False, True, True, False, False, False, False)
    if algorithm == "fedrba_classagg":
        return AlgorithmOptions(algorithm, False, False, False, False, True, True, True)
    return AlgorithmOptions(
        algorithm,
        False,
        True,
        bool(fedrba_config["use_blindspot_loss"]),
        bool(fedrba_config["use_blindspot_server_scaling"]),
        bool(fedrba_config["use_classwise_aggregation"]),
        bool(fedrba_config["use_update_alignment"]),
        True,
    )

