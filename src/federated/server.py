"""End-to-end synchronous federated experiment orchestration."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.config import save_config
from src.data import DatasetBundle, build_dataset
from src.data.partition import (
    apply_lognormal_quantity_skew,
    client_label_histograms,
    partition_labels_dirichlet,
    split_client_indices,
)
from src.logging.experiment_logger import ExperimentLogger
from src.metrics.classification import evaluate_model
from src.models import MLPClassifier
from src.seed import set_global_seed
from src.visualization.plots import generate_run_plots

from .aggregation import aggregate_fedavg, aggregate_fedrba
from .blindspot import update_blindspot_weights
from .client import FederatedClient
from .fedrba import resolve_algorithm


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or an explicit CUDA device."""

    value = requested.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but unavailable.")
    return device


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_checkpoint(
    logger: ExperimentLogger,
    *,
    round_number: int,
    model: torch.nn.Module,
    blindspot_weights: np.ndarray,
    selection_rng: np.random.Generator,
    algorithm: str,
    dataset_name: str,
    periodic: bool,
) -> Path:
    payload = {
        "round": int(round_number),
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "blindspot_weights": np.asarray(blindspot_weights, dtype=np.float64),
        "selection_rng_state": selection_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "algorithm": algorithm,
        "dataset_name": dataset_name,
    }
    last_path = logger.checkpoint_dir / "last.pt"
    torch.save(payload, last_path)
    if periodic:
        torch.save(payload, logger.checkpoint_dir / f"round_{round_number:04d}.pt")
    return last_path


def _build_clients(
    bundle: DatasetBundle,
    config: Mapping[str, Any],
    *,
    seed: int,
    logger: ExperimentLogger,
) -> tuple[list[FederatedClient], dict[int, dict[str, list[int]]]]:
    partition_config = config["partition"]
    quantity = partition_config.get("quantity_skew", {"type": "none"})
    partitions = partition_labels_dirichlet(
        bundle.train_targets,
        num_clients=int(partition_config["num_clients"]),
        alpha=float(partition_config["dirichlet_alpha"]),
        min_client_samples=int(partition_config["min_client_samples"]),
        seed=seed,
    )
    quantity_type = quantity.get("type", "none")
    if quantity_type == "lognormal":
        partitions = apply_lognormal_quantity_skew(
            partitions,
            min_client_samples=int(partition_config["min_client_samples"]),
            sigma=float(quantity.get("sigma", 1.0)),
            seed=seed + 31,
        )
    elif quantity_type != "none":
        raise ValueError("quantity_skew.type must be none or lognormal.")
    splits = split_client_indices(
        partitions,
        validation_fraction=float(partition_config["validation_fraction"]),
        seed=seed + 17,
    )
    histograms = client_label_histograms(
        bundle.train_targets,
        partitions,
        num_classes=bundle.num_classes,
    )
    logger.save_json(
        "client_partitions.json",
        {
            str(client_id): {
                "all": partitions[client_id],
                "train": splits[client_id]["train"],
                "validation": splits[client_id]["validation"],
                "class_histogram": histograms[client_id],
            }
            for client_id in sorted(partitions)
        },
    )
    runtime = config.get("runtime", {})
    clients = [
        FederatedClient(
            client_id,
            bundle.train_dataset,
            bundle.train_targets,
            splits[client_id]["train"],
            splits[client_id]["validation"],
            num_classes=bundle.num_classes,
            batch_size=int(config["federated"]["batch_size"]),
            num_workers=int(runtime.get("num_workers", 0)),
            pin_memory=bool(runtime.get("pin_memory", False)),
        )
        for client_id in sorted(splits)
    ]
    return clients, splits


def run_experiment(
    config: Mapping[str, Any],
    *,
    algorithm: str,
    bundle: DatasetBundle | None = None,
    resume: str | Path | None = None,
    device_override: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one complete reproducible dataset/algorithm/seed experiment."""

    resolved = copy.deepcopy(dict(config))
    seed = int(resolved["experiment"]["seed"])
    set_global_seed(seed)
    dataset_bundle = bundle or build_dataset(resolved["dataset"], seed=seed)
    resolved["dataset"]["num_classes"] = dataset_bundle.num_classes
    resolved["model"]["input_dim"] = dataset_bundle.input_dim
    options = resolve_algorithm(algorithm, resolved["fedrba"])
    run_label = str(resolved["experiment"].get("variant", algorithm))
    alpha = float(resolved["partition"]["dirichlet_alpha"])
    scenario = str(
        resolved["experiment"].get(
            "scenario", f"alpha_{format(alpha, 'g').replace('.', 'p')}"
        )
    )

    output_dir = (
        Path(str(resolved["experiment"]["output_dir"]))
        / str(resolved["dataset"]["name"])
        / run_label
        / scenario
        / f"seed_{seed}"
    )
    if overwrite and output_dir.exists() and resume is None:
        shutil.rmtree(output_dir)
    logger = ExperimentLogger(output_dir)
    if logger.metrics_path.exists() and resume is None:
        raise FileExistsError(
            f"Result directory already contains metrics: {output_dir}. "
            "Use --resume or --overwrite."
        )
    save_config(resolved, output_dir / "config.yaml")
    logger.save_json(
        "dataset_metadata.json",
        {
            "input_dim": dataset_bundle.input_dim,
            "num_classes": dataset_bundle.num_classes,
            "minority_class": dataset_bundle.minority_class,
            "feature_names": dataset_bundle.feature_names,
            "class_names": dataset_bundle.class_names,
            "metadata": dataset_bundle.metadata,
        },
    )

    clients, _ = _build_clients(
        dataset_bundle, resolved, seed=seed, logger=logger
    )
    model = MLPClassifier(
        input_dim=dataset_bundle.input_dim,
        hidden_dims=resolved["model"]["hidden_dims"],
        num_classes=dataset_bundle.num_classes,
        dropout=float(resolved["model"].get("dropout", 0.2)),
    ).cpu()
    q = np.full(dataset_bundle.num_classes, 1.0 / dataset_bundle.num_classes)
    selection_rng = np.random.default_rng(seed + 101)
    start_round = 1
    if resume is not None:
        checkpoint_path = Path(resume)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint["algorithm"] != algorithm:
            raise ValueError("Checkpoint algorithm differs from requested algorithm.")
        if checkpoint["dataset_name"] != resolved["dataset"]["name"]:
            raise ValueError("Checkpoint dataset differs from requested dataset.")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        q = np.asarray(checkpoint["blindspot_weights"], dtype=np.float64)
        selection_rng.bit_generator.state = checkpoint["selection_rng_state"]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if (
            torch.cuda.is_available()
            and checkpoint.get("cuda_rng_state_all") is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        start_round = int(checkpoint["round"]) + 1

    runtime = resolved.get("runtime", {})
    device = resolve_device(
        device_override or str(runtime.get("device", "auto"))
    )
    rounds = int(resolved["federated"]["rounds"])
    clients_per_round = int(resolved["federated"]["clients_per_round"])
    if clients_per_round > len(clients):
        raise ValueError("clients_per_round exceeds the number of clients.")
    history = logger.read_history()
    final_checkpoint: Path | None = None

    print(
        f"dataset={resolved['dataset']['name']} algorithm={algorithm} seed={seed} "
        f"device={device} rounds={rounds}"
    )
    for round_number in range(start_round, rounds + 1):
        selected_ids = sorted(
            int(value)
            for value in selection_rng.choice(
                len(clients), size=clients_per_round, replace=False
            )
        )
        old_q = q.copy()
        updates = []
        for client_id in selected_ids:
            client_seed = seed * 1_000_003 + round_number * 1_009 + client_id
            updates.append(
                clients[client_id].train(
                    model,
                    old_q,
                    device=device,
                    optimizer_config=resolved["optimizer"],
                    local_epochs=int(resolved["federated"]["local_epochs"]),
                    blindspot_strength=float(
                        resolved["fedrba"]["blindspot_strength"]
                    ),
                    use_blindspot_loss=options.use_blindspot_loss,
                    use_fedprox=options.use_fedprox,
                    fedprox_mu=float(resolved["fedprox"]["mu"]),
                    seed=client_seed,
                )
            )
        if options.update_blindspot:
            q, _ = update_blindspot_weights(
                old_q,
                [item.validation_statistics for item in updates],
                temperature=float(
                    resolved["fedrba"]["blindspot_temperature"]
                ),
                ema=float(resolved["fedrba"]["blindspot_ema"]),
            )
        else:
            q = old_q

        global_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        if options.use_rba_aggregation:
            new_state, report = aggregate_fedrba(
                global_state,
                updates,
                q,
                server_lr=float(resolved["fedrba"]["server_lr"]),
                reliability_tau=float(resolved["fedrba"]["reliability_tau"]),
                alignment_min=float(resolved["fedrba"]["alignment_min"]),
                alignment_power=float(resolved["fedrba"]["alignment_power"]),
                blindspot_update_scale=float(
                    resolved["fedrba"]["blindspot_update_scale"]
                ),
                blindspot_scale_min=float(
                    resolved["fedrba"]["blindspot_scale_min"]
                ),
                blindspot_scale_max=float(
                    resolved["fedrba"]["blindspot_scale_max"]
                ),
                use_classwise_aggregation=options.use_classwise_aggregation,
                use_update_alignment=options.use_update_alignment,
                use_blindspot_server_scaling=options.use_blindspot_server_scaling,
            )
        else:
            new_state, report = aggregate_fedavg(
                global_state, updates, server_lr=1.0
            )
        model.load_state_dict(new_state, strict=True)

        metrics, _, _ = evaluate_model(
            model,
            dataset_bundle.test_dataset,
            device=device,
            batch_size=int(resolved["federated"]["batch_size"]),
            num_classes=dataset_bundle.num_classes,
            minority_class=dataset_bundle.minority_class,
            num_workers=int(runtime.get("num_workers", 0)),
            pin_memory=bool(runtime.get("pin_memory", False)),
        )
        total_train = sum(item.train_sample_count for item in updates)
        train_loss = sum(
            item.train_loss * item.train_sample_count for item in updates
        ) / total_train
        record = {
            "round": round_number,
            "train_loss": train_loss,
            "test_loss": metrics["test_loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "minority_recall": metrics["minority_recall"],
            "minority_f1": metrics["minority_f1"],
            "recall_per_class": metrics["recall_per_class"],
            "precision_per_class": metrics["precision_per_class"],
            "f1_per_class": metrics["f1_per_class"],
            "blindspot_weights": q.astype(float).tolist(),
            "selected_clients": selected_ids,
            "client_aggregation_weights": report.backbone_weights,
            "classwise_aggregation_weights": report.classwise_weights,
            "confusion_matrix": metrics["confusion_matrix"],
        }
        logger.log_round(record)
        history.append(record)
        print(
            f"round={round_number:04d} clients={selected_ids} "
            f"loss={metrics['test_loss']:.4f} acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"minority_recall={metrics['minority_recall']:.4f}"
        )
        save_every = int(resolved["evaluation"]["save_every"])
        final_checkpoint = _save_checkpoint(
            logger,
            round_number=round_number,
            model=model,
            blindspot_weights=q,
            selection_rng=selection_rng,
            algorithm=algorithm,
            dataset_name=str(resolved["dataset"]["name"]),
            periodic=(round_number % save_every == 0 or round_number == rounds),
        )

    if not history:
        raise RuntimeError("No round was executed and no previous history was found.")
    final_metrics = dict(history[-1])
    final_metrics["algorithm"] = algorithm
    final_metrics["method"] = run_label
    final_metrics["scenario"] = scenario
    final_metrics["dirichlet_alpha"] = alpha
    final_metrics["dataset"] = str(resolved["dataset"]["name"])
    final_metrics["seed"] = seed
    final_metrics["checkpoint"] = str(final_checkpoint or resume)
    logger.save_json("final_metrics.json", final_metrics)
    generate_run_plots(
        history,
        plot_dir=logger.plot_dir,
        num_clients=len(clients),
        num_classes=dataset_bundle.num_classes,
        class_names=dataset_bundle.class_names,
    )
    return final_metrics
