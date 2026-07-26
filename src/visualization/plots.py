"""Deterministic non-interactive experiment plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _line_plot(
    history: Sequence[Mapping[str, Any]],
    field: str,
    ylabel: str,
    destination: Path,
) -> None:
    rounds = [int(item["round"]) for item in history]
    values = [float(item[field]) for item in history]
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(rounds, values, linewidth=1.8)
    axis.set_xlabel("Communication Round")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_blindspots(
    history: Sequence[Mapping[str, Any]], destination: Path
) -> None:
    weights = np.asarray(
        [item["blindspot_weights"] for item in history], dtype=np.float64
    )
    rounds = [int(item["round"]) for item in history]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for class_id in range(weights.shape[1]):
        axis.plot(rounds, weights[:, class_id], label=f"class {class_id}")
    axis.set_xlabel("Communication Round")
    axis.set_ylabel("Blindspot weight")
    if weights.shape[1] <= 10:
        axis.legend(ncol=2, fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_classwise_heatmap(
    history: Sequence[Mapping[str, Any]],
    *,
    num_clients: int,
    num_classes: int,
    destination: Path,
) -> None:
    sums = np.zeros((num_clients, num_classes), dtype=np.float64)
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for record in history:
        weights = record["classwise_aggregation_weights"]
        for client_key, values in weights.items():
            client_id = int(client_key)
            array = np.asarray(values, dtype=np.float64)
            nonzero = array > 0
            sums[client_id, nonzero] += array[nonzero]
            counts[client_id, nonzero] += 1
    averages = np.divide(
        sums, counts, out=np.zeros_like(sums), where=counts > 0
    )
    figure, axis = plt.subplots(figsize=(max(6, num_classes * 0.7), 6))
    image = axis.imshow(averages, aspect="auto", cmap="viridis")
    axis.set_xlabel("Class")
    axis.set_ylabel("Client")
    axis.set_xticks(range(num_classes))
    figure.colorbar(image, ax=axis, label="Mean aggregation weight")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_confusion_matrix(
    confusion: Sequence[Sequence[int]],
    destination: Path,
    class_names: Sequence[str] | None = None,
) -> None:
    matrix = np.asarray(confusion, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    if class_names and len(class_names) == len(matrix):
        axis.set_xticks(range(len(matrix)), class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(matrix)), class_names)
    if len(matrix) <= 10:
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(len(matrix)):
            for column in range(len(matrix)):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=8,
                )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_class_recall(
    recalls: Sequence[float],
    destination: Path,
    class_names: Sequence[str] | None = None,
) -> None:
    values = np.asarray(recalls, dtype=np.float64)
    labels = (
        list(class_names)
        if class_names and len(class_names) == len(values)
        else [str(index) for index in range(len(values))]
    )
    figure, axis = plt.subplots(figsize=(max(6, len(values) * 0.7), 4.5))
    axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(values)), labels, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Recall")
    axis.set_xlabel("Class")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def generate_run_plots(
    history: Sequence[Mapping[str, Any]],
    *,
    plot_dir: str | Path,
    num_clients: int,
    num_classes: int,
    class_names: Sequence[str] | None,
) -> None:
    """Generate all required per-run plots."""

    if not history:
        return
    destination = Path(plot_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _line_plot(history, "accuracy", "Accuracy", destination / "accuracy_vs_round.png")
    _line_plot(
        history, "macro_f1", "Macro-F1", destination / "macro_f1_vs_round.png"
    )
    _line_plot(
        history,
        "minority_recall",
        "Minority / worst-class recall",
        destination / "minority_recall_vs_round.png",
    )
    plot_blindspots(history, destination / "blindspot_weights_vs_round.png")
    plot_classwise_heatmap(
        history,
        num_clients=num_clients,
        num_classes=num_classes,
        destination=destination / "classwise_client_weight_heatmap.png",
    )
    final = history[-1]
    plot_confusion_matrix(
        final["confusion_matrix"],
        destination / "confusion_matrix.png",
        class_names,
    )
    plot_class_recall(
        final["recall_per_class"],
        destination / "recall_per_class.png",
        class_names,
    )

