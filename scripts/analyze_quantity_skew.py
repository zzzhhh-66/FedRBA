"""Analyze paired Default Credit quantity-skew experiments over three seeds."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.build_quantity_skew_matrix import scenario_name


METHODS = (
    "fedavg",
    "fedavg_classweight",
    "fedprox",
    "fedrba_classagg",
    "fedrba",
)
METRICS = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "minority_recall",
    "minority_f1",
)
EXPECTED_ROUNDS = 150


def _load_run(
    run_dir: Path,
    *,
    condition: str,
    expected_rounds: int,
    alpha: float,
    sigma: float,
) -> tuple[pd.DataFrame, str]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing quantity-skew history: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    required = {"round", *METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{metrics_path} is missing metric columns: {missing}")
    if frame["round"].duplicated().any():
        raise ValueError(f"{metrics_path} contains duplicate rounds.")
    rounds = pd.to_numeric(frame["round"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    expected = np.arange(1, expected_rounds + 1, dtype=np.float64)
    if not np.array_equal(rounds, expected):
        raise ValueError(
            f"{metrics_path} must contain contiguous rounds "
            f"1..{expected_rounds}."
        )
    numeric = frame[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or (values < -1e-12).any()
        or (values > 1.0 + 1e-12).any()
    ):
        raise ValueError(f"{metrics_path} contains invalid metrics.")

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved configuration: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    observed_alpha = float(config["partition"]["dirichlet_alpha"])
    if not np.isclose(observed_alpha, alpha):
        raise ValueError(
            f"{config_path} has alpha={observed_alpha}, expected {alpha}."
        )
    quantity = config["partition"].get("quantity_skew", {})
    observed_type = str(quantity.get("type", "none"))
    observed_sigma = float(quantity.get("sigma", 0.0))
    if condition == "none":
        if observed_type != "none":
            raise ValueError(f"{config_path} is not a no-quantity-skew run.")
    else:
        if observed_type != "lognormal" or not np.isclose(
            observed_sigma, sigma
        ):
            raise ValueError(
                f"{config_path} does not use lognormal sigma={sigma}."
            )

    partition_path = run_dir / "client_partitions.json"
    if not partition_path.is_file():
        raise FileNotFoundError(f"Missing partition audit: {partition_path}")
    digest = hashlib.sha256(partition_path.read_bytes()).hexdigest()
    return frame, digest


def _run_row(
    frame: pd.DataFrame,
    *,
    condition: str,
    method: str,
    seed: int,
    window: int,
    collapse_threshold: float,
    partition_sha256: str,
) -> dict[str, Any]:
    selected = frame.tail(window)
    row: dict[str, Any] = {
        "condition": condition,
        "method": method,
        "seed": seed,
        "window": window,
        "start_round": int(selected["round"].iloc[0]),
        "end_round": int(selected["round"].iloc[-1]),
        "partition_sha256": partition_sha256,
    }
    for metric in METRICS:
        values = selected[metric].to_numpy(dtype=np.float64)
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_temporal_std"] = float(values.std(ddof=1))
    collapsed = (
        selected["minority_recall"].to_numpy(dtype=np.float64)
        <= collapse_threshold
    )
    row["minority_collapse_rate"] = float(collapsed.mean())
    return row


def _aggregate(per_run: pd.DataFrame, expected_seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, method), group in per_run.groupby(
        ["condition", "method"], sort=True
    ):
        seeds = sorted(int(value) for value in group["seed"])
        if seeds != expected_seeds:
            raise ValueError(
                f"Expected seeds {expected_seeds} for {condition}/{method}; "
                f"got {seeds}."
            )
        row: dict[str, Any] = {
            "condition": condition,
            "method": method,
            "n_seeds": len(seeds),
            "window": int(group["window"].iloc[0]),
        }
        for metric in METRICS:
            seed_means = group[f"{metric}_mean"].to_numpy(dtype=np.float64)
            temporal = group[f"{metric}_temporal_std"].to_numpy(
                dtype=np.float64
            )
            row[f"{metric}_mean"] = float(seed_means.mean())
            row[f"{metric}_std"] = float(seed_means.std(ddof=1))
            row[f"{metric}_temporal_std_mean"] = float(temporal.mean())
        collapse = group["minority_collapse_rate"].to_numpy(dtype=np.float64)
        row["minority_collapse_rate_mean"] = float(collapse.mean())
        row["minority_collapse_rate_std"] = float(collapse.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_condition_effects(per_run: pd.DataFrame) -> pd.DataFrame:
    skew_conditions = sorted(set(per_run["condition"]).difference({"none"}))
    if len(skew_conditions) != 1:
        raise ValueError("Expected exactly one explicit quantity-skew condition.")
    skew_condition = skew_conditions[0]
    rows: list[dict[str, Any]] = []
    metrics = (*METRICS, "minority_collapse_rate")
    for method in METHODS:
        subset = per_run[per_run["method"] == method]
        for metric in metrics:
            column = (
                f"{metric}_mean"
                if metric != "minority_collapse_rate"
                else metric
            )
            pivot = subset.pivot(index="seed", columns="condition", values=column)
            difference = (
                pivot[skew_condition] - pivot["none"]
            ).to_numpy(dtype=np.float64)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "lognormal_minus_none_mean": float(difference.mean()),
                    "paired_difference_std": float(difference.std(ddof=1)),
                    "lognormal_higher_seeds": int((difference > 1e-12).sum()),
                    "none_higher_seeds": int((difference < -1e-12).sum()),
                    "ties": int((np.abs(difference) <= 1e-12).sum()),
                }
            )
    return pd.DataFrame(rows)


def _fedrba_comparisons(per_run: pd.DataFrame) -> pd.DataFrame:
    skew_conditions = sorted(set(per_run["condition"]).difference({"none"}))
    if len(skew_conditions) != 1:
        raise ValueError("Expected exactly one explicit quantity-skew condition.")
    skew = per_run[per_run["condition"] == skew_conditions[0]]
    rows: list[dict[str, Any]] = []
    metrics = (*METRICS, "minority_collapse_rate")
    for comparator in METHODS:
        if comparator == "fedrba":
            continue
        for metric in metrics:
            column = (
                f"{metric}_mean"
                if metric != "minority_collapse_rate"
                else metric
            )
            pivot = skew.pivot(index="seed", columns="method", values=column)
            difference = (
                pivot["fedrba"] - pivot[comparator]
            ).to_numpy(dtype=np.float64)
            rows.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "fedrba_minus_comparator_mean": float(difference.mean()),
                    "paired_difference_std": float(difference.std(ddof=1)),
                    "fedrba_higher_seeds": int((difference > 1e-12).sum()),
                    "comparator_higher_seeds": int((difference < -1e-12).sum()),
                    "ties": int((np.abs(difference) <= 1e-12).sum()),
                }
            )
    return pd.DataFrame(rows)


def _plot(aggregate: pd.DataFrame, destination: Path) -> None:
    skew_conditions = sorted(set(aggregate["condition"]).difference({"none"}))
    if len(skew_conditions) != 1:
        raise ValueError("Expected exactly one explicit quantity-skew condition.")
    skew_condition = skew_conditions[0]
    metrics = (
        ("roc_auc_mean", "ROC-AUC"),
        ("pr_auc_mean", "PR-AUC"),
        ("minority_recall_mean", "Minority recall"),
        (
            "minority_collapse_rate_mean",
            "Minority-collapse rate (lower is better)",
        ),
    )
    colors = {"none": "#8d99ae", skew_condition: "#1f4e79"}
    sigma_label = skew_condition.removeprefix("lognormal_sigma_")
    labels = {
        "none": "No explicit quantity skew",
        skew_condition: f"Log-normal σ={sigma_label}",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    x = np.arange(len(METHODS))
    width = 0.38
    for axis, (metric, title) in zip(axes.flat, metrics):
        for index, condition in enumerate(("none", skew_condition)):
            group = (
                aggregate[aggregate["condition"] == condition]
                .set_index("method")
                .reindex(METHODS)
            )
            values = group[metric].to_numpy(dtype=np.float64)
            std_column = metric.replace("_mean", "_std")
            errors = group[std_column].to_numpy(dtype=np.float64)
            offset = (index - 0.5) * width
            axis.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=3,
                color=colors[condition],
                label=labels[condition],
                alpha=0.9,
            )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(
            ["FedAvg", "ClassWeight", "FedProx", "ClassAgg", "FedRBA"],
            rotation=18,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Default Credit quantity-skew comparison "
        "(last-20-round mean ± seed std)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)


def analyze(
    baseline_root: str | Path,
    skew_root: str | Path,
    output_root: str | Path,
    *,
    seeds: Iterable[int] = (1, 2, 3),
    alpha: float = 0.3,
    sigma: float = 1.0,
    window: int = 20,
    collapse_threshold: float = 1e-12,
) -> dict[str, Path]:
    """Validate all 30 paired runs and write paper-ready artifacts."""

    selected_seeds = sorted(int(value) for value in seeds)
    if (
        not selected_seeds
        or selected_seeds != sorted(set(selected_seeds))
        or any(value < 0 for value in selected_seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers.")
    if window <= 1 or window > EXPECTED_ROUNDS:
        raise ValueError(f"window must be in [2, {EXPECTED_ROUNDS}].")
    if collapse_threshold < 0.0 or not np.isfinite(collapse_threshold):
        raise ValueError("collapse_threshold must be finite and non-negative.")

    conditions = {
        "none": (Path(baseline_root), f"alpha_{format(alpha, 'g').replace('.', 'p')}"),
        f"lognormal_sigma_{format(sigma, 'g')}": (
            Path(skew_root),
            scenario_name(alpha=alpha, sigma=sigma),
        ),
    }
    rows: list[dict[str, Any]] = []
    hashes: dict[tuple[str, int], set[str]] = {}
    for condition, (root, scenario) in conditions.items():
        for method in METHODS:
            for seed in selected_seeds:
                run_dir = (
                    root
                    / "default_credit"
                    / method
                    / scenario
                    / f"seed_{seed}"
                )
                frame, digest = _load_run(
                    run_dir,
                    condition=condition,
                    expected_rounds=EXPECTED_ROUNDS,
                    alpha=alpha,
                    sigma=sigma,
                )
                hashes.setdefault((condition, seed), set()).add(digest)
                rows.append(
                    _run_row(
                        frame,
                        condition=condition,
                        method=method,
                        seed=seed,
                        window=window,
                        collapse_threshold=collapse_threshold,
                        partition_sha256=digest,
                    )
                )

    expected_runs = 2 * len(METHODS) * len(selected_seeds)
    if len(rows) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} runs, found {len(rows)}.")
    for key, values in hashes.items():
        if len(values) != 1:
            raise ValueError(
                f"Methods do not share an identical partition for {key}."
            )
    for seed in selected_seeds:
        if hashes[("none", seed)] == hashes[(f"lognormal_sigma_{format(sigma, 'g')}", seed)]:
            raise ValueError(
                f"Seed {seed} has identical baseline and quantity-skew partitions."
            )

    per_run = pd.DataFrame(rows).sort_values(
        ["condition", "method", "seed"]
    )
    aggregate = _aggregate(per_run, selected_seeds)
    condition_effects = _paired_condition_effects(per_run)
    fedrba_comparisons = _fedrba_comparisons(per_run)

    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_run": destination / "quantity_skew_per_run.csv",
        "aggregate": destination / "quantity_skew_aggregate.csv",
        "condition_effects": destination / "quantity_skew_condition_effects.csv",
        "fedrba_comparisons": destination
        / "quantity_skew_fedrba_comparisons.csv",
        "plot": destination / "quantity_skew_comparison.png",
    }
    per_run.to_csv(outputs["per_run"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    condition_effects.to_csv(outputs["condition_effects"], index=False)
    fedrba_comparisons.to_csv(outputs["fedrba_comparisons"], index=False)
    _plot(aggregate, outputs["plot"])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default="outputs")
    parser.add_argument("--skew-root", default="quantity_skew_outputs")
    parser.add_argument(
        "--output-root", default="quantity_skew_analysis"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--collapse-threshold", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = analyze(
        args.baseline_root,
        args.skew_root,
        args.output_root,
        seeds=args.seeds,
        alpha=args.alpha,
        sigma=args.sigma,
        window=args.window,
        collapse_threshold=args.collapse_threshold,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
