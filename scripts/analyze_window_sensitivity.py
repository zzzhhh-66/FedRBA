"""Audit 10/20/30-round sensitivity and GMSC temporal instability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_SCENARIOS = {
    "fashion_mnist": ("alpha_0p1", "alpha_0p3", "alpha_1"),
    "default_credit": ("alpha_0p1", "alpha_0p3"),
    "give_me_some_credit": ("alpha_0p3",),
}
EXPECTED_ROUNDS = {
    "fashion_mnist": 100,
    "default_credit": 150,
    "give_me_some_credit": 150,
}
METHODS = (
    "fedavg",
    "fedavg_classweight",
    "fedprox",
    "fedrba",
    "fedrba_classagg",
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
GMSC_PLOT_METRICS = (
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("minority_recall", "Minority recall"),
)


def _validate_windows(values: Iterable[int]) -> list[int]:
    windows = sorted(int(value) for value in values)
    if not windows or windows != sorted(set(windows)):
        raise ValueError("windows must be non-empty unique integers.")
    if any(value <= 1 for value in windows):
        raise ValueError("each window must contain at least two rounds.")
    required = {10, 20, 30}
    if not required.issubset(windows):
        raise ValueError(
            "windows must include 10, 20, and 30; 20 is the primary window."
        )
    return windows


def _load_run_metrics(
    path: Path,
    *,
    expected_rounds: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing run history: {path}")
    frame = pd.read_csv(path)
    required = {"round", *METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing metric columns: {missing}")
    if frame.empty or frame["round"].duplicated().any():
        raise ValueError(f"{path} has empty or duplicate round records.")
    raw_rounds = pd.to_numeric(frame["round"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(raw_rounds).all()
        or not np.equal(raw_rounds, np.floor(raw_rounds)).all()
    ):
        raise ValueError(f"{path} contains invalid round numbers.")
    rounds = raw_rounds.astype(np.int64)
    expected = np.arange(1, expected_rounds + 1, dtype=np.int64)
    if not np.array_equal(rounds, expected):
        raise ValueError(
            f"{path} must contain contiguous rounds 1..{expected_rounds}; "
            f"observed {rounds[:1].tolist()}..{rounds[-1:].tolist()}."
        )
    numeric = frame[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite metric values.")
    if (values < -1e-12).any() or (values > 1.0 + 1e-12).any():
        raise ValueError(f"{path} contains classification metrics outside [0, 1].")
    return frame


def _run_window_rows(
    frame: pd.DataFrame,
    *,
    dataset: str,
    scenario: str,
    method: str,
    seed: int,
    windows: list[int],
    collapse_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    end_round = int(frame["round"].iloc[-1])
    for window in windows:
        if window > len(frame):
            raise ValueError(
                f"Window {window} exceeds {len(frame)} rounds for "
                f"{dataset}/{scenario}/{method}/seed_{seed}."
            )
        selected = frame.tail(window)
        row: dict[str, Any] = {
            "dataset": dataset,
            "scenario": scenario,
            "method": method,
            "seed": seed,
            "window": window,
            "start_round": end_round - window + 1,
            "end_round": end_round,
            "observed_rounds": len(frame),
        }
        for metric in METRICS:
            values = selected[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_window_mean"] = float(np.mean(values))
            row[f"{metric}_window_std"] = float(np.std(values, ddof=1))
        collapsed = (
            selected["minority_recall"].to_numpy(dtype=np.float64)
            <= collapse_threshold
        )
        row["minority_collapse_rounds"] = int(collapsed.sum())
        row["minority_collapse_rate"] = float(collapsed.mean())
        rows.append(row)
    return rows


def _aggregate_windows(
    per_run: pd.DataFrame,
    *,
    expected_seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_fields = ["dataset", "scenario", "method", "window"]
    for key, group in per_run.groupby(group_fields, sort=True):
        seeds = sorted(int(seed) for seed in group["seed"])
        if seeds != expected_seeds:
            raise ValueError(f"Expected seeds {expected_seeds} for {key}, got {seeds}.")
        row = dict(zip(group_fields, key))
        row["n_seeds"] = len(seeds)
        for metric in METRICS:
            window_means = group[f"{metric}_window_mean"].to_numpy(
                dtype=np.float64
            )
            temporal_stds = group[f"{metric}_window_std"].to_numpy(
                dtype=np.float64
            )
            row[f"{metric}_mean"] = float(np.mean(window_means))
            row[f"{metric}_std"] = float(np.std(window_means, ddof=1))
            row[f"{metric}_temporal_std_mean"] = float(
                np.mean(temporal_stds)
            )
            row[f"{metric}_temporal_std_std"] = float(
                np.std(temporal_stds, ddof=1)
            )
        collapse = group["minority_collapse_rate"].to_numpy(dtype=np.float64)
        row["minority_collapse_rate_mean"] = float(np.mean(collapse))
        row["minority_collapse_rate_std"] = float(np.std(collapse, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_sensitivity_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_fields = ["dataset", "scenario", "method"]
    sensitivity_metrics = (
        *METRICS,
        "minority_collapse_rate",
    )
    for key, group in aggregate.groupby(group_fields, sort=True):
        windows = sorted(int(value) for value in group["window"])
        row: dict[str, Any] = dict(zip(group_fields, key))
        row["windows"] = ",".join(str(value) for value in windows)
        for metric in sensitivity_metrics:
            source = (
                f"{metric}_mean"
                if metric != "minority_collapse_rate"
                else "minority_collapse_rate_mean"
            )
            values = {
                int(item["window"]): float(item[source])
                for _, item in group.iterrows()
            }
            for window, value in values.items():
                row[f"{metric}_window_{window}"] = value
            row[f"{metric}_max_minus_min"] = max(values.values()) - min(
                values.values()
            )
            row[f"{metric}_primary20_minus_window10"] = (
                values[20] - values[10]
            )
            row[f"{metric}_primary20_minus_window30"] = (
                values[20] - values[30]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _gmsc_round_tables(
    histories: list[pd.DataFrame],
    *,
    expected_seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run = pd.concat(histories, ignore_index=True).sort_values(
        ["method", "seed", "round"]
    )
    rows: list[dict[str, Any]] = []
    for key, group in per_run.groupby(["method", "round"], sort=True):
        seeds = sorted(int(value) for value in group["seed"])
        if seeds != expected_seeds:
            raise ValueError(f"Incomplete GMSC curve group {key}: seeds={seeds}.")
        row: dict[str, Any] = {"method": key[0], "round": int(key[1])}
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    return per_run, pd.DataFrame(rows)


def _plot_gmsc(
    per_run: pd.DataFrame,
    destination: Path,
    *,
    rolling_window: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    colors = {
        "fedavg": "#1f77b4",
        "fedavg_classweight": "#ff7f0e",
        "fedprox": "#2ca02c",
        "fedrba": "#d62728",
        "fedrba_classagg": "#9467bd",
    }
    for axis, (metric, title) in zip(axes.flat, GMSC_PLOT_METRICS):
        for method in METHODS:
            method_frame = per_run[per_run["method"] == method]
            seed_curves: list[np.ndarray] = []
            rounds: np.ndarray | None = None
            for _, seed_frame in method_frame.groupby("seed", sort=True):
                seed_frame = seed_frame.sort_values("round")
                rounds = seed_frame["round"].to_numpy(dtype=np.int64)
                smoothed = (
                    seed_frame[metric]
                    .rolling(rolling_window, min_periods=1)
                    .mean()
                    .to_numpy(dtype=np.float64)
                )
                seed_curves.append(smoothed)
            matrix = np.vstack(seed_curves)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0, ddof=1)
            assert rounds is not None
            axis.plot(
                rounds,
                mean,
                color=colors[method],
                linewidth=1.8,
                label=method,
            )
            axis.fill_between(
                rounds,
                mean - std,
                mean + std,
                color=colors[method],
                alpha=0.10,
                linewidth=0,
            )
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_xlim(1, int(per_run["round"].max()))
    axes[1, 0].set_xlabel("Round")
    axes[1, 1].set_xlabel("Round")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Give Me Some Credit: three-seed mean ± std "
        f"({rolling_window}-round trailing mean)"
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze(
    root: str | Path,
    output_root: str | Path,
    *,
    windows: Iterable[int] = (10, 20, 30),
    expected_seeds: Iterable[int] = (1, 2, 3),
    collapse_threshold: float = 1e-12,
    rolling_window: int = 10,
) -> dict[str, Path]:
    """Analyze all 90 main runs and write auditable sensitivity artifacts."""

    selected_windows = _validate_windows(windows)
    seeds = sorted(int(value) for value in expected_seeds)
    if not seeds or seeds != sorted(set(seeds)):
        raise ValueError("expected_seeds must be non-empty and unique.")
    if not np.isfinite(collapse_threshold) or collapse_threshold < 0.0:
        raise ValueError("collapse_threshold must be finite and non-negative.")
    if rolling_window <= 1:
        raise ValueError("rolling_window must be greater than one.")

    source = Path(root)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    per_run_rows: list[dict[str, Any]] = []
    gmsc_histories: list[pd.DataFrame] = []
    discovered = 0
    for dataset, scenarios in DATASET_SCENARIOS.items():
        for scenario in scenarios:
            for method in METHODS:
                for seed in seeds:
                    run_dir = source / dataset / method / scenario / f"seed_{seed}"
                    history = _load_run_metrics(
                        run_dir / "metrics.csv",
                        expected_rounds=EXPECTED_ROUNDS[dataset],
                    )
                    per_run_rows.extend(
                        _run_window_rows(
                            history,
                            dataset=dataset,
                            scenario=scenario,
                            method=method,
                            seed=seed,
                            windows=selected_windows,
                            collapse_threshold=collapse_threshold,
                        )
                    )
                    if dataset == "give_me_some_credit":
                        curve = history[["round", *METRICS]].copy()
                        curve.insert(0, "seed", seed)
                        curve.insert(0, "method", method)
                        gmsc_histories.append(curve)
                    discovered += 1
    expected_runs = 30 * len(seeds)
    if discovered != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} runs, found {discovered}.")

    per_run = pd.DataFrame(per_run_rows).sort_values(
        ["dataset", "scenario", "method", "seed", "window"]
    )
    aggregate = _aggregate_windows(per_run, expected_seeds=seeds)
    sensitivity = _build_sensitivity_table(aggregate)
    gmsc_per_run, gmsc_aggregate = _gmsc_round_tables(
        gmsc_histories, expected_seeds=seeds
    )

    outputs = {
        "per_run": destination / "window_sensitivity_per_run.csv",
        "aggregate": destination / "window_sensitivity_aggregate.csv",
        "comparison": destination / "window_sensitivity_comparison.csv",
        "gmsc_per_run": destination / "gmsc_round_curves_per_run.csv",
        "gmsc_aggregate": destination / "gmsc_round_curves_aggregate.csv",
        "gmsc_plot": destination / "gmsc_three_seed_curves.png",
    }
    per_run.to_csv(outputs["per_run"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    sensitivity.to_csv(outputs["comparison"], index=False)
    gmsc_per_run.to_csv(outputs["gmsc_per_run"], index=False)
    gmsc_aggregate.to_csv(outputs["gmsc_aggregate"], index=False)
    _plot_gmsc(
        gmsc_per_run,
        outputs["gmsc_plot"],
        rolling_window=rolling_window,
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output-root", default="window_sensitivity_outputs")
    parser.add_argument(
        "--windows", nargs="+", type=int, default=[10, 20, 30]
    )
    parser.add_argument(
        "--expected-seeds", nargs="+", type=int, default=[1, 2, 3]
    )
    parser.add_argument("--collapse-threshold", type=float, default=1e-12)
    parser.add_argument("--rolling-window", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = analyze(
        args.root,
        args.output_root,
        windows=args.windows,
        expected_seeds=args.expected_seeds,
        collapse_threshold=args.collapse_threshold,
        rolling_window=args.rolling_window,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
