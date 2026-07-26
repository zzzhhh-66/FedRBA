"""Append-only experiment artifacts and JSON-safe serialization."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite numbers and NumPy-like scalars."""

    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class ExperimentLogger:
    """Manage one ``dataset/algorithm/seed`` result directory."""

    nested_fields = {
        "recall_per_class",
        "precision_per_class",
        "f1_per_class",
        "blindspot_weights",
        "selected_clients",
        "client_aggregation_weights",
        "classwise_aggregation_weights",
        "confusion_matrix",
    }

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plot_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.csv"
        self.aggregation_path = self.output_dir / "aggregation_weights.jsonl"
        self.blindspot_path = self.output_dir / "blindspot_weights.jsonl"

    def log_round(self, record: Mapping[str, Any]) -> None:
        """Append one round to CSV and dedicated JSONL audit logs."""

        safe = json_safe(dict(record))
        encoded = {
            key: (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if key in self.nested_fields
                else value
            )
            for key, value in safe.items()
        }
        write_header = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(encoded))
            if write_header:
                writer.writeheader()
            writer.writerow(encoded)
        self.append_jsonl(
            self.aggregation_path,
            {
                "round": safe["round"],
                "selected_clients": safe["selected_clients"],
                "backbone": safe["client_aggregation_weights"],
                "classifier": safe["classwise_aggregation_weights"],
            },
        )
        self.append_jsonl(
            self.blindspot_path,
            {"round": safe["round"], "weights": safe["blindspot_weights"]},
        )

    @staticmethod
    def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(value), ensure_ascii=False) + "\n")

    def save_json(self, name: str, value: Any) -> None:
        destination = self.output_dir / name
        destination.write_text(
            json.dumps(json_safe(value), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_history(self) -> list[dict[str, Any]]:
        """Read existing metrics, decoding JSON-valued columns."""

        if not self.metrics_path.is_file():
            return []
        history: list[dict[str, Any]] = []
        with self.metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                parsed: dict[str, Any] = {}
                for key, value in row.items():
                    if key in self.nested_fields:
                        parsed[key] = json.loads(value)
                    elif key == "round":
                        parsed[key] = int(value)
                    else:
                        parsed[key] = None if value == "" else float(value)
                history.append(parsed)
        return history

