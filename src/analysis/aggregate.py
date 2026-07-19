"""Validate experiment JSON, aggregate repeated seeds, and mark Pareto points."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


RESULT_METRICS = (
    "peak_memory_gb",
    "tokens_per_second",
    "training_time_seconds",
    "exact_match",
    "trainable_parameters",
    "checkpoint_size_mb",
)
GROUP_COLUMNS = ("sweep", "method", "rank", "max_length", "micro_batch_size")
REQUIRED_COLUMNS = GROUP_COLUMNS[1:] + RESULT_METRICS


def validate_result_record(
    record: Mapping[str, Any], source: str = "record"
) -> dict[str, Any]:
    missing = [field for field in REQUIRED_COLUMNS if field not in record]
    if missing:
        raise ValueError(f"{source} is missing required fields: {missing}")
    normalized = dict(record)
    normalized["sweep"] = str(
        normalized.get("sweep", normalized.get("experiment_sweep", "unspecified"))
    )
    method = str(normalized["method"])
    if method not in {"full_ft", "lora"}:
        raise ValueError(f"{source}: method must be 'full_ft' or 'lora'")
    normalized["method"] = method
    rank = normalized["rank"]
    if method == "lora" and (rank is None or int(rank) <= 0):
        raise ValueError(f"{source}: LoRA requires a positive rank")
    normalized["rank"] = None if method == "full_ft" else int(rank)

    for field in ("max_length", "micro_batch_size", "trainable_parameters"):
        if isinstance(normalized[field], bool) or int(normalized[field]) <= 0:
            raise ValueError(f"{source}: {field} must be a positive integer")
        normalized[field] = int(normalized[field])
    for field in (
        "peak_memory_gb",
        "tokens_per_second",
        "training_time_seconds",
        "checkpoint_size_mb",
    ):
        if isinstance(normalized[field], bool) or float(normalized[field]) < 0:
            raise ValueError(f"{source}: {field} must be non-negative")
        normalized[field] = float(normalized[field])
    exact_match = float(normalized["exact_match"])
    if not 0.0 <= exact_match <= 1.0:
        raise ValueError(f"{source}: exact_match must be in [0, 1]")
    normalized["exact_match"] = exact_match
    if "seed" in normalized and normalized["seed"] is not None:
        normalized["seed"] = int(normalized["seed"])
    return normalized


def _records_from_json(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, Mapping):
        if "runs" in value and isinstance(value["runs"], list):
            return value["runs"]
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError(f"{path} must contain an object, a list, or an object with runs")


def load_result_records(
    inputs: Sequence[str | Path] | Iterable[str | Path],
) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        elif path.suffix == ".json":
            paths.append(path)
        else:
            raise ValueError(f"result input must be a JSON file or directory: {path}")
    if not paths:
        raise ValueError("no JSON result files found")

    records: list[dict[str, Any]] = []
    for path in paths:
        for index, record in enumerate(_records_from_json(path)):
            if not isinstance(record, Mapping):
                raise ValueError(f"{path} record {index} is not an object")
            status = str(record.get("status", "completed"))
            if status not in {"completed", "success"}:
                continue
            normalized = validate_result_record(record, f"{path} record {index}")
            normalized["source_file"] = str(path)
            records.append(normalized)
    if not records:
        raise ValueError("no completed result records found")
    return records


def load_result_attempts(
    inputs: Sequence[str | Path] | Iterable[str | Path],
) -> list[dict[str, Any]]:
    """Load minimal success/OOM records for the maximum-batch analysis."""

    paths: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        paths.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    attempts: list[dict[str, Any]] = []
    for path in paths:
        for index, record in enumerate(_records_from_json(path)):
            if not isinstance(record, Mapping):
                continue
            required = ("method", "max_length", "micro_batch_size")
            if any(field not in record for field in required):
                continue
            method = str(record["method"])
            if method not in {"full_ft", "lora"}:
                raise ValueError(f"{path} record {index}: invalid method {method}")
            status = str(record.get("status", "completed"))
            if status not in {"completed", "success", "oom"}:
                continue
            attempts.append(
                {
                    "sweep": str(record.get("sweep", "unspecified")),
                    "method": method,
                    "rank": None if method == "full_ft" else int(record["rank"]),
                    "max_length": int(record["max_length"]),
                    "micro_batch_size": int(record["micro_batch_size"]),
                    "status": status,
                }
            )
    return attempts


def summarize_batch_feasibility(attempts: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Report the largest successful batch and first observed OOM per method."""

    if not attempts:
        return pd.DataFrame(
            columns=[
                "sweep",
                "method",
                "rank",
                "max_length",
                "maximum_feasible_micro_batch",
                "first_oom_micro_batch",
            ]
        )
    frame = pd.DataFrame(attempts)
    rows: list[dict[str, Any]] = []
    keys = ["sweep", "method", "rank", "max_length"]
    for group_values, group in frame.groupby(keys, dropna=False):
        successful = group[group["status"].isin(["completed", "success"])][
            "micro_batch_size"
        ]
        oom = group[group["status"] == "oom"]["micro_batch_size"]
        rows.append(
            {
                **dict(zip(keys, group_values)),
                "maximum_feasible_micro_batch": (
                    int(successful.max()) if not successful.empty else None
                ),
                "first_oom_micro_batch": int(oom.min()) if not oom.empty else None,
            }
        )
    return pd.DataFrame(rows)


def mark_pareto_efficient(
    frame: pd.DataFrame,
    *,
    memory_column: str = "peak_memory_gb_mean",
    throughput_column: str = "tokens_per_second_mean",
    quality_column: str = "exact_match_mean",
) -> pd.DataFrame:
    """Mark points not dominated on memory (min), speed and quality (max)."""

    result = frame.copy()
    required = (memory_column, throughput_column, quality_column)
    missing = [column for column in required if column not in result]
    if missing:
        raise ValueError(f"cannot compute Pareto frontier; missing columns: {missing}")
    values = result.loc[:, required].astype(float)
    efficient: list[bool] = []
    for index, point in values.iterrows():
        dominated = False
        for other_index, other in values.iterrows():
            if index == other_index:
                continue
            no_worse = (
                other[memory_column] <= point[memory_column]
                and other[throughput_column] >= point[throughput_column]
                and other[quality_column] >= point[quality_column]
            )
            strictly_better = (
                other[memory_column] < point[memory_column]
                or other[throughput_column] > point[throughput_column]
                or other[quality_column] > point[quality_column]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        efficient.append(not dominated)
    result["pareto_efficient"] = efficient
    return result


def aggregate_results(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not records:
        raise ValueError("records must not be empty")
    validated = [
        validate_result_record(record, f"record {index}")
        for index, record in enumerate(records)
    ]
    frame = pd.DataFrame(validated)
    aggregations = {metric: ["mean", "std"] for metric in RESULT_METRICS}
    grouped = frame.groupby(list(GROUP_COLUMNS), dropna=False).agg(aggregations)
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    counts = (
        frame.groupby(list(GROUP_COLUMNS), dropna=False)
        .size()
        .reset_index(name="n_runs")
    )
    grouped = grouped.merge(counts, on=list(GROUP_COLUMNS), how="left")
    # Pareto dominance only makes sense among points from the same experiment;
    # otherwise, for example, a max-batch run could incorrectly dominate a main run.
    return pd.concat(
        [mark_pareto_efficient(group) for _, group in grouped.groupby("sweep")],
        ignore_index=True,
    )


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            {
                key: (None if isinstance(value, float) and math.isnan(value) else value)
                for key, value in row.items()
            }
        )
    return records


def write_aggregates(frame: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe_records(frame), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="result JSON file(s) or directories")
    parser.add_argument("--csv", type=Path, default=Path("reports/aggregate.csv"))
    parser.add_argument("--json", type=Path, default=Path("reports/aggregate.json"))
    parser.add_argument(
        "--batch-csv",
        type=Path,
        default=Path("reports/batch_feasibility.csv"),
        help="maximum successful and first OOM micro-batch summary",
    )
    args = parser.parse_args()
    records = load_result_records(args.inputs)
    frame = aggregate_results(records)
    write_aggregates(frame, args.csv, args.json)
    attempts = load_result_attempts(args.inputs)
    feasibility = summarize_batch_feasibility(attempts)
    args.batch_csv.parent.mkdir(parents=True, exist_ok=True)
    feasibility.to_csv(args.batch_csv, index=False)
    print(f"Aggregated {len(records)} completed runs into {len(frame)} configurations")


if __name__ == "__main__":
    main()
