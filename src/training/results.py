'''
Training result dataclasses.

Defines TrainingResult, EvaluationResult, and ExperimentResult.
'''

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal


# ============================================================================
# Result dataclasses
# ============================================================================


class StopReason(str, Enum):
    """Reasons why training stopped."""
    MAX_STEPS = "max_steps"
    TOKEN_BUDGET = "token_budget"
    DATA_EXHAUSTED = "data_exhausted"


@dataclass(frozen=True)
class TrainingResult:
    """Immutable container for training metrics."""
    optimizer_steps: int
    micro_steps: int
    trained_non_padding_tokens: int
    final_loss: float
    mean_loss: float
    training_time_seconds: float
    measured_time_seconds: float
    tokens_per_second: float
    peak_memory_gb: float
    peak_reserved_memory_gb: float
    stop_reason: str


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable container for evaluation metrics."""
    exact_match: float
    correct: int
    total: int
    unparseable: int
    generation_time_seconds: float


@dataclass(frozen=True)
class ExperimentResult:
    """Immutable container for the complete experiment result."""
    run_id: str
    experiment_name: str
    method: str
    rank: int | None
    max_length: int
    micro_batch_size: int
    effective_batch_size: int
    gradient_accumulation_steps: int
    peak_memory_gb: float | None
    peak_reserved_memory_gb: float | None
    tokens_per_second: float | None
    training_time_seconds: float | None
    exact_match: float | None
    trainable_parameters: int | None
    total_parameters: int | None
    checkpoint_size_mb: float | None
    trained_non_padding_tokens: int | None
    optimizer_steps: int | None
    seed: int
    sweep: str
    model_name: str
    model_revision: str
    dataset_revision: str
    attention_backend: str | None
    status: Literal["completed", "oom", "failed"]
    error_type: str | None
    error_message: str | None


# ============================================================================
# Serialization and validation
# ============================================================================


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value to JSON-safe types."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return {k: _serialize_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    return value


def experiment_result_to_dict(result: ExperimentResult) -> dict[str, Any]:
    """
    Serialize ExperimentResult to a JSON-safe dictionary.

    Responsibilities:
    - serialize enum, path, and dataclass fields
    - preserve None for unavailable failure metrics
    - produce valid JSON
    """
    return _serialize_value(result)


def validate_experiment_result(result: ExperimentResult) -> None:
    """
    Validate an ExperimentResult before writing.

    Responsibilities:
    - enforce required identifiers
    - enforce positive metrics for completed runs where applicable
    - require error_type and error_message for failed runs
    - reject NaN and infinity
    - reject placeholder trainable parameter counts
    """
    errors: list[str] = []

    # Required identifiers
    if not result.run_id:
        errors.append("run_id must be nonempty")
    if not result.experiment_name:
        errors.append("experiment_name must be nonempty")
    if not result.method:
        errors.append("method must be nonempty")
    if not result.model_name:
        errors.append("model_name must be nonempty")
    if not result.model_revision:
        errors.append("model_revision must be nonempty")
    if not result.dataset_revision:
        errors.append("dataset_revision must be nonempty")

    # Status validation
    if result.status not in ("completed", "oom", "failed"):
        errors.append(f"status must be 'completed', 'oom', or 'failed', got '{result.status}'")

    # Failed/OOM runs must have error info
    if result.status in ("failed", "oom"):
        if not result.error_type:
            errors.append("error_type is required for failed/oom runs")
        if not result.error_message:
            errors.append("error_message is required for failed/oom runs")

    # Completed runs validation
    if result.status == "completed":
        if result.optimizer_steps is None or result.optimizer_steps <= 0:
            errors.append("completed runs must have positive optimizer_steps")
        if result.trained_non_padding_tokens is None or result.trained_non_padding_tokens <= 0:
            errors.append("completed runs must have positive trained_non_padding_tokens")

    # NaN/Infinity rejection for numeric fields
    numeric_fields = [
        ("peak_memory_gb", result.peak_memory_gb),
        ("peak_reserved_memory_gb", result.peak_reserved_memory_gb),
        ("tokens_per_second", result.tokens_per_second),
        ("training_time_seconds", result.training_time_seconds),
        ("exact_match", result.exact_match),
    ]

    for name, value in numeric_fields:
        if value is not None and (math.isnan(value) or math.isinf(value)):
            errors.append(f"{name} must not be NaN or infinity, got {value}")

    # Placeholder rejection
    if result.trainable_parameters is not None and result.trainable_parameters <= 0:
        errors.append("trainable_parameters must be positive when set")
    if result.total_parameters is not None and result.total_parameters <= 0:
        errors.append("total_parameters must be positive when set")

    if errors:
        raise ValueError(
            "ExperimentResult validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def write_json_atomic(data: dict[str, Any], path: Path) -> None:
    """
    Write JSON data atomically.

    Responsibilities:
    - write to a temporary file
    - flush and close
    - atomically replace the destination
    - create parent directories
    - avoid partially written results after job termination
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        suffix=".json.tmp",
        prefix=".result_",
    )
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()

        Path(tmp_path).rename(path)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise
