'''
Training result dataclasses.

Defines TrainingResult, EvaluationResult, and ExperimentResult.
'''

from __future__ import annotations

import json
import math
import os
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
    # Primary reported rule: grade the first generated turn only.
    exact_match_first_turn: float = 0.0
    first_turn_correct: int = 0


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
    learning_rate: float | None = None
    evaluation_split: str = "test"
    exact_match_first_turn: float | None = None
    validation_loss: float | None = None


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

    Enforces identity, method-specific rank semantics, status-specific
    fields, completed metrics, finite numeric values, and parameter
    count consistency.
    """
    errors: list[str] = []

    # ------------------------------------------------------------------
    # Required identifiers
    # ------------------------------------------------------------------
    if not result.run_id:
        errors.append("run_id must be nonempty")

    if not result.experiment_name:
        errors.append(
            "experiment_name must be nonempty"
        )

    if not result.method:
        errors.append("method must be nonempty")

    if not result.model_name:
        errors.append("model_name must be nonempty")

    if not result.model_revision:
        errors.append(
            "model_revision must be nonempty"
        )

    if not result.dataset_revision:
        errors.append(
            "dataset_revision must be nonempty"
        )

    # ------------------------------------------------------------------
    # Method and rank contract
    # ------------------------------------------------------------------
    if result.method not in ("lora", "full_ft"):
        errors.append(
            "method must be 'lora' or 'full_ft', "
            f"got {result.method!r}"
        )

    if result.method == "lora":
        rank_is_valid = (
            isinstance(result.rank, int)
            and not isinstance(result.rank, bool)
            and result.rank > 0
        )

        if not rank_is_valid:
            errors.append(
                "LoRA runs require rank to be "
                "a positive integer"
            )

    if (
        result.method == "full_ft"
        and result.rank is not None
    ):
        errors.append(
            "Full FT runs require rank=None"
        )

    # ------------------------------------------------------------------
    # Status contract
    # ------------------------------------------------------------------
    if result.evaluation_split not in (
        "validation",
        "test",
    ):
        errors.append(
            "evaluation_split must be 'validation' "
            f"or 'test', got {result.evaluation_split!r}"
        )

    if (
        result.learning_rate is not None
        and result.learning_rate <= 0
    ):
        errors.append(
            "learning_rate must be positive when set"
        )

    if result.status not in (
        "completed",
        "oom",
        "failed",
    ):
        errors.append(
            "status must be 'completed', 'oom', "
            f"or 'failed', got {result.status!r}"
        )

    if result.status in ("failed", "oom"):
        if not result.error_type:
            errors.append(
                "error_type is required for "
                "failed/oom runs"
            )

        if not result.error_message:
            errors.append(
                "error_message is required for "
                "failed/oom runs"
            )

    if result.status == "completed":
        if result.error_type is not None:
            errors.append(
                "completed runs must not have "
                "error_type"
            )

        if result.error_message is not None:
            errors.append(
                "completed runs must not have "
                "error_message"
            )

        if (
            result.optimizer_steps is None
            or result.optimizer_steps <= 0
        ):
            errors.append(
                "completed runs must have positive "
                "optimizer_steps"
            )

        if (
            result.trained_non_padding_tokens is None
            or result.trained_non_padding_tokens <= 0
        ):
            errors.append(
                "completed runs must have positive "
                "trained_non_padding_tokens"
            )

        if (
            result.exact_match is None
            or not 0.0
            <= result.exact_match
            <= 1.0
        ):
            errors.append(
                "completed exact_match must be "
                "within [0, 1]"
            )

        if (
            result.exact_match_first_turn is None
            or not 0.0
            <= result.exact_match_first_turn
            <= 1.0
        ):
            errors.append(
                "completed exact_match_first_turn must "
                "be within [0, 1]"
            )

        if (
            result.tokens_per_second is None
            or result.tokens_per_second <= 0
        ):
            errors.append(
                "completed runs must have positive "
                "tokens_per_second"
            )

        if (
            result.training_time_seconds is None
            or result.training_time_seconds <= 0
        ):
            errors.append(
                "completed runs must have positive "
                "training_time_seconds"
            )

    # ------------------------------------------------------------------
    # Finite numeric values
    # ------------------------------------------------------------------
    numeric_fields = (
        ("peak_memory_gb", result.peak_memory_gb),
        (
            "peak_reserved_memory_gb",
            result.peak_reserved_memory_gb,
        ),
        (
            "tokens_per_second",
            result.tokens_per_second,
        ),
        (
            "training_time_seconds",
            result.training_time_seconds,
        ),
        ("exact_match", result.exact_match),
        (
            "checkpoint_size_mb",
            result.checkpoint_size_mb,
        ),
        ("learning_rate", result.learning_rate),
    )

    for name, value in numeric_fields:
        if (
            value is not None
            and (
                math.isnan(value)
                or math.isinf(value)
            )
        ):
            errors.append(
                f"{name} must not be NaN or "
                f"infinity, got {value}"
            )

    # ------------------------------------------------------------------
    # Parameter count contract
    # ------------------------------------------------------------------
    if (
        result.trainable_parameters is not None
        and result.trainable_parameters <= 0
    ):
        errors.append(
            "trainable_parameters must be "
            "positive when set"
        )

    if (
        result.total_parameters is not None
        and result.total_parameters <= 0
    ):
        errors.append(
            "total_parameters must be positive "
            "when set"
        )

    if (
        result.trainable_parameters is not None
        and result.total_parameters is not None
        and result.trainable_parameters
        > result.total_parameters
    ):
        errors.append(
            "total_parameters must be greater "
            "than or equal to "
            "trainable_parameters"
        )

    # ------------------------------------------------------------------
    # Memory metric relationship
    # ------------------------------------------------------------------
    if (
        result.peak_memory_gb is not None
        and result.peak_memory_gb < 0
    ):
        errors.append(
            "peak_memory_gb must be non-negative"
        )

    if (
        result.peak_reserved_memory_gb is not None
        and result.peak_reserved_memory_gb < 0
    ):
        errors.append(
            "peak_reserved_memory_gb must be "
            "non-negative"
        )

    if (
        result.peak_memory_gb is not None
        and result.peak_reserved_memory_gb
        is not None
        and result.peak_reserved_memory_gb
        < result.peak_memory_gb
    ):
        errors.append(
            "peak_reserved_memory_gb must be "
            "greater than or equal to "
            "peak_memory_gb"
        )

    if errors:
        raise ValueError(
            "ExperimentResult validation failed:\n"
            + "\n".join(
                f" - {error}"
                for error in errors
            )
        )


def write_json_atomic(
    data: dict[str, Any],
    path: Path,
) -> None:
    """
    Write JSON data using a durable temporary-file
    replacement protocol.

    The temporary file is created in the destination
    directory so os.replace() remains atomic on the
    target filesystem.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_path_string = tempfile.mkstemp(
        dir=path.parent,
        suffix=".json.tmp",
        prefix=".result_",
    )
    tmp_path = Path(tmp_path_string)

    try:
        with open(
            fd,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                data,
                handle,
                indent=2,
                default=str,
            )
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            tmp_path,
            path,
        )

    except Exception:
        try:
            tmp_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise
