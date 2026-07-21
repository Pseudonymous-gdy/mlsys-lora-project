"""
Unit tests for src/training/results.py

Tests:
- completed result validation
- OOM result validation
- NaN rejection
- JSON serialization
- atomic writing
"""

import json
import tempfile
from pathlib import Path

import pytest

from training.results import (
    ExperimentResult,
    experiment_result_to_dict,
    validate_experiment_result,
    write_json_atomic,
)

# ============================================================================
# Helper functions
# ============================================================================


def make_completed_result(overrides: dict | None = None) -> ExperimentResult:
    """Create a valid completed result for testing."""
    result = ExperimentResult(
        run_id="test_lora_r16_l512_mb1_seed42",
        experiment_name="test",
        method="lora",
        rank=16,
        max_length=512,
        micro_batch_size=1,
        effective_batch_size=16,
        gradient_accumulation_steps=16,
        peak_memory_gb=2.5,
        peak_reserved_memory_gb=3.0,
        tokens_per_second=1000.0,
        training_time_seconds=60.0,
        exact_match=0.5,
        trainable_parameters=1000000,
        total_parameters=500000000,
        checkpoint_size_mb=150.0,
        trained_non_padding_tokens=100000,
        optimizer_steps=100,
        seed=42,
        sweep="test",
        model_name="test/model",
        model_revision="abc123",
        dataset_revision="def456",
        attention_backend=None,
        status="completed",
        error_type=None,
        error_message=None,
    )

    if overrides:
        for key, value in overrides.items():
            result = ExperimentResult(**{**result.__dict__, key: value})

    return result


def make_oom_result(overrides: dict | None = None) -> ExperimentResult:
    """Create a valid OOM result for testing."""
    result = ExperimentResult(
        run_id="test_lora_r16_l512_mb1_seed42",
        experiment_name="test",
        method="lora",
        rank=16,
        max_length=512,
        micro_batch_size=1,
        effective_batch_size=16,
        gradient_accumulation_steps=16,
        peak_memory_gb=None,
        peak_reserved_memory_gb=None,
        tokens_per_second=None,
        training_time_seconds=None,
        exact_match=None,
        trainable_parameters=None,
        total_parameters=None,
        checkpoint_size_mb=None,
        trained_non_padding_tokens=None,
        optimizer_steps=None,
        seed=42,
        sweep="test",
        model_name="test/model",
        model_revision="abc123",
        dataset_revision="def456",
        attention_backend=None,
        status="oom",
        error_type="CUDAOutOfMemory",
        error_message="CUDA out of memory",
    )

    if overrides:
        for key, value in overrides.items():
            result = ExperimentResult(**{**result.__dict__, key: value})

    return result


def make_failed_result(overrides: dict | None = None) -> ExperimentResult:
    """Create a valid failed result for testing."""
    result = ExperimentResult(
        run_id="test_lora_r16_l512_mb1_seed42",
        experiment_name="test",
        method="lora",
        rank=16,
        max_length=512,
        micro_batch_size=1,
        effective_batch_size=16,
        gradient_accumulation_steps=16,
        peak_memory_gb=None,
        peak_reserved_memory_gb=None,
        tokens_per_second=None,
        training_time_seconds=None,
        exact_match=None,
        trainable_parameters=None,
        total_parameters=None,
        checkpoint_size_mb=None,
        trained_non_padding_tokens=None,
        optimizer_steps=None,
        seed=42,
        sweep="test",
        model_name="test/model",
        model_revision="abc123",
        dataset_revision="def456",
        attention_backend=None,
        status="failed",
        error_type="ValueError",
        error_message="Invalid configuration",
    )

    if overrides:
        for key, value in overrides.items():
            result = ExperimentResult(**{**result.__dict__, key: value})

    return result


# ============================================================================
# Completed result validation tests
# ============================================================================


class TestCompletedResultValidation:
    def test_valid_completed_result(self):
        result = make_completed_result()
        validate_experiment_result(result)  # Should not raise

    def test_missing_run_id(self):
        result = make_completed_result(overrides={"run_id": ""})
        with pytest.raises(ValueError, match="run_id"):
            validate_experiment_result(result)

    def test_missing_experiment_name(self):
        result = make_completed_result(overrides={"experiment_name": ""})
        with pytest.raises(ValueError, match="experiment_name"):
            validate_experiment_result(result)

    def test_missing_method(self):
        result = make_completed_result(overrides={"method": ""})
        with pytest.raises(ValueError, match="method"):
            validate_experiment_result(result)

    def test_missing_model_name(self):
        result = make_completed_result(overrides={"model_name": ""})
        with pytest.raises(ValueError, match="model_name"):
            validate_experiment_result(result)

    def test_missing_model_revision(self):
        result = make_completed_result(overrides={"model_revision": ""})
        with pytest.raises(ValueError, match="model_revision"):
            validate_experiment_result(result)

    def test_missing_dataset_revision(self):
        result = make_completed_result(overrides={"dataset_revision": ""})
        with pytest.raises(ValueError, match="dataset_revision"):
            validate_experiment_result(result)

    def test_positive_optimizer_steps_required(self):
        result = make_completed_result(overrides={"optimizer_steps": None})
        with pytest.raises(ValueError, match="optimizer_steps"):
            validate_experiment_result(result)

    def test_positive_trained_tokens_required(self):
        result = make_completed_result(overrides={"trained_non_padding_tokens": None})
        with pytest.raises(ValueError, match="trained_non_padding_tokens"):
            validate_experiment_result(result)


# ============================================================================
# OOM result validation tests
# ============================================================================


class TestOomResultValidation:
    def test_valid_oom_result(self):
        result = make_oom_result()
        validate_experiment_result(result)  # Should not raise

    def test_oom_requires_error_type(self):
        result = make_oom_result(overrides={"error_type": None})
        with pytest.raises(ValueError, match="error_type"):
            validate_experiment_result(result)

    def test_oom_requires_error_message(self):
        result = make_oom_result(overrides={"error_message": None})
        with pytest.raises(ValueError, match="error_message"):
            validate_experiment_result(result)

    def test_oom_allows_none_metrics(self):
        result = make_oom_result()
        assert result.peak_memory_gb is None
        assert result.tokens_per_second is None
        assert result.exact_match is None
        validate_experiment_result(result)  # Should not raise


# ============================================================================
# Failed result validation tests
# ============================================================================


class TestFailedResultValidation:
    def test_valid_failed_result(self):
        result = make_failed_result()
        validate_experiment_result(result)  # Should not raise

    def test_failed_requires_error_type(self):
        result = make_failed_result(overrides={"error_type": None})
        with pytest.raises(ValueError, match="error_type"):
            validate_experiment_result(result)

    def test_failed_requires_error_message(self):
        result = make_failed_result(overrides={"error_message": None})
        with pytest.raises(ValueError, match="error_message"):
            validate_experiment_result(result)


# ============================================================================
# NaN/Infinity rejection tests
# ============================================================================


class TestNaNRejection:
    def test_rejects_nan_peak_memory(self):
        result = make_completed_result(overrides={"peak_memory_gb": float("nan")})
        with pytest.raises(ValueError, match="peak_memory_gb"):
            validate_experiment_result(result)

    def test_rejects_inf_tokens_per_second(self):
        result = make_completed_result(overrides={"tokens_per_second": float("inf")})
        with pytest.raises(ValueError, match="tokens_per_second"):
            validate_experiment_result(result)

    def test_rejects_nan_exact_match(self):
        result = make_completed_result(overrides={"exact_match": float("nan")})
        with pytest.raises(ValueError, match="exact_match"):
            validate_experiment_result(result)

    def test_rejects_inf_training_time(self):
        result = make_completed_result(overrides={"training_time_seconds": float("inf")})
        with pytest.raises(ValueError, match="training_time_seconds"):
            validate_experiment_result(result)

    def test_allows_none_values(self):
        """None values should be allowed for optional fields."""
        result = make_completed_result(overrides={"peak_memory_gb": None})
        validate_experiment_result(result)  # Should not raise


# ============================================================================
# Placeholder rejection tests
# ============================================================================


class TestPlaceholderRejection:
    def test_rejects_zero_trainable_parameters(self):
        result = make_completed_result(overrides={"trainable_parameters": 0})
        with pytest.raises(ValueError, match="trainable_parameters"):
            validate_experiment_result(result)

    def test_rejects_negative_trainable_parameters(self):
        result = make_completed_result(overrides={"trainable_parameters": -1})
        with pytest.raises(ValueError, match="trainable_parameters"):
            validate_experiment_result(result)

    def test_rejects_zero_total_parameters(self):
        result = make_completed_result(overrides={"total_parameters": 0})
        with pytest.raises(ValueError, match="total_parameters"):
            validate_experiment_result(result)


# ============================================================================
# JSON serialization tests
# ============================================================================


class TestSerialization:
    def test_experiment_result_to_dict(self):
        result = make_completed_result()
        data = experiment_result_to_dict(result)

        assert isinstance(data, dict)
        assert data["run_id"] == "test_lora_r16_l512_mb1_seed42"
        assert data["status"] == "completed"
        assert data["method"] == "lora"

    def test_serialization_produces_valid_json(self):
        result = make_completed_result()
        data = experiment_result_to_dict(result)

        json_str = json.dumps(data)
        loaded = json.loads(json_str)

        assert loaded["run_id"] == result.run_id
        assert loaded["status"] == result.status

    def test_none_values_preserved_in_serialization(self):
        result = make_oom_result()
        data = experiment_result_to_dict(result)

        assert data["peak_memory_gb"] is None
        assert data["tokens_per_second"] is None

    def test_stop_reason_serialization(self):
        result = make_completed_result()
        data = experiment_result_to_dict(result)

        assert isinstance(data["status"], str)


# ============================================================================
# Atomic writing tests
# ============================================================================


class TestAtomicWriting:
    def test_write_json_atomic_creates_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        path = Path(tmp.name)
        tmp.close()

        data = {"test": "value", "number": 42}
        write_json_atomic(data, path)

        assert path.exists()
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded["test"] == "value"
        assert loaded["number"] == 42

        path.unlink()

    def test_write_json_atomic_creates_parent_dirs(self):
        tmp_dir = tempfile.mkdtemp()
        path = Path(tmp_dir) / "subdir" / "result.json"

        data = {"test": "value"}
        write_json_atomic(data, path)

        assert path.exists()
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded["test"] == "value"

        path.unlink()
        path.parent.rmdir()
        path.parent.parent.rmdir()

    def test_write_json_atomic_produces_valid_json(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        path = Path(tmp.name)
        tmp.close()

        data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
        }
        write_json_atomic(data, path)

        with open(path, "r") as f:
            loaded = json.load(f)

        assert loaded == data

        path.unlink()
