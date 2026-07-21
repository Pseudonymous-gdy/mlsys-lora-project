"""
Contract test: Result schema validation.

Validates:
- Result JSON schema compliance
- Required fields present
- Type constraints enforced
"""

import json

# import sys
import tempfile
from pathlib import Path

# from unittest.mock import MagicMock
# Mock transformers to avoid version conflict during import
# sys.modules.setdefault('transformers', MagicMock())
from training.results import (
    ExperimentResult,
    experiment_result_to_dict,
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


# ============================================================================
# Schema validation tests
# ============================================================================


class TestResultSchema:
    def test_completed_result_has_all_required_fields(self):
        """Completed result should have all required fields."""
        result = make_completed_result()
        data = experiment_result_to_dict(result)

        required_fields = [
            "run_id",
            "experiment_name",
            "method",
            "rank",
            "max_length",
            "micro_batch_size",
            "effective_batch_size",
            "gradient_accumulation_steps",
            "peak_memory_gb",
            "peak_reserved_memory_gb",
            "tokens_per_second",
            "training_time_seconds",
            "exact_match",
            "trainable_parameters",
            "total_parameters",
            "checkpoint_size_mb",
            "trained_non_padding_tokens",
            "optimizer_steps",
            "seed",
            "sweep",
            "model_name",
            "model_revision",
            "dataset_revision",
            "attention_backend",
            "status",
            "error_type",
            "error_message",
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_completed_result_field_types(self):
        """Completed result fields should have correct types."""
        result = make_completed_result()
        data = experiment_result_to_dict(result)

        assert isinstance(data["run_id"], str)
        assert isinstance(data["experiment_name"], str)
        assert isinstance(data["method"], str)
        assert isinstance(data["rank"], int)
        assert isinstance(data["max_length"], int)
        assert isinstance(data["micro_batch_size"], int)
        assert isinstance(data["effective_batch_size"], int)
        assert isinstance(data["gradient_accumulation_steps"], int)
        assert isinstance(data["peak_memory_gb"], (int, float))
        assert isinstance(data["tokens_per_second"], (int, float))
        assert isinstance(data["training_time_seconds"], (int, float))
        assert isinstance(data["exact_match"], (int, float))
        assert isinstance(data["trainable_parameters"], int)
        assert isinstance(data["total_parameters"], int)
        assert isinstance(data["checkpoint_size_mb"], (int, float))
        assert isinstance(data["trained_non_padding_tokens"], int)
        assert isinstance(data["optimizer_steps"], int)
        assert isinstance(data["seed"], int)
        assert isinstance(data["sweep"], str)
        assert isinstance(data["model_name"], str)
        assert isinstance(data["model_revision"], str)
        assert isinstance(data["dataset_revision"], str)
        assert isinstance(data["status"], str)

    def test_oom_result_schema(self):
        """OOM result should have correct schema with None metrics."""
        result = make_completed_result(overrides={
            "status": "oom",
            "peak_memory_gb": None,
            "tokens_per_second": None,
            "training_time_seconds": None,
            "exact_match": None,
            "trainable_parameters": None,
            "total_parameters": None,
            "checkpoint_size_mb": None,
            "trained_non_padding_tokens": None,
            "optimizer_steps": None,
            "error_type": "CUDAOutOfMemory",
            "error_message": "CUDA out of memory",
        })

        data = experiment_result_to_dict(result)

        assert data["status"] == "oom"
        assert data["peak_memory_gb"] is None
        assert data["error_type"] == "CUDAOutOfMemory"
        assert data["error_message"] == "CUDA out of memory"

    def test_failed_result_schema(self):
        """Failed result should have correct schema with error info."""
        result = make_completed_result(overrides={
            "status": "failed",
            "peak_memory_gb": None,
            "tokens_per_second": None,
            "training_time_seconds": None,
            "exact_match": None,
            "trainable_parameters": None,
            "total_parameters": None,
            "checkpoint_size_mb": None,
            "trained_non_padding_tokens": None,
            "optimizer_steps": None,
            "error_type": "ValueError",
            "error_message": "Invalid configuration",
        })

        data = experiment_result_to_dict(result)

        assert data["status"] == "failed"
        assert data["error_type"] == "ValueError"
        assert data["error_message"] == "Invalid configuration"

    def test_result_serializes_to_valid_json(self):
        """Result should serialize to valid JSON."""
        result = make_completed_result()
        data = experiment_result_to_dict(result)

        json_str = json.dumps(data)
        loaded = json.loads(json_str)

        assert loaded == data

    def test_result_roundtrip(self):
        """Result should survive JSON roundtrip."""
        result = make_completed_result()
        data = experiment_result_to_dict(result)

        json_str = json.dumps(data)
        loaded_data = json.loads(json_str)

        assert loaded_data["run_id"] == result.run_id
        assert loaded_data["status"] == result.status
        assert loaded_data["method"] == result.method
        assert loaded_data["exact_match"] == result.exact_match

    def test_write_json_atomic_produces_valid_file(self):
        """write_json_atomic should produce a valid JSON file."""
        result = make_completed_result()
        data = experiment_result_to_dict(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.json"
            write_json_atomic(data, output_path)

            assert output_path.exists()
            with open(output_path, "r") as f:
                loaded = json.load(f)

            assert loaded["run_id"] == result.run_id
            assert loaded["status"] == result.status

    def test_status_values_are_valid(self):
        """Status field should be one of the allowed values."""
        valid_statuses = {"completed", "oom", "failed"}

        for status in valid_statuses:
            result = make_completed_result(overrides={
                "status": status,
                "error_type": "TestError" if status != "completed" else None,
                "error_message": "Test error" if status != "completed" else None,
            })
            data = experiment_result_to_dict(result)
            assert data["status"] in valid_statuses

    def test_method_values_are_valid(self):
        """Method field should be one of the allowed values."""
        valid_methods = {"full_ft", "lora"}

        for method in valid_methods:
            result = make_completed_result(overrides={"method": method})
            data = experiment_result_to_dict(result)
            assert data["method"] in valid_methods

    def test_rank_is_positive_for_lora(self):
        """Rank should be positive for LoRA method."""
        result = make_completed_result(overrides={"method": "lora", "rank": 16})
        data = experiment_result_to_dict(result)

        assert data["rank"] > 0
        assert isinstance(data["rank"], int)

    def test_rank_none_for_full_ft(self):
        """Rank should be None for full FT method."""
        result = make_completed_result(overrides={"method": "full_ft", "rank": None})
        data = experiment_result_to_dict(result)

        assert data["rank"] is None
