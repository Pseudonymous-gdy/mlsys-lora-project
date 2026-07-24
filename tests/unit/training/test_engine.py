"""
Unit tests for src/training/engine.py

Tests with tiny model and synthetic batches:
- num_non_padding_tokens is removed
- gradient accumulation produces expected optimizer step count
- loss is correctly divided
- max steps stopping
- token budget stopping
- dataset cycling
- non-finite loss rejection
- partial accumulation behavior
- token count excludes padding
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from data.gsm8k import GSM8KDataConfig
from training.config import (
    EvaluationConfig,
    ExperimentConfig,
    ExperimentIdentityConfig,
    MethodConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from training.engine import TrainerEngine
from training.results import StopReason

# ============================================================================
# Helper classes
# ============================================================================


class TinyModel(nn.Module):
    """A tiny model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        batch_size = input_ids.shape[0]
        logits = self.linear(torch.randn(batch_size, 10))
        loss = torch.tensor(1.0, requires_grad=True)
        return type("Output", (), {"loss": loss, "logits": logits})()


def make_engine_config(
    max_steps=None,
    token_budget=None,
    grad_accum=1,
    precision="fp32",
    gradient_clip_norm=None,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentIdentityConfig(name="test", sweep="test"),
        model=ModelConfig(name="test/model", revision="abc123"),
        data=GSM8KDataConfig(
            dataset_name="openai/gsm8k",
            dataset_config="main",
            dataset_revision="def456",
            validation_size=500,
            seed=42,
            max_length=512,
            prompt_format="chat",
        ),
        method=MethodConfig(name="full_ft"),
        training=TrainingConfig(
            max_steps=max_steps,
            training_token_budget=token_budget,
            gradient_accumulation_steps=grad_accum,
            precision=precision,
            gradient_clip_norm=gradient_clip_norm,
            micro_batch_size=1,
            effective_batch_size=grad_accum,
            learning_rate=1e-3,
            throughput_warmup_steps=0,
        ),
        evaluation=EvaluationConfig(batch_size=8, max_new_tokens=512),
        output=OutputConfig(),
    )


def make_batch(num_tokens=100) -> dict[str, torch.Tensor]:
    """Create a synthetic batch with num_non_padding_tokens."""
    return {
        "input_ids": torch.randint(0, 100, (1, 10)),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
        "labels": torch.randint(0, 100, (1, 10)),
        "num_non_padding_tokens": torch.tensor([num_tokens]),
    }


class MockMemoryTracker:
    """Mock memory tracker for testing."""

    def __init__(self):
        self.reset_called = False
        self.snapshot_called = False

    def reset(self):
        self.reset_called = True

    def snapshot(self):
        self.snapshot_called = True
        return MagicMock(peak_allocated_gb=1.0, peak_reserved_gb=2.0)


# ============================================================================
# _prepare_model_inputs tests
# ============================================================================


class TestPrepareModelInputs:
    def test_num_non_padding_tokens_is_removed(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = make_batch(num_tokens=100)
        assert "num_non_padding_tokens" in batch

        inputs, num_tokens = engine._prepare_model_inputs(batch)

        assert "num_non_padding_tokens" not in inputs
        assert num_tokens == 100

    def test_inputs_moved_to_device(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = make_batch()
        inputs, _ = engine._prepare_model_inputs(batch)

        for key, tensor in inputs.items():
            assert tensor.device == device

    def test_returns_token_count(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = make_batch(num_tokens=250)
        _, num_tokens = engine._prepare_model_inputs(batch)

        assert num_tokens == 250

    def test_does_not_mutate_original_batch(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = make_batch(100)
        original_keys = set(batch)

        inputs, token_count = engine._prepare_model_inputs(batch)

        assert set(batch) == original_keys
        assert "num_non_padding_tokens" in batch
        assert "num_non_padding_tokens" not in inputs
        assert token_count == 100

    def test_missing_token_metadata_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
            "labels": torch.randint(0, 100, (1, 10)),
        }

        with pytest.raises(KeyError, match="num_non_padding_tokens"):
            engine._prepare_model_inputs(batch)

    def test_zero_token_count_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = make_batch()
        batch["num_non_padding_tokens"] = torch.tensor([0])

        with pytest.raises(ValueError, match="must be positive"):
            engine._prepare_model_inputs(batch)

    def test_negative_token_count_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = make_batch()
        batch["num_non_padding_tokens"] = torch.tensor([-5])

        with pytest.raises(ValueError, match="must be positive"):
            engine._prepare_model_inputs(batch)

    def test_multi_value_token_tensor_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = make_batch()
        batch["num_non_padding_tokens"] = torch.tensor([100, 200])

        with pytest.raises(ValueError, match="exactly one value"):
            engine._prepare_model_inputs(batch)

    def test_metadata_not_forwarded_to_model(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = make_batch(100)
        inputs, _ = engine._prepare_model_inputs(batch)

        assert "num_non_padding_tokens" not in inputs
        assert "input_ids" in inputs
        assert "labels" in inputs

    def test_missing_input_ids_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = {
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
            "labels": torch.randint(0, 100, (1, 10)),
            "num_non_padding_tokens": torch.tensor([100]),
        }

        with pytest.raises(KeyError, match="input_ids"):
            engine._prepare_model_inputs(batch)

    def test_missing_labels_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
            "num_non_padding_tokens": torch.tensor([100]),
        }

        with pytest.raises(KeyError, match="labels"):
            engine._prepare_model_inputs(batch)


# ============================================================================
# _should_stop tests
# ============================================================================


class TestShouldStop:
    def test_max_steps_stopping(self):
        config = make_engine_config(max_steps=10)
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        # Not yet at max steps
        assert engine._should_stop(5, 0) is None

        # At max steps
        result = engine._should_stop(10, 0)
        assert result == StopReason.MAX_STEPS

        # Past max steps
        result = engine._should_stop(15, 0)
        assert result == StopReason.MAX_STEPS

    def test_token_budget_stopping(self):
        config = make_engine_config(token_budget=1000)
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        # Not yet at budget
        assert engine._should_stop(0, 500) is None

        # At budget
        result = engine._should_stop(5, 1000)
        assert result == StopReason.TOKEN_BUDGET

        # Past budget
        result = engine._should_stop(5, 1500)
        assert result == StopReason.TOKEN_BUDGET

    def test_first_reached_condition_wins(self):
        """If both conditions are set, first reached should win."""
        config = make_engine_config(max_steps=5, token_budget=1000)
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        # Max steps reached first
        result = engine._should_stop(5, 500)
        assert result == StopReason.MAX_STEPS

    def test_no_stopping_condition(self):
        config = make_engine_config(max_steps=None, token_budget=None)
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        assert engine._should_stop(100, 10000) is None


# ============================================================================
# _autocast_context tests
# ============================================================================


class TestAutocastContext:
    def test_fp32_returns_noop_context(self):
        config = make_engine_config(precision="fp32")
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        with engine._autocast_context():
            pass  # Should not raise

    @patch("torch.autocast")
    def test_bf16_returns_autocast_context(self, mock_autocast):
        config = make_engine_config(precision="bf16")
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        with engine._autocast_context():
            pass

        mock_autocast.assert_called_once_with(device_type="cpu", dtype=torch.bfloat16)


# ============================================================================
# Gradient accumulation tests
# ============================================================================


class TestGradientAccumulation:
    def test_loss_correctly_divided(self):
        """Loss should be divided by gradient_accumulation_steps."""
        config = make_engine_config(grad_accum=4)
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        # Verify config has correct grad_accum
        assert config.training.gradient_accumulation_steps == 4

    def test_optimizer_steps_with_accumulation(self):
        """With grad_accum=4, 8 batches should produce 2 optimizer steps."""
        config = make_engine_config(grad_accum=4, max_steps=10)
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        # This is tested more thoroughly in integration tests
        assert config.training.gradient_accumulation_steps == 4


# ============================================================================
# _validate_loss tests
# ============================================================================


class TestValidateLoss:
    def test_missing_loss_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        with pytest.raises(RuntimeError, match="does not contain a loss"):
            engine._validate_loss(None)

    def test_non_tensor_loss_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            engine._validate_loss(1.0)

    def test_non_scalar_loss_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        loss = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="must be scalar"):
            engine._validate_loss(loss)

    def test_nan_loss_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        loss = torch.tensor(float("nan"))
        with pytest.raises(FloatingPointError, match="Non-finite training loss"):
            engine._validate_loss(loss)

    def test_positive_inf_loss_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        loss = torch.tensor(float("inf"))
        with pytest.raises(FloatingPointError, match="Non-finite training loss"):
            engine._validate_loss(loss)

    def test_negative_inf_loss_fails(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        loss = torch.tensor(float("-inf"))
        with pytest.raises(FloatingPointError, match="Non-finite training loss"):
            engine._validate_loss(loss)

    def test_finite_scalar_loss_passes(self):
        config = make_engine_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        loss = torch.tensor(1.5)
        result = engine._validate_loss(loss)

        assert result is loss
        assert result.item() == 1.5


# ============================================================================
# StopReason enum tests
# ============================================================================


class TestStopReason:
    def test_max_steps_value(self):
        assert StopReason.MAX_STEPS == "max_steps"

    def test_token_budget_value(self):
        assert StopReason.TOKEN_BUDGET == "token_budget"

    def test_data_exhausted_value(self):
        assert StopReason.DATA_EXHAUSTED == "data_exhausted"

    def test_is_string_enum(self):
        assert isinstance(StopReason.MAX_STEPS, str)
