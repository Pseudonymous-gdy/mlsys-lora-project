"""
Unit tests for src/training/config.py

Tests:
- valid base configuration
- invalid effective batch relationship
- missing revisions
- invalid LoRA fields
- invalid precision
- missing stopping conditions
- path serialization
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from data.gsm8k import GSM8KDataConfig
from training.config import (
    EvaluationConfig,
    ExperimentConfig,
    ExperimentIdentityConfig,
    MethodConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
    config_to_dict,
    load_experiment_config,
    validate_experiment_config,
    write_resolved_config,
)

# ============================================================================
# Helper functions
# ============================================================================


def make_minimal_config(overrides: dict | None = None) -> ExperimentConfig:
    """Create a minimal valid config for testing."""
    config = ExperimentConfig(
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
            micro_batch_size=1,
            effective_batch_size=16,
            gradient_accumulation_steps=16,
            max_steps=30,
            training_token_budget=None,
        ),
        evaluation=EvaluationConfig(batch_size=8, max_new_tokens=512),
        output=OutputConfig(),
    )

    if overrides:
        # Apply overrides by replacing fields
        for key, value in overrides.items():
            if key == "experiment":
                config = ExperimentConfig(**{**config.__dict__, "experiment": value})
            elif key == "model":
                config = ExperimentConfig(**{**config.__dict__, "model": value})
            elif key == "method":
                config = ExperimentConfig(**{**config.__dict__, "method": value})
            elif key == "training":
                config = ExperimentConfig(**{**config.__dict__, "training": value})
            elif key == "evaluation":
                config = ExperimentConfig(**{**config.__dict__, "evaluation": value})
            elif key == "data":
                config = ExperimentConfig(**{**config.__dict__, "data": value})
            elif key == "output":
                config = ExperimentConfig(**{**config.__dict__, "output": value})

    return config


def write_temp_yaml(content: dict) -> Path:
    """Write a dict to a temporary YAML file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(content, tmp)
    tmp.close()
    return Path(tmp.name)


# ============================================================================
# Valid configuration tests
# ============================================================================


class TestValidConfiguration:
    def test_valid_full_ft_config(self):
        config = make_minimal_config()
        validate_experiment_config(config)  # Should not raise

    def test_valid_lora_config(self):
        config = make_minimal_config({
            "method": MethodConfig(
                name="lora",
                rank=16,
                alpha=32,
                dropout=0.05,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            )
        })
        validate_experiment_config(config)  # Should not raise

    def test_valid_config_with_token_budget(self):
        config = make_minimal_config({
            "training": TrainingConfig(
                micro_batch_size=1,
                effective_batch_size=16,
                gradient_accumulation_steps=16,
                max_steps=None,
                training_token_budget=1_000_000,
            )
        })
        validate_experiment_config(config)  # Should not raise

    def test_valid_config_with_both_stopping_conditions(self):
        config = make_minimal_config({
            "training": TrainingConfig(
                micro_batch_size=1,
                effective_batch_size=16,
                gradient_accumulation_steps=16,
                max_steps=30,
                training_token_budget=1_000_000,
            )
        })
        validate_experiment_config(config)  # Should not raise


# ============================================================================
# Invalid configuration tests
# ============================================================================


class TestInvalidConfiguration:
    def test_empty_experiment_name(self):
        config = make_minimal_config({
            "experiment": ExperimentIdentityConfig(name="", sweep="test")
        })
        with pytest.raises(ValueError, match="experiment.name"):
            validate_experiment_config(config)

    def test_empty_experiment_sweep(self):
        config = make_minimal_config({
            "experiment": ExperimentIdentityConfig(name="test", sweep="")
        })
        with pytest.raises(ValueError, match="experiment.sweep"):
            validate_experiment_config(config)

    def test_empty_model_name(self):
        config = make_minimal_config({
            "model": ModelConfig(name="", revision="abc123")
        })
        with pytest.raises(ValueError, match="model.name"):
            validate_experiment_config(config)

    def test_empty_model_revision(self):
        config = make_minimal_config({
            "model": ModelConfig(name="test/model", revision="")
        })
        with pytest.raises(ValueError, match="model.revision"):
            validate_experiment_config(config)

    def test_invalid_effective_batch_relationship(self):
        """effective_batch_size must equal micro_batch_size × gradient_accumulation_steps."""
        config = make_minimal_config({
            "training": TrainingConfig(
                micro_batch_size=2,
                effective_batch_size=16,
                gradient_accumulation_steps=4,  # 2 × 4 = 8, not 16
            )
        })
        with pytest.raises(ValueError, match="effective_batch_size"):
            validate_experiment_config(config)

    def test_micro_batch_size_zero(self):
        config = make_minimal_config({
            "training": TrainingConfig(micro_batch_size=0, effective_batch_size=0, gradient_accumulation_steps=1)
        })
        with pytest.raises(ValueError):
            validate_experiment_config(config)

    def test_effective_batch_size_zero(self):
        config = make_minimal_config({
            "training": TrainingConfig(micro_batch_size=1, effective_batch_size=0, gradient_accumulation_steps=1)
        })
        with pytest.raises(ValueError):
            validate_experiment_config(config)

    def test_gradient_accumulation_steps_zero(self):
        config = make_minimal_config({
            "training": TrainingConfig(micro_batch_size=1, effective_batch_size=1, gradient_accumulation_steps=0)
        })
        with pytest.raises(ValueError):
            validate_experiment_config(config)

    def test_negative_learning_rate(self):
        config = make_minimal_config({
            "training": TrainingConfig(learning_rate=-0.001)
        })
        with pytest.raises(ValueError, match="learning_rate"):
            validate_experiment_config(config)

    def test_negative_weight_decay(self):
        config = make_minimal_config({
            "training": TrainingConfig(weight_decay=-0.1)
        })
        with pytest.raises(ValueError, match="weight_decay"):
            validate_experiment_config(config)

    def test_missing_stopping_conditions(self):
        """At least one of max_steps or training_token_budget must be set."""
        config = make_minimal_config({
            "training": TrainingConfig(
                max_steps=None,
                training_token_budget=None,
            )
        })
        with pytest.raises(ValueError, match="max_steps or training_token_budget"):
            validate_experiment_config(config)

    def test_invalid_precision(self):
        config = make_minimal_config({
            "training": TrainingConfig(precision="fp16")
        })
        with pytest.raises(ValueError, match="precision"):
            validate_experiment_config(config)

    def test_lora_missing_rank(self):
        config = make_minimal_config({
            "method": MethodConfig(name="lora", rank=None, alpha=32)
        })
        with pytest.raises(ValueError, match="rank"):
            validate_experiment_config(config)

    def test_lora_missing_alpha(self):
        config = make_minimal_config({
            "method": MethodConfig(name="lora", rank=16, alpha=None)
        })
        with pytest.raises(ValueError, match="alpha"):
            validate_experiment_config(config)

    def test_lora_zero_rank(self):
        config = make_minimal_config({
            "method": MethodConfig(name="lora", rank=0, alpha=32)
        })
        with pytest.raises(ValueError, match="rank"):
            validate_experiment_config(config)

    def test_lora_invalid_dropout(self):
        config = make_minimal_config({
            "method": MethodConfig(name="lora", rank=16, alpha=32, dropout=1.5)
        })
        with pytest.raises(ValueError, match="dropout"):
            validate_experiment_config(config)

    def test_lora_empty_target_modules(self):
        config = make_minimal_config({
            "method": MethodConfig(name="lora", rank=16, alpha=32, target_modules=())
        })
        with pytest.raises(ValueError, match="target_modules"):
            validate_experiment_config(config)

    def test_invalid_evaluation_batch_size(self):
        config = make_minimal_config({
            "evaluation": EvaluationConfig(batch_size=0, max_new_tokens=512)
        })
        with pytest.raises(ValueError, match="evaluation"):
            validate_experiment_config(config)

    def test_invalid_max_new_tokens(self):
        config = make_minimal_config({
            "evaluation": EvaluationConfig(batch_size=8, max_new_tokens=0)
        })
        with pytest.raises(ValueError, match="max_new_tokens"):
            validate_experiment_config(config)

    def test_rejects_non_positive_gradient_clip_norm(self):
        config = make_minimal_config({
            "training": TrainingConfig(gradient_clip_norm=0)
        })
        with pytest.raises(ValueError, match="gradient_clip_norm"):
            validate_experiment_config(config)

    @pytest.mark.parametrize("max_steps", [0, -1])
    def test_rejects_non_positive_max_steps(self, max_steps):
        config = make_minimal_config({
            "training": TrainingConfig(
                micro_batch_size=1,
                effective_batch_size=16,
                gradient_accumulation_steps=16,
                max_steps=max_steps,
                training_token_budget=1_000_000,
            )
        })
        with pytest.raises(ValueError, match="max_steps must be > 0"):
            validate_experiment_config(config)

    @pytest.mark.parametrize("token_budget", [0, -1])
    def test_rejects_non_positive_token_budget(self, token_budget):
        config = make_minimal_config({
            "training": TrainingConfig(
                micro_batch_size=1,
                effective_batch_size=16,
                gradient_accumulation_steps=16,
                max_steps=30,
                training_token_budget=token_budget,
            )
        })
        with pytest.raises(ValueError, match="training_token_budget must be > 0"):
            validate_experiment_config(config)

    def test_rejects_warmup_equal_to_max_steps(self):
        config = make_minimal_config({
            "training": TrainingConfig(
                micro_batch_size=1,
                effective_batch_size=16,
                gradient_accumulation_steps=16,
                max_steps=2,
                throughput_warmup_steps=2,
            )
        })
        with pytest.raises(ValueError, match="throughput_warmup_steps must be smaller"):
            validate_experiment_config(config)

    def test_rejects_warmup_greater_than_max_steps(self):
        config = make_minimal_config({
            "training": TrainingConfig(
                micro_batch_size=1,
                effective_batch_size=16,
                gradient_accumulation_steps=16,
                max_steps=2,
                throughput_warmup_steps=3,
            )
        })
        with pytest.raises(ValueError, match="throughput_warmup_steps must be smaller"):
            validate_experiment_config(config)

    def test_accepts_warmup_smaller_than_max_steps(self):
        config = make_minimal_config({
            "training": TrainingConfig(
                micro_batch_size=1,
                effective_batch_size=16,
                gradient_accumulation_steps=16,
                max_steps=2,
                throughput_warmup_steps=1,
            )
        })
        validate_experiment_config(config)  # Should not raise


# ============================================================================
# Training config structure tests
# ============================================================================


class TestTrainingConfigStructure:
    def test_training_config_exposes_gradient_clip_norm(self):
        config = make_minimal_config()
        assert hasattr(config.training, "gradient_clip_norm")
        assert not hasattr(config.training, "max_grad_norm")


# ============================================================================
# YAML loading tests
# ============================================================================


class TestYamlLoading:
    def test_load_valid_yaml(self):
        content = {
            "experiment": {"name": "test", "sweep": "test"},
            "model": {"name": "test/model", "revision": "abc123"},
            "data": {"revision": "def456"},
            "method": {"name": "full_ft"},
            "training": {
                "micro_batch_size": 1,
                "effective_batch_size": 16,
                "gradient_accumulation_steps": 16,
                "max_steps": 30,
            },
            "evaluation": {"batch_size": 8, "max_new_tokens": 512},
        }
        path = write_temp_yaml(content)
        config = load_experiment_config(path)

        assert config.experiment.name == "test"
        assert config.method.name == "full_ft"
        path.unlink()

    def test_load_empty_yaml(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        tmp.write("")
        tmp.close()
        path = Path(tmp.name)

        with pytest.raises(ValueError, match="empty"):
            load_experiment_config(path)
        path.unlink()

    def test_load_non_mapping_yaml(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        tmp.write("- item1\n- item2\n")
        tmp.close()
        path = Path(tmp.name)

        with pytest.raises(ValueError, match="mapping"):
            load_experiment_config(path)
        path.unlink()

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_experiment_config(Path("/nonexistent/config.yaml"))


# ============================================================================
# Serialization tests
# ============================================================================


class TestSerialization:
    def test_config_to_dict(self):
        config = make_minimal_config()
        result = config_to_dict(config)

        assert isinstance(result, dict)
        assert result["experiment"]["name"] == "test"
        assert result["method"]["name"] == "full_ft"

    def test_config_to_dict_preserves_paths(self):
        config = make_minimal_config()
        result = config_to_dict(config)

        assert "results_dir" in result["output"]
        assert "checkpoints_dir" in result["output"]

    def test_write_resolved_config(self):
        config = make_minimal_config()
        tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        path = Path(tmp.name)
        tmp.close()

        write_resolved_config(config, path)

        assert path.exists()
        with open(path, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded["experiment"]["name"] == "test"

        path.unlink()
