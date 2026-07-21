"""
Unit tests for src/training/optim.py

Tests:
- only trainable parameters enter optimizer
- empty parameter set fails
- configured learning rate and weight decay are applied
"""

import pytest
import torch
import torch.nn as nn

from training.optim import build_optimizer
from training.config import (
    ExperimentConfig,
    ExperimentIdentityConfig,
    ModelConfig,
    MethodConfig,
    TrainingConfig,
    EvaluationConfig,
    OutputConfig,
)
from data.gsm8k import GSM8KDataConfig


# ============================================================================
# Helper models
# ============================================================================


class SimpleModel(nn.Module):
    """A simple model with configurable trainable/frozen parameters."""

    def __init__(self, freeze_second=False):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)  # 55 params
        self.linear2 = nn.Linear(5, 2)   # 12 params

        if freeze_second:
            for param in self.linear2.parameters():
                param.requires_grad = False


# ============================================================================
# Helper config
# ============================================================================


def make_config(lr=1e-3, weight_decay=0.01) -> ExperimentConfig:
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
            learning_rate=lr,
            weight_decay=weight_decay,
            micro_batch_size=1,
            effective_batch_size=16,
            gradient_accumulation_steps=16,
            max_steps=30,
        ),
        evaluation=EvaluationConfig(batch_size=8, max_new_tokens=512),
        output=OutputConfig(),
    )


# ============================================================================
# Tests
# ============================================================================


class TestBuildOptimizer:
    def test_only_trainable_parameters_enter_optimizer(self):
        model = SimpleModel(freeze_second=True)
        config = make_config()

        optimizer = build_optimizer(model, config)

        # Count parameters in optimizer
        optimizer_param_count = sum(p.numel() for group in optimizer.param_groups for p in group["params"])

        # Should only include linear1 (55 params), not linear2 (12 params)
        assert optimizer_param_count == 55

    def test_empty_parameter_set_fails(self):
        model = SimpleModel(freeze_second=False)
        for param in model.parameters():
            param.requires_grad = False

        config = make_config()

        with pytest.raises(ValueError, match="no trainable parameters"):
            build_optimizer(model, config)

    def test_configured_learning_rate_applied(self):
        model = SimpleModel()
        config = make_config(lr=5e-4)

        optimizer = build_optimizer(model, config)

        assert optimizer.param_groups[0]["lr"] == 5e-4

    def test_configured_weight_decay_applied(self):
        model = SimpleModel()
        config = make_config(weight_decay=0.1)

        optimizer = build_optimizer(model, config)

        assert optimizer.param_groups[0]["weight_decay"] == 0.1

    def test_optimizer_is_adamw(self):
        model = SimpleModel()
        config = make_config()

        optimizer = build_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_all_trainable_parameters_included(self):
        model = SimpleModel(freeze_second=False)
        config = make_config()

        optimizer = build_optimizer(model, config)

        # Count parameters in optimizer
        optimizer_param_count = sum(p.numel() for group in optimizer.param_groups for p in group["params"])

        # Should include all params (55 + 12 = 67)
        assert optimizer_param_count == 67

    def test_frozen_parameters_excluded(self):
        model = SimpleModel(freeze_second=True)
        config = make_config()

        optimizer = build_optimizer(model, config)

        # Check that linear2 params are not in optimizer
        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        linear2_param_ids = {id(p) for p in model.linear2.parameters()}

        assert len(optimizer_param_ids & linear2_param_ids) == 0
