"""
Unit tests for src/methods/full_ft.py

Tests:
- all parameters become trainable
- parameter counts are correct
- validation detects manually frozen parameters
"""

import pytest
import torch
import torch.nn as nn

from methods.common import ParameterStats
from methods.full_ft import configure_full_finetuning, validate_full_finetuning


# ============================================================================
# Helper models
# ============================================================================


class SimpleModel(nn.Module):
    """A simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)  # 55 params
        self.linear2 = nn.Linear(5, 2)   # 12 params
        # Total: 67 params


class PartiallyFrozenModel(nn.Module):
    """A model with some frozen parameters."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)
        # Freeze linear2
        for param in self.linear2.parameters():
            param.requires_grad = False


# ============================================================================
# configure_full_finetuning tests
# ============================================================================


class TestConfigureFullFinetuning:
    def test_all_parameters_become_trainable(self):
        model = SimpleModel()
        # Initially all trainable, but let's freeze some
        for param in model.linear2.parameters():
            param.requires_grad = False

        stats = configure_full_finetuning(model)

        # All parameters should now be trainable
        assert all(p.requires_grad for p in model.parameters())
        assert stats.trainable_parameters == 67
        assert stats.frozen_parameters == 0
        assert stats.trainable_fraction == 1.0

    def test_returns_parameter_stats(self):
        model = SimpleModel()
        stats = configure_full_finetuning(model)

        assert isinstance(stats, ParameterStats)
        assert stats.total_parameters == 67
        assert stats.trainable_parameters == 67

    def test_does_not_modify_model_architecture(self):
        model = SimpleModel()
        original_modules = list(model.modules())

        configure_full_finetuning(model)

        assert list(model.modules()) == original_modules

    def test_already_trainable_model(self):
        model = SimpleModel()
        stats = configure_full_finetuning(model)

        assert stats.total_parameters == 67
        assert stats.trainable_parameters == 67


# ============================================================================
# validate_full_finetuning tests
# ============================================================================


class TestValidateFullFinetuning:
    def test_passes_with_all_trainable(self):
        model = SimpleModel()
        for param in model.parameters():
            param.requires_grad = True

        validate_full_finetuning(model)  # Should not raise

    def test_detects_frozen_parameters(self):
        model = PartiallyFrozenModel()

        with pytest.raises(AssertionError, match="Found frozen parameter"):
            validate_full_finetuning(model)

    def test_detects_all_frozen(self):
        model = SimpleModel()
        for param in model.parameters():
            param.requires_grad = False

        with pytest.raises((AssertionError, ValueError)):
            validate_full_finetuning(model)

    def test_does_not_assume_specific_parameter_names(self):
        """Validation should work regardless of parameter naming."""

        class CustomNamedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.custom_layer = nn.Linear(5, 3)

        model = CustomNamedModel()
        validate_full_finetuning(model)  # Should not raise

    def test_raises_descriptive_error_with_frozen(self):
        model = PartiallyFrozenModel()

        with pytest.raises(AssertionError) as exc_info:
            validate_full_finetuning(model)

        error_msg = str(exc_info.value)
        assert "frozen" in error_msg.lower()
