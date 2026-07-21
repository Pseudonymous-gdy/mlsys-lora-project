"""
Unit tests for src/methods/common.py

Tests:
- correct total/trainable/frozen counts
- zero trainable rejection
- optimizer coverage detection
"""

import pytest
import torch
import torch.nn as nn

from methods.common import (
    ParameterStats,
    count_parameters,
    get_trainable_parameter_names,
    assert_has_trainable_parameters,
    assert_optimizer_matches_trainable_parameters,
)


# ============================================================================
# Helper models
# ============================================================================


class SimpleModel(nn.Module):
    """A simple model with configurable trainable/frozen parameters."""

    def __init__(self, freeze_second=False):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)  # 55 params
        self.linear2 = nn.Linear(5, 2)   # 12 params
        # Total: 67 params

        if freeze_second:
            for param in self.linear2.parameters():
                param.requires_grad = False


class AllFrozenModel(nn.Module):
    """A model with all parameters frozen."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # 55 params
        for param in self.parameters():
            param.requires_grad = False


# ============================================================================
# ParameterStats tests
# ============================================================================


class TestParameterStats:
    def test_valid_stats(self):
        stats = ParameterStats(
            total_parameters=100,
            trainable_parameters=60,
            frozen_parameters=40,
            trainable_fraction=0.6,
        )
        assert stats.total_parameters == 100
        assert stats.trainable_parameters == 60
        assert stats.frozen_parameters == 40
        assert stats.trainable_fraction == 0.6

    def test_invariant_total_equals_trainable_plus_frozen(self):
        with pytest.raises(ValueError, match="Parameter counts do not add up"):
            ParameterStats(
                total_parameters=100,
                trainable_parameters=60,
                frozen_parameters=50,  # 60 + 50 != 100
                trainable_fraction=0.6,
            )

    def test_trainable_fraction_bounds(self):
        with pytest.raises(ValueError, match="Trainable fraction must be between 0 and 1"):
            ParameterStats(
                total_parameters=100,
                trainable_parameters=60,
                frozen_parameters=40,
                trainable_fraction=1.5,
            )

    def test_trainable_fraction_zero(self):
        stats = ParameterStats(
            total_parameters=100,
            trainable_parameters=0,
            frozen_parameters=100,
            trainable_fraction=0.0,
        )
        assert stats.trainable_fraction == 0.0

    def test_trainable_fraction_one(self):
        stats = ParameterStats(
            total_parameters=100,
            trainable_parameters=100,
            frozen_parameters=0,
            trainable_fraction=1.0,
        )
        assert stats.trainable_fraction == 1.0


# ============================================================================
# count_parameters tests
# ============================================================================


class TestCountParameters:
    def test_all_trainable(self):
        model = SimpleModel(freeze_second=False)
        stats = count_parameters(model)

        assert stats.total_parameters == 67
        assert stats.trainable_parameters == 67
        assert stats.frozen_parameters == 0
        assert stats.trainable_fraction == 1.0

    def test_partial_frozen(self):
        model = SimpleModel(freeze_second=True)
        stats = count_parameters(model)

        assert stats.total_parameters == 67
        assert stats.trainable_parameters == 55  # linear1 only
        assert stats.frozen_parameters == 12     # linear2
        assert stats.trainable_fraction == pytest.approx(55 / 67, rel=1e-5)

    def test_all_frozen(self):
        model = AllFrozenModel()
        stats = count_parameters(model)

        assert stats.total_parameters == 55
        assert stats.trainable_parameters == 0
        assert stats.frozen_parameters == 55
        assert stats.trainable_fraction == 0.0

    def test_does_not_modify_model(self):
        model = SimpleModel(freeze_second=False)
        original_grads = [p.requires_grad for p in model.parameters()]

        count_parameters(model)

        after_grads = [p.requires_grad for p in model.parameters()]
        assert original_grads == after_grads


# ============================================================================
# get_trainable_parameter_names tests
# ============================================================================


class TestGetTrainableParameterNames:
    def test_all_trainable(self):
        model = SimpleModel(freeze_second=False)
        names = get_trainable_parameter_names(model)

        assert len(names) == 4  # weight1, bias1, weight2, bias2
        assert "linear1.weight" in names
        assert "linear1.bias" in names
        assert "linear2.weight" in names
        assert "linear2.bias" in names

    def test_partial_frozen(self):
        model = SimpleModel(freeze_second=True)
        names = get_trainable_parameter_names(model)

        assert len(names) == 2  # linear1 only
        assert "linear1.weight" in names
        assert "linear1.bias" in names
        assert "linear2.weight" not in names
        assert "linear2.bias" not in names

    def test_all_frozen(self):
        model = AllFrozenModel()
        names = get_trainable_parameter_names(model)

        assert len(names) == 0

    def test_preserves_model_traversal_order(self):
        model = SimpleModel(freeze_second=False)
        names = get_trainable_parameter_names(model)

        # Should match the order of named_parameters()
        expected = tuple(name for name, _ in model.named_parameters())
        assert names == expected


# ============================================================================
# assert_has_trainable_parameters tests
# ============================================================================


class TestAssertHasTrainableParameters:
    def test_passes_with_trainable_params(self):
        model = SimpleModel(freeze_second=False)
        assert_has_trainable_parameters(model)  # Should not raise

    def test_raises_with_all_frozen(self):
        model = AllFrozenModel()
        with pytest.raises(ValueError, match="no trainable parameters"):
            assert_has_trainable_parameters(model)


# ============================================================================
# assert_optimizer_matches_trainable_parameters tests
# ============================================================================


class TestAssertOptimizerMatchesTrainableParameters:
    def test_passes_with_correct_optimizer(self):
        model = SimpleModel(freeze_second=False)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )
        assert_optimizer_matches_trainable_parameters(model, optimizer)  # Should not raise

    def test_raises_with_missing_trainable_param(self):
        model = SimpleModel(freeze_second=False)
        # Create optimizer with only linear1 params (missing linear2)
        optimizer = torch.optim.AdamW(
            list(model.linear1.parameters()),
            lr=1e-3,
        )
        with pytest.raises(ValueError, match="trainable parameter"):
            assert_optimizer_matches_trainable_parameters(model, optimizer)

    def test_raises_with_frozen_param_added(self):
        model = SimpleModel(freeze_second=True)
        # Create optimizer with all params (including frozen linear2)
        optimizer = torch.optim.AdamW(
            list(model.parameters()),
            lr=1e-3,
        )
        with pytest.raises(ValueError, match="frozen parameter"):
            assert_optimizer_matches_trainable_parameters(model, optimizer)

    def test_passes_with_partial_frozen(self):
        model = SimpleModel(freeze_second=True)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )
        assert_optimizer_matches_trainable_parameters(model, optimizer)  # Should not raise
