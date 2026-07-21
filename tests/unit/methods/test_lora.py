"""
Unit tests for src/methods/lora.py

Tests with tiny compatible models:
- valid targets are resolved
- missing targets fail
- LoRA parameters become trainable
- base parameters remain frozen
- trainable count is lower than Full FT
"""

import pytest
import torch.nn as nn

from methods.lora import (
    LoraMethodConfig,
    resolve_lora_target_modules,
    validate_lora_trainability,
)

# ============================================================================
# Helper models
# ============================================================================


class TinyTransformerLikeModel(nn.Module):
    """A tiny model mimicking transformer module structure.

    The model also provides the minimal generation preparation API required
    by PEFT when LoRA is configured with task_type="CAUSAL_LM".
    """

    def __init__(self):
        super().__init__()
        # Simulate attention layers
        self.q_proj = nn.Linear(10, 5)
        self.k_proj = nn.Linear(10, 5)
        self.v_proj = nn.Linear(10, 5)
        self.o_proj = nn.Linear(5, 10)

        # Simulate MLP layers
        self.gate_proj = nn.Linear(10, 20)
        self.up_proj = nn.Linear(10, 20)
        self.down_proj = nn.Linear(20, 10)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Return minimal generation inputs expected by PEFT CAUSAL_LM.

        The tests in this file do not perform text generation. This method
        exists only to satisfy the causal-language-model interface expected
        by PeftModelForCausalLM during LoRA wrapping.
        """
        model_inputs = {"input_ids": input_ids}

        if "attention_mask" in kwargs:
            model_inputs["attention_mask"] = kwargs["attention_mask"]

        return model_inputs


class ModelWithoutTargetModules(nn.Module):
    """A model without the expected target module suffixes."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)


# ============================================================================
# LoraMethodConfig tests
# ============================================================================


class TestLoraMethodConfig:
    def test_create_valid_config(self):
        config = LoraMethodConfig(
            rank=16,
            alpha=32,
            dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            bias="none",
        )

        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.05
        assert config.bias == "none"

    def test_target_modules_is_tuple(self):
        config = LoraMethodConfig(
            rank=8,
            alpha=16,
            dropout=0.0,
            target_modules=("q_proj",),
            bias="none",
        )

        assert isinstance(config.target_modules, tuple)


# ============================================================================
# resolve_lora_target_modules tests
# ============================================================================


class TestResolveLoraTargetModules:
    def test_valid_targets_resolved(self):
        model = TinyTransformerLikeModel()
        requested = ("q_proj", "k_proj", "v_proj", "o_proj")

        validated = resolve_lora_target_modules(model, requested)

        assert validated == requested
        assert isinstance(validated, tuple)

    def test_rejects_empty_config(self):
        model = TinyTransformerLikeModel()

        with pytest.raises(
            ValueError,
            match="target_modules configuration is empty",
        ):
            resolve_lora_target_modules(model, ())

        with pytest.raises(
            ValueError,
            match="target_modules configuration is empty",
        ):
            resolve_lora_target_modules(model, tuple())

    def test_missing_target_fails(self):
        model = ModelWithoutTargetModules()
        requested = ("q_proj", "k_proj")

        with pytest.raises(ValueError, match="not found in model"):
            resolve_lora_target_modules(model, requested)

    def test_includes_available_module_names_in_error(self):
        model = ModelWithoutTargetModules()
        requested = ("q_proj",)

        with pytest.raises(ValueError) as exc_info:
            resolve_lora_target_modules(model, requested)

        error_msg = str(exc_info.value)
        assert "layer1" in error_msg or "layer2" in error_msg

    def test_partial_match_fails(self):
        """If any requested module is missing, should fail."""
        model = TinyTransformerLikeModel()
        requested = ("q_proj", "nonexistent_proj")

        with pytest.raises(ValueError, match="not found in model"):
            resolve_lora_target_modules(model, requested)

    def test_suffix_matching(self):
        """Target modules should match as suffixes."""

        class ModelWithNestedModules(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Module()
                self.layer1.q_proj = nn.Linear(10, 5)
                self.layer2 = nn.Module()
                self.layer2.k_proj = nn.Linear(10, 5)

        model = ModelWithNestedModules()
        requested = ("q_proj", "k_proj")

        validated = resolve_lora_target_modules(model, requested)

        assert validated == requested


# ============================================================================
# validate_lora_trainability tests
# ============================================================================


class TestValidateLoraTrainability:
    def test_passes_with_lora_model(self):
        """This test requires PEFT, skip if not installed."""
        pytest.importorskip("peft")

        from peft import LoraConfig, get_peft_model

        model = TinyTransformerLikeModel()
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_config)

        validate_lora_trainability(peft_model)

    def test_detects_non_adapter_trainable_params(self):
        """If base model params are trainable, should fail."""
        pytest.importorskip("peft")

        from peft import LoraConfig, get_peft_model

        model = TinyTransformerLikeModel()
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_config)

        # Manually unfreeze a base-model parameter.
        peft_model.base_model.model.q_proj.weight.requires_grad = True

        with pytest.raises(
            ValueError,
            match=r"non-adapter trainable parameters",
        ):
            validate_lora_trainability(peft_model)

    def test_detects_no_trainable_params(self):
        """If no params are trainable, should fail."""
        model = TinyTransformerLikeModel()

        for param in model.parameters():
            param.requires_grad = False

        with pytest.raises(ValueError, match="no trainable parameters"):
            validate_lora_trainability(model)

    def test_trainable_count_lower_than_full_ft(self):
        """LoRA should have fewer trainable params than Full FT."""
        pytest.importorskip("peft")

        from peft import LoraConfig, get_peft_model

        from methods.common import count_parameters

        # Full FT model
        full_ft_model = TinyTransformerLikeModel()

        for param in full_ft_model.parameters():
            param.requires_grad = True

        full_ft_stats = count_parameters(full_ft_model)

        # LoRA model
        lora_model = TinyTransformerLikeModel()
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(lora_model, lora_config)
        lora_stats = count_parameters(peft_model)

        assert (
            lora_stats.trainable_parameters
            < full_ft_stats.trainable_parameters
        )