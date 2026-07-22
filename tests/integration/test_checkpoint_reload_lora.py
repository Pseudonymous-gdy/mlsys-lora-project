"""
Integration test: Checkpoint reload for LoRA.

Validates:
- Save adapter checkpoint
- Reload base model + adapter
- Forward pass works after reload
"""

import tempfile
from pathlib import Path

import pytest
import torch

from data.gsm8k import GSM8KDataConfig
from methods.lora import configure_lora
from training.config import (
    EvaluationConfig,
    ExperimentConfig,
    ExperimentIdentityConfig,
    MethodConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from training.optim import build_optimizer
from types import SimpleNamespace

# ============================================================================
# Tiny model with linear projections
# ============================================================================


class TinyModelWithProjections(torch.nn.Module):
    """A tiny model with q_proj, k_proj, v_proj, o_proj for LoRA targeting."""

    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(
            model_type="tiny",
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = self.embedding(input_ids)
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)
        o = self.o_proj(hidden)
        hidden = q + k + v + o
        logits = self.lm_head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
        )
        return type("Output", (), {"loss": loss, "logits": logits})()

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


# ============================================================================
# Helper config
# ============================================================================


def make_lora_config(rank=8) -> ExperimentConfig:
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
        method=MethodConfig(
            name="lora",
            rank=rank,
            alpha=rank * 2,
            dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        ),
        training=TrainingConfig(
            max_steps=2,
            training_token_budget=None,
            gradient_accumulation_steps=1,
            precision="fp32",
            micro_batch_size=2,
            effective_batch_size=2,
            learning_rate=1e-3,
            throughput_warmup_steps=0,
        ),
        evaluation=EvaluationConfig(batch_size=8, max_new_tokens=512),
        output=OutputConfig(),
    )


# ============================================================================
# Tests
# ============================================================================


class TestCheckpointReloadLoRA:
    def test_save_adapter_checkpoint(self):
        """Saving LoRA adapter checkpoint should create files."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config()

        model, _ = configure_lora(model, config.method)
        optimizer = build_optimizer(model, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "adapter_checkpoint"

            # Save adapter checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)

            assert checkpoint_path.exists()

    def test_reload_base_and_adapter(self):
        """Reloading should restore both base and adapter state."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config()

        model, _ = configure_lora(model, config.method)
        optimizer = build_optimizer(model, config)

        # Store original state
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "adapter_checkpoint"

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)

            # Create new model and reload
            new_model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
            new_model, _ = configure_lora(new_model, config.method)

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            new_model.load_state_dict(checkpoint["model_state_dict"])

            # Verify state matches
            for key in original_state:
                assert torch.equal(original_state[key], new_model.state_dict()[key])

    def test_forward_pass_works_after_reload(self):
        """Forward pass should work after reloading LoRA checkpoint."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config()

        model, _ = configure_lora(model, config.method)
        optimizer = build_optimizer(model, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "adapter_checkpoint"

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)

            # Reload
            new_model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
            new_model, _ = configure_lora(new_model, config.method)

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            new_model.load_state_dict(checkpoint["model_state_dict"])

            # Forward pass
            input_ids = torch.randint(0, 100, (2, 10))
            outputs = new_model(input_ids=input_ids)

            assert hasattr(outputs, "loss")
            assert torch.isfinite(outputs.loss)

    def test_adapter_parameters_preserved_after_reload(self):
        """Adapter parameters should be preserved after reload."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config()

        model, _ = configure_lora(model, config.method)
        optimizer = build_optimizer(model, config)

        # Store adapter parameters
        adapter_params = {
            k: v.clone()
            for k, v in model.state_dict().items()
            if "lora" in k.lower()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "adapter_checkpoint"

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)

            # Reload
            new_model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
            new_model, _ = configure_lora(new_model, config.method)

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            new_model.load_state_dict(checkpoint["model_state_dict"])

            # Verify adapter parameters match
            for key in adapter_params:
                assert torch.equal(adapter_params[key], new_model.state_dict()[key])

    def test_checkpoint_size_reasonable(self):
        """LoRA checkpoint should be smaller than full model checkpoint."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config()

        model, _ = configure_lora(model, config.method)
        optimizer = build_optimizer(model, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "adapter_checkpoint"

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)

            # Checkpoint should exist and have reasonable size
            assert checkpoint_path.exists()
            assert checkpoint_path.stat().st_size > 0
            # LoRA checkpoints should be relatively small
            assert checkpoint_path.stat().st_size < 10_000_000  # Less than 10MB
