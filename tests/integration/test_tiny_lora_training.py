"""
Integration test: Tiny LoRA training.

Validates:
- LoRA setup with adapter injection
- Only adapter parameters change
- Base parameters remain frozen
"""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from data.gsm8k import CausalLMCollator, GSM8KDataConfig
from methods.lora import configure_lora
from metrics.throughput import ThroughputTracker
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
from training.optim import build_optimizer

# ============================================================================
# Synthetic dataset
# ============================================================================


class TinyTokenDataset(Dataset):
    """A tiny dataset with tokenized examples."""

    def __init__(self, num_samples=16, seq_length=32):
        self.num_samples = num_samples
        self.seq_length = seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        length = self.seq_length
        input_ids = list(range(1, length + 1))
        attention_mask = [1] * length
        labels = list(range(1, length + 1))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


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


def make_lora_config(max_steps=2, rank=8) -> ExperimentConfig:
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
            max_steps=max_steps,
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


class MockMemoryTracker:
    def reset(self):
        pass

    def snapshot(self):
        class FakeMetrics:
            peak_allocated_gb = 1.0
            peak_reserved_gb = 2.0
        return FakeMetrics()


# ============================================================================
# Tests
# ============================================================================


class TestTinyLoRATraining:
    def test_lora_adapter_parameters_trainable(self):
        """LoRA should make only adapter parameters trainable."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config()

        model, _ = configure_lora(model, config.method)

        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        frozen_params = [name for name, param in model.named_parameters() if not param.requires_grad]

        assert len(trainable_params) > 0, "Should have trainable adapter parameters"
        assert len(frozen_params) > 0, "Should have frozen base parameters"

    def test_base_parameters_frozen_after_lora(self):
        """Base model parameters should be frozen after LoRA injection."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config()

        model, _ = configure_lora(model, config.method)

        # Check that original projection layers are frozen
        for name, param in model.named_parameters():
            if "lora" not in name.lower():
                assert not param.requires_grad, f"Base parameter {name} should be frozen"

    def test_optimizer_contains_only_adapter_parameters(self):
        """Optimizer should contain only LoRA adapter parameters."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config()

        model, _ = configure_lora(model, config.method)
        optimizer = build_optimizer(model, config)

        param_groups = optimizer.param_groups
        total_optimized_params = sum(len(pg["params"]) for pg in param_groups)
        total_trainable_params = sum(1 for param in model.parameters() if param.requires_grad)

        assert total_optimized_params == total_trainable_params

    def test_two_optimizer_steps_execute(self):
        """Two optimizer steps should execute without error."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config(max_steps=2)

        model, _ = configure_lora(model, config.method)
        optimizer = build_optimizer(model, config)

        dataset = TinyTokenDataset(num_samples=8, seq_length=32)
        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())
        tracker = ThroughputTracker(
            device=device,
            warmup_optimizer_steps=0,
        )

        result = engine.train(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            throughput_tracker=tracker,
        )

        assert result.optimizer_steps == 2
        assert result.micro_steps == 2
        assert result.stop_reason == "max_steps"
        assert result.trained_non_padding_tokens > 0
        assert result.measured_time_seconds > 0
        assert result.tokens_per_second > 0

    def test_adapter_parameters_change_after_training(self):
        """Adapter parameters should change after training steps."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config(max_steps=2)

        model, _ = configure_lora(model, config.method)
        optimizer = build_optimizer(model, config)

        initial_adapter_params = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if (
                "lora" in name.lower()
                and parameter.requires_grad
            )
        }

        dataset = TinyTokenDataset(num_samples=8, seq_length=32)
        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())
        tracker = ThroughputTracker(
            device=device,
            warmup_optimizer_steps=0,
        )

        engine.train(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            throughput_tracker=tracker,
        )

        changed = [
            name
            for name, parameter in model.named_parameters()
            if (
                "lora" in name.lower()
                and parameter.requires_grad
                and not torch.equal(
                    parameter.detach(),
                    initial_adapter_params[name],
                )
            )
        ]

        assert changed, "At least some adapter parameters should have changed"

    def test_base_parameters_unchanged_after_training(self):
        """Base parameters should remain unchanged after LoRA training."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config(max_steps=2)

        model, _ = configure_lora(model, config.method)
        optimizer = build_optimizer(model, config)

        initial_base_params = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if "lora" not in name.lower()
        }

        dataset = TinyTokenDataset(num_samples=8, seq_length=32)
        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())
        tracker = ThroughputTracker(
            device=device,
            warmup_optimizer_steps=0,
        )

        engine.train(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            throughput_tracker=tracker,
        )

        for name, parameter in model.named_parameters():
            if "lora" not in name.lower():
                assert torch.equal(
                    parameter.detach(),
                    initial_base_params[name],
                ), f"Base parameter {name} should not change"

    def test_loss_is_finite_during_training(self):
        """Loss should be finite during LoRA training."""
        pytest.importorskip("peft")

        model = TinyModelWithProjections(vocab_size=100, hidden_size=32)
        config = make_lora_config(max_steps=4)

        model, _ = configure_lora(model, config.method)
        optimizer = build_optimizer(model, config)

        dataset = TinyTokenDataset(num_samples=16, seq_length=32)
        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())
        tracker = ThroughputTracker(
            device=device,
            warmup_optimizer_steps=0,
        )

        result = engine.train(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            throughput_tracker=tracker,
        )

        assert torch.isfinite(torch.tensor(result.final_loss))
        assert torch.isfinite(torch.tensor(result.mean_loss))
