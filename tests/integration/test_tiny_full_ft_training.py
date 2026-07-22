"""
Integration test: Tiny full fine-tuning training.

Validates:
- Full FT setup (all parameters trainable)
- 2 optimizer steps execute
- Parameters change after training
"""

import torch
from torch.utils.data import DataLoader, Dataset

from data.gsm8k import CausalLMCollator, GSM8KDataConfig
from methods.full_ft import configure_full_finetuning
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
# Tiny model
# ============================================================================


class TinyTransformerLikeModel(torch.nn.Module):
    """A tiny model resembling a transformer with multiple layers."""

    def __init__(self, vocab_size=100, hidden_size=32, num_layers=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(num_layers)
        ])
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
        )
        return type("Output", (), {"loss": loss, "logits": logits})()


# ============================================================================
# Helper config
# ============================================================================


def make_full_ft_config(max_steps=2) -> ExperimentConfig:
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


class TestTinyFullFTTraining:
    def test_all_parameters_trainable_after_configure(self):
        """Full FT should make all parameters trainable."""
        model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
        config = make_full_ft_config()

        configure_full_finetuning(model)

        for param in model.parameters():
            assert param.requires_grad, "All parameters should be trainable in full FT"

    def test_optimizer_contains_all_parameters(self):
        """Optimizer should contain all model parameters."""
        model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
        config = make_full_ft_config()

        configure_full_finetuning(model)
        optimizer = build_optimizer(model, config)

        param_groups = optimizer.param_groups
        total_optimized_params = sum(len(pg["params"]) for pg in param_groups)
        total_model_params = len(list(model.parameters()))

        assert total_optimized_params == total_model_params

    def test_two_optimizer_steps_execute(self):
        """Two optimizer steps should execute without error."""
        model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
        config = make_full_ft_config(max_steps=2)

        configure_full_finetuning(model)
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

    def test_parameters_change_after_training(self):
        """Parameters should change after training steps."""
        model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
        config = make_full_ft_config(max_steps=2)

        configure_full_finetuning(model)
        optimizer = build_optimizer(model, config)

        initial_params = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
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
            if not torch.equal(
                parameter.detach(),
                initial_params[name],
            )
        ]

        assert changed, "At least some parameters should have changed"

    def test_loss_is_finite_during_training(self):
        """Loss values should be finite during training."""
        model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
        config = make_full_ft_config(max_steps=4)

        configure_full_finetuning(model)
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
