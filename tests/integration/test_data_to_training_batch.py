"""
Integration test: Data to training batch pipeline.

Validates the flow:
Dataset → CausalLMCollator → TrainerEngine → model.forward

Uses existing GSM8K collator contract and small synthetic dataset.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from data.gsm8k import CausalLMCollator, GSM8KDataConfig
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

# ============================================================================
# Synthetic dataset
# ============================================================================


class SyntheticDataset(Dataset):
    """A tiny dataset that mimics GSM8K tokenized output."""

    def __init__(self, num_samples=10, seq_length=20):
        self.num_samples = num_samples
        self.seq_length = seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate tokenized output from GSM8K pipeline (as lists, not tensors)
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


class TinyCausalLM(torch.nn.Module):
    """A tiny model that accepts the same inputs as CausalLM."""

    def __init__(self, vocab_size=100, hidden_size=16):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = self.embedding(input_ids)
        logits = self.linear(hidden)
        # Simple cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
        )
        return type("Output", (), {"loss": loss, "logits": logits})()


# ============================================================================
# Helper config
# ============================================================================


def make_config() -> ExperimentConfig:
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
            max_steps=2,
            training_token_budget=None,
            gradient_accumulation_steps=1,
            precision="fp32",
            micro_batch_size=2,
            effective_batch_size=2,
            learning_rate=1e-3,
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


class TestDataToTrainingBatch:
    def test_collator_produces_expected_keys(self):
        """CausalLMCollator should produce input_ids, attention_mask, labels, num_non_padding_tokens."""
        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)

        samples = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5],
            }
            for _ in range(4)
        ]

        batch = collator(samples)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "num_non_padding_tokens" in batch
        assert batch["input_ids"].shape[0] == 4
        assert batch["num_non_padding_tokens"] > 0

    def test_collator_pads_to_multiple(self):
        """CausalLMCollator should pad to multiple of pad_to_multiple_of."""
        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=8)

        samples = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [1, 2, 3],
            }
            for _ in range(2)
        ]

        batch = collator(samples)
        assert batch["input_ids"].shape[1] % 8 == 0

    def test_engine_receives_batch_from_collator(self):
        """TrainerEngine should correctly process batch from collator."""
        config = make_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)
        samples = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5],
            }
            for _ in range(4)
        ]
        batch = collator(samples)

        inputs, num_tokens = engine._prepare_model_inputs(batch)

        assert "num_non_padding_tokens" not in inputs
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "labels" in inputs
        assert num_tokens > 0

    def test_model_forward_with_engine_inputs(self):
        """Model should accept inputs prepared by engine."""
        model = TinyCausalLM(vocab_size=100, hidden_size=16)

        config = make_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 100, (2, 10)),
            "num_non_padding_tokens": torch.tensor(16),
        }

        inputs, num_tokens = engine._prepare_model_inputs(batch)

        outputs = model(**inputs)

        assert hasattr(outputs, "loss")
        assert outputs.loss.dim() == 0
        assert torch.isfinite(outputs.loss)

    def test_full_pipeline_collator_to_model(self):
        """Full pipeline: Dataset → DataLoader → Collator → Engine → Model."""
        dataset = SyntheticDataset(num_samples=8, seq_length=20)
        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        model = TinyCausalLM(vocab_size=100, hidden_size=16)

        config = make_config()
        device = torch.device("cpu")
        engine = TrainerEngine(config, device, MockMemoryTracker())

        batch = next(iter(loader))
        inputs, num_tokens = engine._prepare_model_inputs(batch)

        outputs = model(**inputs)

        assert torch.isfinite(outputs.loss)
        assert num_tokens > 0
