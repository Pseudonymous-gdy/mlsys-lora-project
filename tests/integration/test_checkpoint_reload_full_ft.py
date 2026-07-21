"""
Integration test: Checkpoint reload for full fine-tuning.

Validates:
- Save checkpoint creates directory with files
- Directory has non-zero size
- Reload restores model state
- Forward pass works after reload
"""

import tempfile
from pathlib import Path

import torch

from data.gsm8k import GSM8KDataConfig
from methods.full_ft import configure_full_finetuning
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


def make_full_ft_config() -> ExperimentConfig:
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


# ============================================================================
# Tests
# ============================================================================


class TestCheckpointReloadFullFT:
    def test_save_checkpoint_creates_directory(self):
        """Saving checkpoint should create directory with files."""
        model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
        config = make_full_ft_config()

        configure_full_finetuning(model)
        optimizer = build_optimizer(model, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"

            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)

            assert checkpoint_path.exists()

    def test_checkpoint_directory_has_nonzero_size(self):
        """Checkpoint directory should have non-zero size."""
        model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
        config = make_full_ft_config()

        configure_full_finetuning(model)
        optimizer = build_optimizer(model, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)

            assert checkpoint_path.stat().st_size > 0

    def test_reload_restores_model_state(self):
        """Reloading checkpoint should restore model state."""
        model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
        config = make_full_ft_config()

        configure_full_finetuning(model)
        optimizer = build_optimizer(model, config)

        # Store original state
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)

            # Create new model and reload
            new_model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            new_model.load_state_dict(checkpoint["model_state_dict"])

            # Verify state matches
            for key in original_state:
                assert torch.equal(original_state[key], new_model.state_dict()[key])

    def test_forward_pass_works_after_reload(self):
        """Forward pass should work after reloading checkpoint."""
        model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
        config = make_full_ft_config()

        configure_full_finetuning(model)
        optimizer = build_optimizer(model, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)

            # Reload
            new_model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            new_model.load_state_dict(checkpoint["model_state_dict"])

            # Forward pass
            input_ids = torch.randint(0, 100, (2, 10))
            outputs = new_model(input_ids=input_ids)

            assert hasattr(outputs, "loss")
            assert torch.isfinite(outputs.loss)

    def test_optimizer_state_restored_after_reload(self):
        """Optimizer state should be restored after reload."""
        model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
        config = make_full_ft_config()

        configure_full_finetuning(model)
        optimizer = build_optimizer(model, config)

        # Run one optimizer step to populate state
        input_ids = torch.randint(0, 100, (2, 16))
        labels = input_ids.clone()
        output = model(input_ids=input_ids, labels=labels)
        loss = output.logits.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert len(optimizer.state) > 0, "Optimizer state should be populated after step"

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)

            # Reload
            new_model = TinyTransformerLikeModel(vocab_size=100, hidden_size=32, num_layers=2)
            configure_full_finetuning(new_model)
            new_optimizer = build_optimizer(new_model, config)

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Verify optimizer has state after reload
            assert len(new_optimizer.state) > 0, "Optimizer state should be restored after reload"
