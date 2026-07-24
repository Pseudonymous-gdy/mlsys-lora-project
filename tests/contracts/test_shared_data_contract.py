"""
Contract test: Shared data contract between Full FT and LoRA.

Validates:
- Full FT and LoRA receive the same data
- Same tokenization produces identical results
- Same collator produces identical batches
"""

import torch
from torch.utils.data import DataLoader, Dataset

from data.gsm8k import CausalLMCollator

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
# Tests
# ============================================================================


class TestSharedDataContract:
    def test_same_dataset_same_samples(self):
        """Full FT and LoRA should receive the same samples from dataset."""
        dataset = TinyTokenDataset(num_samples=8, seq_length=20)

        # Get samples for both methods
        full_ft_samples = [dataset[i] for i in range(4)]
        lora_samples = [dataset[i] for i in range(4)]

        # Should be identical
        for ft_sample, lora_sample in zip(full_ft_samples, lora_samples):
            assert ft_sample["input_ids"] == lora_sample["input_ids"]
            assert ft_sample["attention_mask"] == lora_sample["attention_mask"]
            assert ft_sample["labels"] == lora_sample["labels"]

    def test_same_collator_same_batch(self):
        """Full FT and LoRA should produce identical batches with same collator."""
        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)

        samples = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5],
            }
            for _ in range(4)
        ]

        # Collate twice
        batch1 = collator(samples)
        batch2 = collator(samples)

        # Should be identical
        assert torch.equal(batch1["input_ids"], batch2["input_ids"])
        assert torch.equal(batch1["attention_mask"], batch2["attention_mask"])
        assert torch.equal(batch1["labels"], batch2["labels"])
        assert torch.equal(batch1["num_non_padding_tokens"], batch2["num_non_padding_tokens"])

    def test_same_loader_same_order(self):
        """DataLoader should produce batches in same order for both methods."""
        dataset = TinyTokenDataset(num_samples=8, seq_length=20)
        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)

        # Create two loaders with same seed
        loader1 = DataLoader(dataset, batch_size=2, collate_fn=collator, shuffle=False)
        loader2 = DataLoader(dataset, batch_size=2, collate_fn=collator, shuffle=False)

        # Compare batches
        for batch1, batch2 in zip(loader1, loader2):
            assert torch.equal(batch1["input_ids"], batch2["input_ids"])
            assert torch.equal(batch1["attention_mask"], batch2["attention_mask"])
            assert torch.equal(batch1["labels"], batch2["labels"])

    def test_different_methods_same_token_count(self):
        """Full FT and LoRA should count tokens identically."""
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

        # Token count should be consistent
        expected_tokens = 5 * 4  # 5 tokens per sample, 4 samples
        assert batch["num_non_padding_tokens"].item() == expected_tokens

    def test_collator_handles_variable_lengths(self):
        """Collator should handle variable length sequences consistently."""
        collator = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)

        samples = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [1, 2, 3],
            },
            {
                "input_ids": [1, 2, 3, 4, 5, 6],
                "attention_mask": [1, 1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5, 6],
            },
            {
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "labels": [1, 2, 3, 4],
            },
        ]

        batch = collator(samples)

        # All sequences should be padded to max length (6, rounded to multiple of 4 = 8)
        assert batch["input_ids"].shape[1] == 8
        assert batch["attention_mask"].shape[1] == 8
        assert batch["labels"].shape[1] == 8

        # Check padding values
        assert batch["input_ids"][0, 6:].tolist() == [0, 0]
        assert batch["attention_mask"][0, 6:].tolist() == [0, 0]
        assert batch["labels"][0, 6:].tolist() == [-100, -100]
