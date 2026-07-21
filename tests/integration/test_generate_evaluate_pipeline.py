"""
Integration test: Generate and evaluate pipeline.

Validates:
- Formatted examples → generate_predictions → save_predictions_jsonl → EvaluationResult
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path

from evaluation.generate import generate_predictions, save_predictions_jsonl
from evaluation.exact_match import score_prediction, compute_exact_match
from training.results import EvaluationResult


# ============================================================================
# Tiny model for generation
# ============================================================================


class TinyGenerationModel(torch.nn.Module):
    """A tiny model that can generate text."""

    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embedding(input_ids)
        logits = self.linear(hidden)
        return type("Output", (), {"logits": logits})()

    def generate(self, input_ids, attention_mask=None, max_new_tokens=10, **kwargs):
        """Simple generate method that returns input_ids plus some dummy tokens."""
        batch_size = input_ids.shape[0]
        # Generate some dummy continuation tokens
        continuation = torch.randint(1, 50, (batch_size, max_new_tokens), dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, continuation], dim=1)


class TinyTokenizer:
    """A minimal tokenizer mock for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.padding_side = "right"
        self.pad_token_id = 0

    def __call__(self, texts, padding=False, return_tensors=None, **kwargs):
        batch_size = len(texts)
        max_len = max(len(t) % self.vocab_size + 5 for t in texts)

        input_ids = []
        attention_mask = []
        for text in texts:
            ids = [ord(c) % self.vocab_size for c in text[:max_len]]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        return result

    def batch_decode(self, token_ids, skip_special_tokens=False, **kwargs):
        texts = []
        for ids in token_ids:
            text = "".join(chr(int(i) % 128) for i in ids if i > 0)
            texts.append(text)
        return texts


# ============================================================================
# Test examples
# ============================================================================


def make_test_examples(num_examples=4):
    """Create test examples with prompts and reference answers."""
    examples = []
    for i in range(num_examples):
        examples.append({
            "example_id": f"test-{i}",
            "question": f"What is {i} + {i}?",
            "prompt": f"Solve: {i} + {i} = ?",
            "reference_answer": f"Reasoning: {i} + {i} = {i*2}\n#### {i*2}",
            "gold_answer": str(i * 2),
        })
    return examples


# ============================================================================
# Tests
# ============================================================================


class TestGenerateEvaluatePipeline:
    def test_generate_predictions_returns_records(self):
        """generate_predictions should return list of records."""
        model = TinyGenerationModel(vocab_size=100, hidden_size=32)
        tokenizer = TinyTokenizer(vocab_size=100)
        examples = make_test_examples(num_examples=4)

        records = generate_predictions(
            model,
            tokenizer,
            examples,
            batch_size=2,
            max_new_tokens=10,
        )

        assert isinstance(records, list)
        assert len(records) == 4

    def test_prediction_record_has_required_fields(self):
        """Each prediction record should have required fields."""
        model = TinyGenerationModel(vocab_size=100, hidden_size=32)
        tokenizer = TinyTokenizer(vocab_size=100)
        examples = make_test_examples(num_examples=2)

        records = generate_predictions(
            model,
            tokenizer,
            examples,
            batch_size=2,
            max_new_tokens=10,
        )

        for record in records:
            assert "example_id" in record
            assert "question" in record
            assert "prompt" in record
            assert "prediction" in record
            assert "reference_answer" in record
            assert "correct" in record

    def test_save_predictions_jsonl_creates_file(self):
        """save_predictions_jsonl should create a JSONL file."""
        records = [
            {
                "example_id": "test-0",
                "question": "What is 1+1?",
                "prompt": "Solve: 1+1",
                "prediction": "1+1=2",
                "reference_answer": "#### 2",
                "is_correct": True,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"
            save_predictions_jsonl(records, output_path)

            assert output_path.exists()

    def test_saved_jsonl_is_valid(self):
        """Saved JSONL file should contain valid JSON lines."""
        records = [
            {
                "example_id": f"test-{i}",
                "question": f"What is {i}+{i}?",
                "prompt": f"Solve: {i}+{i}",
                "prediction": f"{i}+{i}={i*2}",
                "reference_answer": f"#### {i*2}",
                "is_correct": True,
            }
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"
            save_predictions_jsonl(records, output_path)

            with open(output_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 3
            for line in lines:
                data = json.loads(line)
                assert "example_id" in data

    def test_full_pipeline_generate_to_jsonl(self):
        """Full pipeline: examples → generate → save → verify."""
        model = TinyGenerationModel(vocab_size=100, hidden_size=32)
        tokenizer = TinyTokenizer(vocab_size=100)
        examples = make_test_examples(num_examples=4)

        records = generate_predictions(
            model,
            tokenizer,
            examples,
            batch_size=2,
            max_new_tokens=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"
            save_predictions_jsonl(records, output_path)

            assert output_path.exists()
            with open(output_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 4

    def test_evaluation_result_from_predictions(self):
        """Should be able to create EvaluationResult from predictions."""
        records = [
            {
                "example_id": f"test-{i}",
                "prediction": f"answer {i}",
                "reference_answer": f"#### {i}",
                "correct": True,
            }
            for i in range(4)
        ]

        correct = sum(1 for r in records if r.get("correct", False))
        total = len(records)
        accuracy = correct / total if total > 0 else 0.0

        result = EvaluationResult(
            exact_match=accuracy,
            correct=correct,
            total=total,
            unparseable=0,
            generation_time_seconds=0.0,
        )

        assert result.exact_match == 1.0
        assert result.correct == 4
        assert result.total == 4

    def test_score_prediction_correct_match(self):
        """score_prediction should detect correct answer."""
        prediction = "The answer is 42.\n#### 42"
        reference = "Reasoning steps...\n#### 42"

        score = score_prediction(prediction, reference)

        assert score["correct"] is True

    def test_score_prediction_incorrect_match(self):
        """score_prediction should detect incorrect answer."""
        prediction = "The answer is 42.\n#### 42"
        reference = "Reasoning steps...\n#### 43"

        score = score_prediction(prediction, reference)

        assert score["correct"] is False

    def test_compute_exact_match_aggregates(self):
        """compute_exact_match should aggregate scores."""
        predictions = ["#### 1", "#### 2", "#### 3", "#### 4"]
        references = ["#### 1", "#### 2", "#### 99", "#### 4"]

        result = compute_exact_match(predictions, references)

        assert result["exact_match"] == 0.75
        assert result["correct"] == 3
        assert result["total"] == 4
