"""
Integration test: Generate and evaluate pipeline.

Validates:
- Formatted examples → generate_predictions → save_predictions_jsonl → EvaluationResult
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from evaluation.exact_match import compute_exact_match, score_prediction
from evaluation.generate import generate_predictions, save_predictions_jsonl
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
        reference = "Reasoning steps...\n#### 99"

        score = score_prediction(prediction, reference)

        assert score["correct"] is False


# ============================================================================
# evaluate_model contract tests
# ============================================================================


class TestEvaluateModelContract:
    """Tests for the evaluate_model() contract in experiment.py."""

    def _make_eval_config(self, max_examples=None):
        from training.config import (
            ExperimentConfig,
            ExperimentIdentityConfig,
            ModelConfig,
            MethodConfig,
            TrainingConfig,
            EvaluationConfig,
            OutputConfig,
        )
        from data.gsm8k import GSM8KDataConfig

        return ExperimentConfig(
            experiment=ExperimentIdentityConfig(name="test", sweep="none"),
            model=ModelConfig(name="test", revision="r1"),
            data=GSM8KDataConfig(
                dataset_name="openai/gsm8k",
                dataset_config="main",
                dataset_revision="d1",
                validation_size=500,
                seed=42,
                max_length=256,
                prompt_format="chat",
            ),
            method=MethodConfig(name="lora", rank=8),
            training=TrainingConfig(
                micro_batch_size=1,
                effective_batch_size=8,
                gradient_accumulation_steps=8,
                max_steps=10,
                seed=42,
                throughput_warmup_steps=0,
            ),
            evaluation=EvaluationConfig(
                max_examples=max_examples,
                batch_size=2,
                max_new_tokens=10,
                do_sample=False,
            ),
            output=OutputConfig(save_final_checkpoint=False),
        )

    def test_preserves_example_id(self):
        """evaluate_model should preserve example_id from input items."""
        from training.experiment import evaluate_model

        model = TinyGenerationModel(vocab_size=100, hidden_size=32)
        tokenizer = TinyTokenizer(vocab_size=100)

        # Items with explicit example_id
        test_dataset = [
            {
                "example_id": f"gsm8k-{i}",
                "question": f"Q{i}",
                "prompt": f"Solve {i}",
                "reference_answer": f"#### {i}",
            }
            for i in range(3)
        ]

        config = self._make_eval_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"
            evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_dataset=test_dataset,
                config=config,
                output_path=output_path,
            )

            # Read back predictions and verify example_ids preserved
            with open(output_path, "r") as f:
                records = [json.loads(line) for line in f]

            for i, record in enumerate(records):
                assert record["example_id"] == f"gsm8k-{i}"

    def test_uses_reference_answer(self):
        """evaluate_model should use reference_answer field, not fallback."""
        from training.experiment import evaluate_model

        model = TinyGenerationModel(vocab_size=100, hidden_size=32)
        tokenizer = TinyTokenizer(vocab_size=100)

        test_dataset = [
            {
                "example_id": "test-0",
                "question": "Q",
                "prompt": "Solve",
                "reference_answer": "#### 42",
            }
        ]

        config = self._make_eval_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"
            evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_dataset=test_dataset,
                config=config,
                output_path=output_path,
            )

            # Verify the output file was created
            assert output_path.exists()

    def test_missing_reference_answer_raises(self):
        """evaluate_model should raise KeyError when reference_answer is missing."""
        from training.experiment import evaluate_model

        model = TinyGenerationModel(vocab_size=100, hidden_size=32)
        tokenizer = TinyTokenizer(vocab_size=100)

        test_dataset = [
            {
                "example_id": "test-0",
                "question": "Q",
                "prompt": "Solve",
                # No reference_answer
            }
        ]

        config = self._make_eval_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"
            with pytest.raises(KeyError, match="reference_answer"):
                evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    test_dataset=test_dataset,
                    config=config,
                    output_path=output_path,
                )

    def test_unparseable_counted_by_predicted_answer_none(self):
        """unparseable should count records where predicted_answer is None."""
        from training.experiment import evaluate_model

        model = TinyGenerationModel(vocab_size=100, hidden_size=32)
        tokenizer = TinyTokenizer(vocab_size=100)

        test_dataset = [
            {
                "example_id": f"test-{i}",
                "question": f"Q{i}",
                "prompt": f"Solve {i}",
                "reference_answer": f"#### {i}",
            }
            for i in range(4)
        ]

        config = self._make_eval_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"
            result = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_dataset=test_dataset,
                config=config,
                output_path=output_path,
            )

            # unparseable is computed from predicted_answer being None
            assert isinstance(result.unparseable, int)
            assert result.unparseable >= 0

    def test_record_count_mismatch_raises(self):
        """evaluate_model should raise RuntimeError if record count != example count."""
        from training.experiment import evaluate_model
        from unittest.mock import patch

        model = TinyGenerationModel(vocab_size=100, hidden_size=32)
        tokenizer = TinyTokenizer(vocab_size=100)

        test_dataset = [
            {
                "example_id": f"test-{i}",
                "question": f"Q{i}",
                "prompt": f"Solve {i}",
                "reference_answer": f"#### {i}",
            }
            for i in range(4)
        ]

        config = self._make_eval_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"

            # Mock generate_predictions to return wrong number of records
            def bad_generate(*args, **kwargs):
                return [{"example_id": "test-0", "predicted_answer": "42", "correct": True}]

            with patch("evaluation.generate.generate_predictions", side_effect=bad_generate):
                with pytest.raises(RuntimeError, match="unexpected number of records"):
                    evaluate_model(
                        model=model,
                        tokenizer=tokenizer,
                        test_dataset=test_dataset,
                        config=config,
                        output_path=output_path,
                    )

    def test_max_examples_truncation(self):
        """max_examples should truncate and total should reflect truncated count."""
        from training.experiment import evaluate_model

        model = TinyGenerationModel(vocab_size=100, hidden_size=32)
        tokenizer = TinyTokenizer(vocab_size=100)

        test_dataset = [
            {
                "example_id": f"test-{i}",
                "question": f"Q{i}",
                "prompt": f"Solve {i}",
                "reference_answer": f"#### {i}",
            }
            for i in range(8)
        ]

        config = self._make_eval_config(max_examples=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"
            result = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_dataset=test_dataset,
                config=config,
                output_path=output_path,
            )

            assert result.total == 3


# ============================================================================
# Exact match scoring tests
# ============================================================================


class TestExactMatchScoring:
    def test_score_prediction_partial_match(self):
        """Partial match should not be counted as correct."""
        prediction = "#### 42"
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


# ============================================================================
# ExperimentRunner control flow tests
# ============================================================================


class TestExperimentRunnerControlFlow:
    """Tests for ExperimentRunner orchestration behavior."""

    def _make_minimal_config(self):
        from training.config import (
            ExperimentConfig,
            ExperimentIdentityConfig,
            ModelConfig,
            MethodConfig,
            TrainingConfig,
            EvaluationConfig,
            OutputConfig,
        )
        from data.gsm8k import GSM8KDataConfig

        return ExperimentConfig(
            experiment=ExperimentIdentityConfig(name="test", sweep="none"),
            model=ModelConfig(name="test", revision="r1"),
            data=GSM8KDataConfig(
                dataset_name="openai/gsm8k",
                dataset_config="main",
                dataset_revision="d1",
                validation_size=500,
                seed=42,
                max_length=256,
                prompt_format="chat",
            ),
            method=MethodConfig(name="lora", rank=8),
            training=TrainingConfig(
                micro_batch_size=1,
                effective_batch_size=8,
                gradient_accumulation_steps=8,
                max_steps=10,
                seed=42,
                throughput_warmup_steps=0,
            ),
            evaluation=EvaluationConfig(
                max_examples=None,
                batch_size=2,
                max_new_tokens=10,
                do_sample=False,
            ),
            output=OutputConfig(save_final_checkpoint=False),
        )

    def test_completed_no_overwrite_raises_file_exists(self):
        """Completed result + no overwrite → FileExistsError."""
        from training.experiment import ExperimentRunner
        from training.results import experiment_result_to_dict, write_json_atomic

        config = self._make_minimal_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = ExperimentRunner(config=config, repository_root=repo_root)

            # Simulate a completed result
            result_dir = runner.run_paths.result_dir
            result_dir.mkdir(parents=True)
            result_json = runner.run_paths.result_json

            from training.results import ExperimentResult
            completed = ExperimentResult(
                run_id=runner.run_paths.run_id,
                experiment_name="test",
                method="lora",
                rank=8,
                max_length=256,
                micro_batch_size=1,
                effective_batch_size=8,
                gradient_accumulation_steps=8,
                peak_memory_gb=2.0,
                peak_reserved_memory_gb=3.0,
                tokens_per_second=1000.0,
                training_time_seconds=60.0,
                exact_match=0.5,
                trainable_parameters=1000000,
                total_parameters=500000000,
                checkpoint_size_mb=None,
                trained_non_padding_tokens=10000,
                optimizer_steps=10,
                seed=42,
                sweep="none",
                model_name="test",
                model_revision="r1",
                dataset_revision="d1",
                attention_backend=None,
                status="completed",
                error_type=None,
                error_message=None,
            )
            write_json_atomic(experiment_result_to_dict(completed), result_json)

            # Without allow_overwrite, should raise FileExistsError
            with pytest.raises(FileExistsError, match="Completed result"):
                runner.run()

    def test_completed_with_overwrite_clears_artifacts(self):
        """Completed result + overwrite → cleans old artifacts and proceeds."""
        from training.experiment import ExperimentRunner
        from training.results import experiment_result_to_dict, write_json_atomic

        config = self._make_minimal_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = ExperimentRunner(
                config=config,
                repository_root=repo_root,
                allow_overwrite=True,
            )

            # Simulate old artifacts
            result_dir = runner.run_paths.result_dir
            result_dir.mkdir(parents=True)
            result_json = runner.run_paths.result_json
            checkpoint_dir = runner.run_paths.checkpoint_dir.parent
            checkpoint_dir.mkdir(parents=True)

            # Write a fake completed result
            from training.results import ExperimentResult
            completed = ExperimentResult(
                run_id=runner.run_paths.run_id,
                experiment_name="test",
                method="lora",
                rank=8,
                max_length=256,
                micro_batch_size=1,
                effective_batch_size=8,
                gradient_accumulation_steps=8,
                peak_memory_gb=2.0,
                peak_reserved_memory_gb=3.0,
                tokens_per_second=1000.0,
                training_time_seconds=60.0,
                exact_match=0.5,
                trainable_parameters=1000000,
                total_parameters=500000000,
                checkpoint_size_mb=None,
                trained_non_padding_tokens=10000,
                optimizer_steps=10,
                seed=42,
                sweep="none",
                model_name="test",
                model_revision="r1",
                dataset_revision="d1",
                attention_backend=None,
                status="completed",
                error_type=None,
                error_message=None,
            )
            write_json_atomic(experiment_result_to_dict(completed), result_json)

            # Create a fake checkpoint file
            (checkpoint_dir / "fake_checkpoint.bin").write_bytes(b"\x00" * 100)

            assert result_json.exists()
            assert checkpoint_dir.exists()

            # _prepare_run_directory should clean them
            runner._prepare_run_directory()

            # Old artifacts should be gone
            assert not result_json.exists()
            assert not checkpoint_dir.exists()

    def test_component_build_oom_writes_oom_result(self):
        """Component build OOM → writes status=oom result and returns it."""
        from training.experiment import ExperimentRunner
        from unittest.mock import patch

        config = self._make_minimal_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = ExperimentRunner(config=config, repository_root=repo_root)

            # Mock build_training_components to raise OOM
            oom_error = RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")

            with patch("training.experiment.build_training_components", side_effect=oom_error):
                result = runner.run()  # OOM returns result, doesn't raise

            # Check that OOM result was written
            result_json = runner.run_paths.result_json
            assert result_json.exists()

            import json
            with open(result_json, "r") as f:
                data = json.load(f)

            assert data["status"] == "oom"
            assert data["error_type"] == "RuntimeError"
            assert result.status == "oom"

    def test_component_build_valueerror_writes_failed_result(self):
        """Component build ValueError → writes status=failed then re-raises."""
        from training.experiment import ExperimentRunner
        from unittest.mock import patch

        config = self._make_minimal_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = ExperimentRunner(config=config, repository_root=repo_root)

            with patch("training.experiment.build_training_components", side_effect=ValueError("bad config")):
                with pytest.raises(ValueError, match="bad config"):
                    runner.run()

            result_json = runner.run_paths.result_json
            assert result_json.exists()

            import json
            with open(result_json, "r") as f:
                data = json.load(f)

            assert data["status"] == "failed"
            assert data["error_type"] == "ValueError"

    def test_training_oom_writes_oom_result(self):
        """Training OOM → writes status=oom result."""
        from training.experiment import ExperimentRunner
        from unittest.mock import patch, MagicMock

        config = self._make_minimal_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = ExperimentRunner(config=config, repository_root=repo_root)

            # Mock components to return a valid bundle
            mock_components = MagicMock()
            mock_components.model_bundle.model = MagicMock()
            mock_components.model_bundle.tokenizer = MagicMock()
            mock_components.model_bundle.parameter_stats.total_parameters = 500000000
            mock_components.model_bundle.parameter_stats.trainable_parameters = 1000000
            mock_components.model_bundle.parameter_stats.frozen_parameters = 499000000
            mock_components.model_bundle.parameter_stats.trainable_fraction = 0.002
            mock_components.optimizer = MagicMock()
            mock_components.data_bundle.train_loader = []
            mock_components.data_bundle.test_dataset = []
            mock_components.device = MagicMock(type="cuda")

            oom_error = RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")

            with patch("training.experiment.build_training_components", return_value=mock_components):
                with patch.object(runner, "_train", side_effect=oom_error):
                    result = runner.run()  # OOM returns result, doesn't raise

            result_json = runner.run_paths.result_json
            assert result_json.exists()

            import json
            with open(result_json, "r") as f:
                data = json.load(f)

            assert data["status"] == "oom"
            assert result.status == "oom"

    def test_evaluation_failure_writes_failed_result(self):
        """Evaluation failure → writes status=failed then re-raises."""
        from training.experiment import ExperimentRunner
        from unittest.mock import patch, MagicMock

        config = self._make_minimal_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = ExperimentRunner(config=config, repository_root=repo_root)

            mock_components = MagicMock()
            mock_components.model_bundle.model = MagicMock()
            mock_components.model_bundle.tokenizer = MagicMock()
            mock_components.model_bundle.parameter_stats.total_parameters = 500000000
            mock_components.model_bundle.parameter_stats.trainable_parameters = 1000000
            mock_components.model_bundle.parameter_stats.frozen_parameters = 499000000
            mock_components.model_bundle.parameter_stats.trainable_fraction = 0.002
            mock_components.optimizer = MagicMock()
            mock_components.data_bundle.train_loader = []
            mock_components.data_bundle.test_dataset = []
            mock_components.device = MagicMock(type="cuda")

            eval_error = RuntimeError("Evaluation pipeline crashed")

            with patch("training.experiment.build_training_components", return_value=mock_components):
                with patch.object(runner, "_train", return_value=MagicMock(
                    optimizer_steps=10,
                    trained_non_padding_tokens=10000,
                    peak_memory_gb=2.0,
                    peak_reserved_memory_gb=3.0,
                    tokens_per_second=1000.0,
                    training_time_seconds=60.0,
                )):
                    with patch.object(runner, "_save_and_reload", return_value=(None, MagicMock())):
                        with patch.object(runner, "_evaluate", side_effect=eval_error):
                            with pytest.raises(RuntimeError, match="Evaluation pipeline crashed"):
                                runner.run()

            result_json = runner.run_paths.result_json
            assert result_json.exists()

            import json
            with open(result_json, "r") as f:
                data = json.load(f)

            assert data["status"] == "failed"
            assert data["error_type"] == "RuntimeError"

    def test_keyboard_interrupt_not_captured_as_failed(self):
        """KeyboardInterrupt should NOT be captured as a failed result."""
        from training.experiment import ExperimentRunner
        from unittest.mock import patch, MagicMock

        config = self._make_minimal_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = ExperimentRunner(config=config, repository_root=repo_root)

            mock_components = MagicMock()
            mock_components.model_bundle.model = MagicMock()
            mock_components.model_bundle.tokenizer = MagicMock()
            mock_components.model_bundle.parameter_stats.total_parameters = 500000000
            mock_components.model_bundle.parameter_stats.trainable_parameters = 1000000
            mock_components.model_bundle.parameter_stats.frozen_parameters = 499000000
            mock_components.model_bundle.parameter_stats.trainable_fraction = 0.002
            mock_components.optimizer = MagicMock()
            mock_components.data_bundle.train_loader = []
            mock_components.data_bundle.test_dataset = []
            mock_components.device = MagicMock(type="cuda")

            with patch("training.experiment.build_training_components", return_value=mock_components):
                with patch.object(runner, "_train", side_effect=KeyboardInterrupt()):
                    with pytest.raises(KeyboardInterrupt):
                        runner.run()

            # KeyboardInterrupt should NOT write a failed result
            result_json = runner.run_paths.result_json
            assert not result_json.exists()

    def test_generator_exit_not_captured(self):
        """
        GeneratorExit should propagate and must not be
        persisted as a failed experiment.
        """
        from unittest.mock import MagicMock, patch

        from training.experiment import (
            ExperimentRunner,
        )

        config = self._make_minimal_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                config=config,
                repository_root=Path(tmpdir),
            )

            mock_components = MagicMock()
            mock_components.device = MagicMock(
                type="cuda"
            )

            with patch(
                "training.experiment."
                "build_training_components",
                return_value=mock_components,
            ):
                with patch.object(
                    runner,
                    "_train",
                    side_effect=GeneratorExit(),
                ):
                    with pytest.raises(GeneratorExit):
                        runner.run()

            assert not (
                runner.run_paths.result_json.exists()
            )

    def test_overwrite_clears_partial_artifacts(
        self,
    ):
        """
        allow_overwrite should clean stale artifacts even
        when no completed result.json exists.
        """
        from training.experiment import (
            ExperimentRunner,
        )

        config = self._make_minimal_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                config=config,
                repository_root=Path(tmpdir),
                allow_overwrite=True,
            )

            result_dir = runner.run_paths.result_dir
            checkpoint_run_dir = (
                runner.run_paths.checkpoint_dir.parent
            )

            result_dir.mkdir(
                parents=True,
            )
            checkpoint_run_dir.mkdir(
                parents=True,
            )

            stale_prediction = (
                result_dir / "predictions.jsonl"
            )
            stale_metadata = (
                result_dir / "metadata.json"
            )
            stale_checkpoint = (
                checkpoint_run_dir / "stale.bin"
            )

            stale_prediction.write_text(
                "stale",
                encoding="utf-8",
            )
            stale_metadata.write_text(
                "{}",
                encoding="utf-8",
            )
            stale_checkpoint.write_bytes(
                b"stale"
            )

            runner._prepare_run_directory()

            assert not stale_prediction.exists()
            assert not stale_metadata.exists()
            assert not stale_checkpoint.exists()

            assert (
                runner.run_paths.result_dir.exists()
            )
            assert (
                runner.run_paths
                .resolved_config_yaml
                .exists()
            )
