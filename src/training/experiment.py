'''
Experiment orchestration module.

Defines ExperimentRunner which coordinates the full training and evaluation pipeline.
'''

from __future__ import annotations

import datetime
import json
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from metrics.memory import CudaMemoryTracker, is_cuda_oom
from metrics.throughput import ThroughputTracker
from training.checkpoint import (
    CheckpointInfo,
    save_final_checkpoint,
    verify_checkpoint_reload,
)
from training.config import write_resolved_config
from training.engine import TrainerEngine
from training.results import (
    EvaluationResult,
    ExperimentResult,
    TrainingResult,
    experiment_result_to_dict,
    validate_experiment_result,
    write_json_atomic,
)
from training.setup import (
    TrainingComponents,
    build_training_components,
)

if TYPE_CHECKING:
    from training.config import ExperimentConfig


# ============================================================================
# Run paths
# ============================================================================


@dataclass(frozen=True)
class RunPaths:
    """Immutable container for all run-specific paths."""
    run_id: str
    result_dir: Path
    result_json: Path
    predictions_jsonl: Path
    resolved_config_yaml: Path
    metadata_json: Path
    checkpoint_dir: Path


# ============================================================================
# Run ID and paths
# ============================================================================


def build_run_id(config: ExperimentConfig) -> str:
    """
    Generate a deterministic, filesystem-safe identifier.

    Responsibilities:
    - include experiment name, method, relevant rank, length, batch, and seed
    - reject path separators and unsafe characters
    """
    method = config.method.name
    rank = config.method.rank
    max_length = config.data.max_length
    micro_batch_size = config.training.micro_batch_size
    seed = config.training.seed
    name = config.experiment.name

    parts = [name, method]

    if rank is not None:
        parts.append(f"r{rank}")

    parts.append(f"l{max_length}")
    parts.append(f"mb{micro_batch_size}")
    parts.append(f"seed{seed}")

    run_id = "_".join(parts)

    # Whitelist: must start with alphanumeric, then only alphanumerics, dots, underscores, hyphens
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(f"run_id contains unsafe characters: {run_id}")

    return run_id


def _require_within(
    path: Path,
    parent: Path,
    *,
    label: str,
) -> Path:
    """Ensure *path* resolves strictly inside *parent*."""
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()

    try:
        resolved_path.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes allowed root: "
            f"{resolved_path}"
        ) from exc

    return resolved_path


def build_run_paths(config: ExperimentConfig, repository_root: Path) -> RunPaths:
    """
    Resolve result and checkpoint root directories from config.

    Responsibilities:
    - resolve configured result and checkpoint roots
    - do not create any files yet
    - prevent path escape from configured directories
    """
    repository_root = Path(repository_root).resolve()

    run_id = build_run_id(config)

    # Resolve and validate root directories
    results_root = _require_within(
        repository_root / config.output.results_dir,
        repository_root,
        label="results_dir",
    )
    checkpoints_root = _require_within(
        repository_root / config.output.checkpoints_dir,
        repository_root,
        label="checkpoints_dir",
    )

    # Validate run-specific paths
    result_dir = _require_within(
        results_root / run_id,
        results_root,
        label="result run directory",
    )
    checkpoint_dir = _require_within(
        checkpoints_root / run_id / "final",
        checkpoints_root,
        label="checkpoint run directory",
    )

    return RunPaths(
        run_id=run_id,
        result_dir=result_dir,
        result_json=result_dir / "result.json",
        predictions_jsonl=result_dir / "predictions.jsonl",
        resolved_config_yaml=result_dir / "resolved_config.yaml",
        metadata_json=result_dir / "metadata.json",
        checkpoint_dir=checkpoint_dir,
    )


# ============================================================================
# Metadata collection
# ============================================================================


def _get_git_commit(repo_root: Path) -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_git_dirty(repo_root: Path) -> bool:
    """Check if git working tree is dirty."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _get_package_version(package_name: str) -> str | None:
    """Get installed package version."""
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def collect_run_metadata(
    config: ExperimentConfig,
    components: TrainingComponents,
    repository_root: Path,
) -> dict[str, object]:
    """
    Collect comprehensive run metadata.

    Must record:
    - UTC timestamp
    - hostname
    - git commit
    - git dirty status
    - Python version
    - PyTorch version
    - Transformers version
    - PEFT version
    - CUDA runtime
    - GPU name
    - GPU total memory
    - selected attention backend
    - model and dataset versions
    - resolved paths
    - parameter statistics
    """

    try:
        import peft
        peft_version = peft.__version__
    except Exception:
        peft_version = None

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_version = torch.version.cuda
    else:
        gpu_name = None
        gpu_total_memory_gb = None
        cuda_version = None

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git_commit": _get_git_commit(repository_root),
        "git_dirty": _get_git_dirty(repository_root),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": _get_package_version("transformers"),
        "peft_version": peft_version,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "gpu_total_memory_gb": gpu_total_memory_gb,
        "attention_backend": config.model.attention_backend,
        "model_name": config.model.name,
        "model_revision": config.model.revision,
        "dataset_revision": config.data.dataset_revision,
        "parameter_stats": {
            "total_parameters": components.model_bundle.parameter_stats.total_parameters,
            "trainable_parameters": components.model_bundle.parameter_stats.trainable_parameters,
            "frozen_parameters": components.model_bundle.parameter_stats.frozen_parameters,
            "trainable_fraction": components.model_bundle.parameter_stats.trainable_fraction,
        },
    }


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: object,
    test_dataset: object,
    config: ExperimentConfig,
    output_path: Path,
) -> EvaluationResult:
    """
    Evaluate model on test dataset.

    Responsibilities:
    1. optionally select first max_examples samples
    2. call existing generate_predictions
    3. call existing save_predictions_jsonl
    4. aggregate correct, total, and exact match
    5. count unparseable predictions
    6. measure generation time
    7. return EvaluationResult
    """
    from evaluation.generate import generate_predictions, save_predictions_jsonl

    # Convert dataset to examples list
    examples = []
    for idx, item in enumerate(test_dataset):
        if "reference_answer" not in item:
            raise KeyError(
                "Evaluation example is missing "
                "'reference_answer'."
            )

        examples.append({
            "example_id": str(item.get("example_id", idx)),
            "question": str(item.get("question", "")),
            "prompt": str(item["prompt"]),
            "reference_answer": str(item["reference_answer"]),
        })

    # Optionally limit examples
    max_examples = config.evaluation.max_examples
    if max_examples is not None:
        examples = examples[:max_examples]

    # Generate predictions
    start_time = time.perf_counter()
    records = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=config.evaluation.batch_size,
        max_new_tokens=config.evaluation.max_new_tokens,
        generation_kwargs={"do_sample": config.evaluation.do_sample},
    )
    generation_time = time.perf_counter() - start_time

    # Validate record count
    if len(records) != len(examples):
        raise RuntimeError(
            "Generation returned an unexpected "
            "number of records."
        )

    # Save predictions
    save_predictions_jsonl(records, output_path)

    # Aggregate results
    total = len(records)
    correct = sum(1 for r in records if r.get("correct", False))
    unparseable = sum(
        1
        for record in records
        if record.get("predicted_answer") is None
    )
    exact_match = correct / total if total > 0 else 0.0

    return EvaluationResult(
        exact_match=exact_match,
        correct=correct,
        total=total,
        unparseable=unparseable,
        generation_time_seconds=generation_time,
    )


# ============================================================================
# ExperimentRunner
# ============================================================================


class ExperimentRunner:
    """
    End-to-end experiment orchestrator.

    Coordinates the full pipeline:
    config → setup → train → save → reload → generate → evaluate → result.json
    """

    def __init__(
        self,
        config: ExperimentConfig,
        repository_root: Path,
        *,
        allow_overwrite: bool = False,
    ) -> None:
        self.config = config
        self.repository_root = Path(repository_root).resolve()
        self.allow_overwrite = allow_overwrite
        self.run_paths = build_run_paths(config, self.repository_root)

    def _prepare_run_directory(self) -> None:
        """
        Prepare a clean run directory.

        Without allow_overwrite, a completed result is
        protected. With allow_overwrite, all artifacts for
        this run ID are removed before execution.
        """
        import shutil

        result_dir = self.run_paths.result_dir
        result_json = self.run_paths.result_json
        checkpoint_run_dir = (
            self.run_paths.checkpoint_dir.parent
        )

        if self.allow_overwrite:
            if result_dir.exists():
                shutil.rmtree(result_dir)

            if checkpoint_run_dir.exists():
                shutil.rmtree(
                    checkpoint_run_dir
                )

        elif result_json.exists():
            try:
                with open(
                    result_json,
                    "r",
                    encoding="utf-8",
                ) as handle:
                    existing = json.load(handle)
            except json.JSONDecodeError:
                existing = None

            if (
                existing is not None
                and existing.get("status")
                == "completed"
            ):
                raise FileExistsError(
                    "Completed result already exists: "
                    f"{result_json}"
                )

        result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_resolved_config(
            self.config,
            self.run_paths.resolved_config_yaml,
        )

    def _build_components(self) -> TrainingComponents:
        """
        Build all training components.

        Responsibilities:
        - delegate to build_training_components
        - avoid rebuilding setup logic
        """
        return build_training_components(self.config)

    def _train(self, components: TrainingComponents) -> TrainingResult:
        """
        Execute training.

        Responsibilities:
        - build metric trackers
        - instantiate TrainerEngine
        - delegate training
        """
        device = components.device
        config = self.config

        # Build trackers
        memory_tracker = CudaMemoryTracker(device)
        throughput_tracker = ThroughputTracker(
            device=device,
            warmup_optimizer_steps=config.training.throughput_warmup_steps,
        )

        # Build engine
        engine = TrainerEngine(
            config=config,
            device=device,
            memory_tracker=memory_tracker,
        )

        # Execute training
        return engine.train(
            model=components.model_bundle.model,
            optimizer=components.optimizer,
            train_loader=components.data_bundle.train_loader,
            throughput_tracker=throughput_tracker,
        )

    def _save_and_reload(
        self,
        components: TrainingComponents,
    ) -> tuple[CheckpointInfo | None, object]:
        """
        Save final artifact and reload.

        Responsibilities:
        - save final artifact when enabled
        - reload
        - return checkpoint metadata and reloaded model

        If checkpoint saving is disabled:
        - checkpoint size is None
        - evaluation can use in-memory model
        """
        if not self.config.output.save_final_checkpoint:
            return None, components.model_bundle.model

        # Save checkpoint
        checkpoint_info = save_final_checkpoint(
            model=components.model_bundle.model,
            tokenizer=components.model_bundle.tokenizer,
            config=self.config,
            path=self.run_paths.checkpoint_dir,
        )

        # Reload and verify
        reloaded_model = verify_checkpoint_reload(
            config=self.config,
            checkpoint_path=self.run_paths.checkpoint_dir,
            tokenizer=components.model_bundle.tokenizer,
            device=components.device,
        )

        # Mark as verified
        checkpoint_info = CheckpointInfo(
            path=checkpoint_info.path,
            method=checkpoint_info.method,
            size_bytes=checkpoint_info.size_bytes,
            size_mb=checkpoint_info.size_mb,
            reload_verified=True,
        )

        return checkpoint_info, reloaded_model

    def _evaluate(
        self,
        model: object,
        components: TrainingComponents,
    ) -> EvaluationResult:
        """
        Evaluate model and save predictions.

        Responsibilities:
        - call evaluate_model
        - save predictions to run result directory
        """
        return evaluate_model(
            model=model,
            tokenizer=components.model_bundle.tokenizer,
            test_dataset=components.data_bundle.test_dataset,
            config=self.config,
            output_path=self.run_paths.predictions_jsonl,
        )

    def _build_completed_result(
        self,
        training_result: TrainingResult,
        checkpoint_info: CheckpointInfo | None,
        evaluation_result: EvaluationResult,
    ) -> ExperimentResult:
        """
        Build completed experiment result.

        Responsibilities:
        - combine config, parameters, training, checkpoint, and evaluation metrics
        - set status="completed"
        - validate before writing
        """
        param_stats = self._components.model_bundle.parameter_stats

        result = ExperimentResult(
            run_id=self.run_paths.run_id,
            experiment_name=self.config.experiment.name,
            method=self.config.method.name,
            rank=self.config.method.rank,
            max_length=self.config.data.max_length,
            micro_batch_size=self.config.training.micro_batch_size,
            effective_batch_size=self.config.training.effective_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            peak_memory_gb=training_result.peak_memory_gb,
            peak_reserved_memory_gb=training_result.peak_reserved_memory_gb,
            tokens_per_second=training_result.tokens_per_second,
            training_time_seconds=training_result.training_time_seconds,
            exact_match=evaluation_result.exact_match,
            trainable_parameters=param_stats.trainable_parameters,
            total_parameters=param_stats.total_parameters,
            checkpoint_size_mb=checkpoint_info.size_mb if checkpoint_info else None,
            trained_non_padding_tokens=training_result.trained_non_padding_tokens,
            optimizer_steps=training_result.optimizer_steps,
            seed=self.config.training.seed,
            sweep=self.config.experiment.sweep,
            model_name=self.config.model.name,
            model_revision=self.config.model.revision,
            dataset_revision=self.config.data.dataset_revision,
            attention_backend=self.config.model.attention_backend,
            status="completed",
            error_type=None,
            error_message=None,
        )

        validate_experiment_result(result)
        return result

    def _build_oom_result(
        self,
        error: Exception,
    ) -> ExperimentResult:
        """
        Build OOM experiment result.

        Responsibilities:
        - set status="oom"
        - preserve identity config fields
        - set unavailable metrics to None
        - record error type and concise message
        """
        return ExperimentResult(
            run_id=self.run_paths.run_id,
            experiment_name=self.config.experiment.name,
            method=self.config.method.name,
            rank=self.config.method.rank,
            max_length=self.config.data.max_length,
            micro_batch_size=self.config.training.micro_batch_size,
            effective_batch_size=self.config.training.effective_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            peak_memory_gb=None,
            peak_reserved_memory_gb=None,
            tokens_per_second=None,
            training_time_seconds=None,
            exact_match=None,
            trainable_parameters=None,
            total_parameters=None,
            checkpoint_size_mb=None,
            trained_non_padding_tokens=None,
            optimizer_steps=None,
            seed=self.config.training.seed,
            sweep=self.config.experiment.sweep,
            model_name=self.config.model.name,
            model_revision=self.config.model.revision,
            dataset_revision=self.config.data.dataset_revision,
            attention_backend=self.config.model.attention_backend,
            status="oom",
            error_type=type(error).__name__,
            error_message=str(error)[:500],
        )

    def _build_failed_result(
        self,
        error: Exception,
    ) -> ExperimentResult:
        """
        Build failed experiment result.

        Responsibilities:
        - set status="failed"
        - distinguish implementation failure from OOM
        - preserve diagnostic information
        - do not mark failed runs as completed
        """
        return ExperimentResult(
            run_id=self.run_paths.run_id,
            experiment_name=self.config.experiment.name,
            method=self.config.method.name,
            rank=self.config.method.rank,
            max_length=self.config.data.max_length,
            micro_batch_size=self.config.training.micro_batch_size,
            effective_batch_size=self.config.training.effective_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            peak_memory_gb=None,
            peak_reserved_memory_gb=None,
            tokens_per_second=None,
            training_time_seconds=None,
            exact_match=None,
            trainable_parameters=None,
            total_parameters=None,
            checkpoint_size_mb=None,
            trained_non_padding_tokens=None,
            optimizer_steps=None,
            seed=self.config.training.seed,
            sweep=self.config.experiment.sweep,
            model_name=self.config.model.name,
            model_revision=self.config.model.revision,
            dataset_revision=self.config.data.dataset_revision,
            attention_backend=self.config.model.attention_backend,
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error)[:500],
        )

    def run(self) -> ExperimentResult:
        """
        Execute the full experiment pipeline.

        Orchestration:
        1. prepare paths
        2. write resolved config
        3. build components
        4. write metadata
        5. train
        6. save final artifact
        7. reload final artifact
        8. evaluate
        9. build completed result
        10. write result.json
        11. return result

        Exception behavior:
        CUDA OOM
        → write status=oom
        → release references
        → return OOM result

        Other exceptions
        → write status=failed
        → log failure and re-raise
        """
        # Prepare run directory
        self._prepare_run_directory()

        try:
            # Build components
            self._components = self._build_components()

            # Write metadata
            metadata = collect_run_metadata(
                self.config,
                self._components,
                self.repository_root,
            )
            write_json_atomic(metadata, self.run_paths.metadata_json)

            # Train
            training_result = self._train(self._components)

            # Save and reload
            checkpoint_info, reloaded_model = self._save_and_reload(self._components)

            # Evaluate
            evaluation_result = self._evaluate(reloaded_model, self._components)

            # Build and write completed result
            result = self._build_completed_result(
                training_result=training_result,
                checkpoint_info=checkpoint_info,
                evaluation_result=evaluation_result,
            )

            write_json_atomic(
                experiment_result_to_dict(result),
                self.run_paths.result_json,
            )

            return result

        except Exception as error:
            oom = is_cuda_oom(error)

            if oom:
                result = self._build_oom_result(
                    error
                )
            else:
                result = self._build_failed_result(
                    error
                )

            try:
                validate_experiment_result(result)

                write_json_atomic(
                    experiment_result_to_dict(result),
                    self.run_paths.result_json,
                )
            finally:
                if hasattr(self, "_components"):
                    del self._components

            if not oom:
                raise

            return result
