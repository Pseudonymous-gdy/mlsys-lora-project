'''
Training configuration module.

This module is the only module allowed to parse YAML.
'''

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from data.gsm8k import GSM8KDataConfig

# ============================================================================
# Configuration dataclasses
# ============================================================================


@dataclass(frozen=True)
class ExperimentIdentityConfig:
    """Identifies the experiment and sweep it belongs to."""
    name: str
    sweep: str


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model loading."""
    name: str
    revision: str
    fallback_name: str | None = None
    fallback_revision: str | None = None
    trust_remote_code: bool = False
    attention_backend: str | None = None


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for the training method (Full FT or LoRA)."""
    name: Literal["full_ft", "lora"]
    rank: int | None = None
    alpha: int | None = None
    dropout: float | None = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the training loop."""
    seed: int = 42
    precision: Literal["bf16", "fp32"] = "bf16"
    micro_batch_size: int = 1
    effective_batch_size: int = 16
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int | None = None
    training_token_budget: int | None = 1_000_000
    num_workers: int = 4
    pin_memory: bool = True
    throughput_warmup_steps: int = 3
    gradient_clip_norm: float | None = 1.0


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluation/generation."""
    batch_size: int = 8
    max_new_tokens: int = 512
    do_sample: bool = False
    max_examples: int | None = None
    # Tuning sweeps must score the held-out validation split. The official
    # test split is reserved for the final reported comparison.
    split: Literal["validation", "test"] = "test"


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for output paths and persistence."""
    results_dir: Path = field(default_factory=lambda: Path("results"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_final_checkpoint: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""
    experiment: ExperimentIdentityConfig
    model: ModelConfig
    data: GSM8KDataConfig
    method: MethodConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig


# ============================================================================
# YAML loading and validation
# ============================================================================


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    """Safely load a YAML file and return the parsed document."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if doc is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(doc, dict):
        raise ValueError(
            f"Configuration must be a YAML mapping, got {type(doc).__name__}: {path}"
        )

    return doc


def _resolve_path(value: Any) -> Path:
    """Convert a value to a Path object."""
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _resolve_tuple(value: Any) -> tuple[str, ...]:
    """Convert a value to a tuple of strings."""
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    if isinstance(value, str):
        return (value,)
    return ()


def _parse_data_config(raw: dict[str, Any]) -> GSM8KDataConfig:
    """Parse the data section into a GSM8KDataConfig."""
    data = raw.get("data", {})
    return GSM8KDataConfig(
        dataset_name=data.get("name", "openai/gsm8k"),
        dataset_config=data.get("config", "main"),
        dataset_revision=data.get("revision", ""),
        validation_size=data.get("validation_size", 500),
        seed=data.get("split_seed", 42),
        max_length=data.get("max_length", 512),
        prompt_format=data.get("prompt_format", "chat"),
        system_prompt=data.get("system_prompt", None),
    )


def _parse_experiment_config(raw: dict[str, Any]) -> ExperimentIdentityConfig:
    """Parse the experiment identity section."""
    exp = raw.get("experiment", {})
    return ExperimentIdentityConfig(
        name=exp.get("name", ""),
        sweep=exp.get("sweep", ""),
    )


def _parse_model_config(raw: dict[str, Any]) -> ModelConfig:
    """Parse the model section."""
    model = raw.get("model", {})
    return ModelConfig(
        name=model.get("name", ""),
        revision=model.get("revision", ""),
        fallback_name=model.get("fallback_name"),
        fallback_revision=model.get("fallback_revision"),
        trust_remote_code=model.get("trust_remote_code", False),
        attention_backend=model.get("attention_backend"),
    )


def _parse_method_config(raw: dict[str, Any]) -> MethodConfig:
    """Parse the method section."""
    method = raw.get("method", {})
    return MethodConfig(
        name=method.get("name", "lora"),
        rank=method.get("rank"),
        alpha=method.get("alpha"),
        dropout=method.get("dropout", 0.05),
        target_modules=_resolve_tuple(method.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
        bias=method.get("bias", "none"),
    )


def _parse_training_config(raw: dict[str, Any]) -> TrainingConfig:
    """Parse the training section."""
    training = raw.get("training", {})
    return TrainingConfig(
        seed=training.get("seed", 42),
        precision=training.get("precision", "bf16"),
        micro_batch_size=training.get("micro_batch_size", 1),
        effective_batch_size=training.get("effective_batch_size", 16),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 16),
        learning_rate=training.get("learning_rate", 2e-4),
        weight_decay=training.get("weight_decay", 0.0),
        max_steps=training.get("max_steps"),
        training_token_budget=training.get("training_token_budget", 1_000_000),
        num_workers=training.get("num_workers", 4),
        pin_memory=training.get("pin_memory", True),
        throughput_warmup_steps=training.get("throughput_warmup_steps", 3),
        gradient_clip_norm=training.get("gradient_clip_norm", 1.0),
    )


def _parse_evaluation_config(raw: dict[str, Any]) -> EvaluationConfig:
    """Parse the evaluation section."""
    evaluation = raw.get("evaluation", {})
    return EvaluationConfig(
        batch_size=evaluation.get("batch_size", 8),
        max_new_tokens=evaluation.get("max_new_tokens", 512),
        do_sample=evaluation.get("do_sample", False),
        max_examples=evaluation.get("max_examples"),
        split=evaluation.get("split", "test"),
    )


def _parse_output_config(raw: dict[str, Any]) -> OutputConfig:
    """Parse the output section."""
    output = raw.get("output", {})
    return OutputConfig(
        results_dir=_resolve_path(output.get("results_dir", "results")),
        checkpoints_dir=_resolve_path(output.get("checkpoints_dir", "checkpoints")),
        save_final_checkpoint=output.get("save_final_checkpoint", True),
    )


def load_experiment_config(path: Path) -> ExperimentConfig:
    """
    Load and validate an experiment configuration from a YAML file.

    Responsibilities:
    1. open YAML safely
    2. reject empty or non-mapping documents
    3. parse every section
    4. apply only documented defaults
    5. construct typed dataclasses
    6. call validate_experiment_config
    7. return an immutable configuration
    """
    raw = _safe_load_yaml(path)

    config = ExperimentConfig(
        experiment=_parse_experiment_config(raw),
        model=_parse_model_config(raw),
        data=_parse_data_config(raw),
        method=_parse_method_config(raw),
        training=_parse_training_config(raw),
        evaluation=_parse_evaluation_config(raw),
        output=_parse_output_config(raw),
    )

    validate_experiment_config(config)

    return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    """
    Validate an experiment configuration.

    Must validate:
    - experiment.name is nonempty
    - experiment.sweep is nonempty
    - model.name is nonempty
    - model.revision is nonempty
    - data.revision is nonempty
    - micro_batch_size > 0
    - effective_batch_size > 0
    - gradient_accumulation_steps > 0
    - effective_batch_size = micro_batch_size × gradient_accumulation_steps
    - learning_rate > 0
    - weight_decay ≥ 0
    - num_workers ≥ 0
    - throughput_warmup_steps ≥ 0
    - evaluation.batch_size > 0
    - evaluation.max_new_tokens > 0
    - evaluation.split is 'validation' or 'test'

    Stopping conditions:
    - at least one of max_steps or training_token_budget must be set

    Method conditions:
    - full_ft must not require rank or alpha
    - lora requires positive rank and alpha
    - 0 ≤ dropout < 1
    - target_modules must be nonempty

    Precision conditions:
    - invalid precision names fail during configuration loading
    """
    errors: list[str] = []

    # Experiment identity
    if not config.experiment.name:
        errors.append("experiment.name must be nonempty")
    if not config.experiment.sweep:
        errors.append("experiment.sweep must be nonempty")

    # Model
    if not config.model.name:
        errors.append("model.name must be nonempty")
    if not config.model.revision:
        errors.append("model.revision must be nonempty")

    # Data
    if not config.data.dataset_revision:
        errors.append("data.revision must be nonempty")

    # Training
    t = config.training
    if t.max_steps is not None and t.max_steps <= 0:
        errors.append(
            "max_steps must be > 0 when set"
        )
    if (
        t.training_token_budget is not None
        and t.training_token_budget <= 0
    ):
        errors.append(
            "training_token_budget must be > 0 when set"
        )
    if (
    t.max_steps is not None
        and t.throughput_warmup_steps >= t.max_steps
    ):
        errors.append(
            "throughput_warmup_steps must be smaller "
            "than max_steps"
        )
    if t.micro_batch_size <= 0:
        errors.append("micro_batch_size must be > 0")
    if t.effective_batch_size <= 0:
        errors.append("effective_batch_size must be > 0")
    if t.gradient_accumulation_steps <= 0:
        errors.append("gradient_accumulation_steps must be > 0")

    if t.micro_batch_size > 0 and t.gradient_accumulation_steps > 0:
        expected = t.micro_batch_size * t.gradient_accumulation_steps
        if t.effective_batch_size != expected:
            errors.append(
                f"effective_batch_size ({t.effective_batch_size}) must equal "
                f"micro_batch_size ({t.micro_batch_size}) × "
                f"gradient_accumulation_steps ({t.gradient_accumulation_steps}) = {expected}"
            )

    if t.learning_rate <= 0:
        errors.append("learning_rate must be > 0")
    if t.weight_decay < 0:
        errors.append("weight_decay must be >= 0")
    if t.num_workers < 0:
        errors.append("num_workers must be >= 0")
    if t.throughput_warmup_steps < 0:
        errors.append("throughput_warmup_steps must be >= 0")

    if t.gradient_clip_norm is not None and t.gradient_clip_norm <= 0:
        errors.append("gradient_clip_norm must be > 0")

    # Stopping conditions
    if t.max_steps is None and t.training_token_budget is None:
        errors.append(
            "At least one of max_steps or training_token_budget must be set"
        )

    # Evaluation
    e = config.evaluation
    if e.batch_size <= 0:
        errors.append("evaluation.batch_size must be > 0")
    if e.max_new_tokens <= 0:
        errors.append("evaluation.max_new_tokens must be > 0")
    if e.split not in ("validation", "test"):
        errors.append(
            "evaluation.split must be 'validation' or 'test', "
            f"got '{e.split}'"
        )
    if e.max_examples is not None and e.max_examples <= 0:
        errors.append(
            "evaluation.max_examples must be > 0 when set"
        )

    # Precision
    if t.precision not in ("bf16", "fp32"):
        errors.append(f"precision must be 'bf16' or 'fp32', got '{t.precision}'")

    # Method conditions
    m = config.method
    if m.name not in ("full_ft", "lora"):
        errors.append(f"method.name must be 'full_ft' or 'lora', got '{m.name}'")

    if m.name == "full_ft":
        if m.rank is not None:
            errors.append("full_ft must not specify rank")
        if m.alpha is not None:
            errors.append("full_ft must not specify alpha")

    if m.name == "lora":
        if m.rank is None or m.rank <= 0:
            errors.append(f"lora requires positive rank, got {m.rank}")
        if m.alpha is None or m.alpha <= 0:
            errors.append(f"lora requires positive alpha, got {m.alpha}")
        if m.dropout is None or not (0 <= m.dropout < 1):
            errors.append(f"lora dropout must be in [0, 1), got {m.dropout}")

    if not m.target_modules:
        errors.append("target_modules must be nonempty")

    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    if (
        config.training.gradient_clip_norm is not None
        and config.training.gradient_clip_norm <= 0
    ):
        raise ValueError(
            "training.gradient_clip_norm must be positive "
            "when provided"
        )


# ============================================================================
# Serialization utilities
# ============================================================================


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses and Paths to JSON/YAML-safe objects."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, set)):
        return [_dataclass_to_dict(item) for item in obj]
    return obj


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """
    Convert nested dataclasses and paths into YAML/JSON-safe objects.

    Preserves resolved defaults and supports resolved_config.yaml and run metadata.
    """
    return _dataclass_to_dict(config)


def write_resolved_config(config: ExperimentConfig, path: Path) -> None:
    """
    Write the fully resolved configuration to a YAML file.

    Responsibilities:
    - create parent directories
    - write the fully resolved configuration
    - use atomic replacement
    - never overwrite a different completed run silently
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config_to_dict(config)

    # Atomic write using temporary file
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        suffix=".yaml.tmp",
        prefix=".resolved_",
    )
    try:
        with open(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        # Atomic rename
        Path(tmp_path).rename(path)
    except Exception:
        # Clean up temp file on failure
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise
