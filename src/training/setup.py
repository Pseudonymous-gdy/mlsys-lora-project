'''
Training setup module.

Builds model, data, and optimizer components from validated configuration.
'''

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from methods import configure_training_method
from methods.common import ParameterStats, assert_optimizer_matches_trainable_parameters
from training.config import ExperimentConfig
from training.optim import build_optimizer

# ============================================================================
# Bundle dataclasses
# ============================================================================


@dataclass
class ModelBundle:
    """Bundles the model, tokenizer, and related metadata."""
    model: Any
    tokenizer: Any
    parameter_stats: ParameterStats
    resolved_model_name: str
    resolved_model_revision: str
    dtype_name: str
    attention_backend: str


@dataclass
class DataBundle:
    """Bundles data loaders and dataset metadata."""
    train_loader: DataLoader
    validation_loader: DataLoader
    validation_dataset: Any
    test_dataset: Any
    train_examples: int
    validation_examples: int
    test_examples: int


@dataclass
class TrainingComponents:
    """All components required for training."""
    model_bundle: ModelBundle
    data_bundle: DataBundle
    optimizer: torch.optim.Optimizer
    device: torch.device


# ============================================================================
# Seed and device resolution
# ============================================================================


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility.

    Responsibilities:
    - seed Python
    - seed NumPy
    - seed PyTorch CPU
    - seed all CUDA devices
    - configure deterministic behavior where practical
    - not promise bit-for-bit determinism for unsupported CUDA kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    """
    Resolve the training device.

    Responsibilities:
    - require one CUDA device for real experiments
    - return the current CUDA device
    - produce a clear error when launched outside a GPU allocation

    Unit tests may bypass this function with explicit CPU components.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available. "
            "Training requires a GPU allocation. "
            "Use explicit CPU components for unit tests only."
        )

    device_count = torch.cuda.device_count()
    if device_count != 1:
        raise RuntimeError(
            f"Expected exactly 1 CUDA device, found {device_count}. "
            "This implementation supports single-GPU training only."
        )

    return torch.device("cuda")


def resolve_dtype(config: ExperimentConfig, device: torch.device) -> torch.dtype:
    """
    Resolve the training dtype from configuration.

    Responsibilities:
    - map bf16 to torch.bfloat16
    - map fp32 to torch.float32
    - verify BF16 support on the selected CUDA device
    - raise a descriptive compatibility error
    """
    precision = config.training.precision

    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"BF16 precision requested but not supported on {torch.cuda.get_device_name(device)}. "
                "Use 'fp32' precision or a GPU with BF16 support (Ampere or later)."
            )
        return torch.bfloat16

    if precision == "fp32":
        return torch.float32

    raise ValueError(
        f"Unknown precision: '{precision}'. Must be 'bf16' or 'fp32'."
    )


# ============================================================================
# Tokenizer and model loading
# ============================================================================


def load_tokenizer(config: ExperimentConfig) -> Any:
    """
    Load the tokenizer with pinned revision.

    Responsibilities:
    - call AutoTokenizer.from_pretrained
    - pass both model name and pinned revision
    - pass trust_remote_code
    - ensure a valid padding token
    - use EOS as pad token only when pad token is absent
    - return the tokenizer without modifying chat templates
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )

    # Ensure a valid padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(
                f"Tokenizer for '{config.model.name}' has no pad_token or eos_token. "
                "A valid padding token is required for training."
            )

    return tokenizer


def load_base_model(
    config: ExperimentConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Any:
    """
    Load the base model with pinned revision.

    Responsibilities:
    - call AutoModelForCausalLM.from_pretrained
    - pass model name and pinned revision
    - pass the selected dtype
    - pass the selected attention implementation
    when configured
    - move the model to the single CUDA device
    - leave runtime forward options to the training loop
    - return an unmodified base model

    The fallback model must not be selected silently.
    """
    from transformers import AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "trust_remote_code": (
            config.model.trust_remote_code
        ),
        "dtype": dtype,
    }

    if config.model.attention_backend is not None:
        model_kwargs["attn_implementation"] = config.model.attention_backend

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        **model_kwargs,
    )

    # Move model to the selected device
    model = model.to(device)

    return model


# ============================================================================
# Bundle construction
# ============================================================================


def build_model_bundle(config: ExperimentConfig, device: torch.device) -> ModelBundle:
    """
    Build the complete model bundle.

    Responsibilities:
    1. resolve dtype
    2. load tokenizer
    3. load base model
    4. configure Full FT or LoRA through configure_training_method
    5. validate parameter statistics
    6. return ModelBundle
    """
    dtype = resolve_dtype(config, device)
    tokenizer = load_tokenizer(config)
    base_model = load_base_model(config, dtype, device)

    # Configure training method (Full FT or LoRA)
    configured_model, parameter_stats = configure_training_method(base_model, config)

    # Determine resolved model name and revision
    resolved_name = config.model.name
    resolved_revision = config.model.revision

    # Determine dtype name
    dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float32"

    # Determine attention backend
    attention_backend = config.model.attention_backend

    return ModelBundle(
        model=configured_model,
        tokenizer=tokenizer,
        parameter_stats=parameter_stats,
        resolved_model_name=resolved_name,
        resolved_model_revision=resolved_revision,
        dtype_name=dtype_name,
        attention_backend=attention_backend or "default",
    )


def build_data_bundle(config: ExperimentConfig, tokenizer: Any) -> DataBundle:
    """
    Build the complete data bundle.

    Responsibilities:
    1. call the existing prepare_gsm8k_datasets
    2. construct the existing CausalLMCollator
    3. create train and validation DataLoaders
    4. expose the validation and formatted test datasets
    5. record split sizes

    Train DataLoader:
        shuffle = true
        batch size = configured micro-batch
        drop_last = false

    Validation DataLoader:
        shuffle = false
        drop_last = false

    Both use the same collator.
    """
    from data.gsm8k import (
        CausalLMCollator,
        prepare_gsm8k_datasets,
    )

    # Build GSM8K data config from experiment config
    data_config = config.data

    # Prepare datasets
    datasets = prepare_gsm8k_datasets(data_config, tokenizer)

    # Construct collator
    pad_token_id = tokenizer.pad_token_id
    pad_to_multiple_of = 8  # Default padding multiple for GPU efficiency
    collator = CausalLMCollator(
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.training.micro_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=collator,
    )

    test_dataset = datasets["test"]

    return DataBundle(
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_dataset=datasets["validation"],
        test_dataset=test_dataset,
        train_examples=len(datasets["train"]),
        validation_examples=len(datasets["validation"]),
        test_examples=len(datasets["test"]),
    )


def build_training_components(config: ExperimentConfig) -> TrainingComponents:
    """
    Build all training components from configuration.

    Responsibilities:
    1. set the global seed
    2. resolve the CUDA device
    3. construct ModelBundle
    4. construct DataBundle
    5. construct the optimizer
    6. validate optimizer parameter coverage
    7. return all components
    """
    # Set global seed first
    set_global_seed(config.training.seed)

    # Resolve device
    device = resolve_device()

    # Build model bundle (includes method configuration)
    model_bundle = build_model_bundle(config, device)

    # Build data bundle
    data_bundle = build_data_bundle(config, model_bundle.tokenizer)

    # Build optimizer
    optimizer = build_optimizer(model_bundle.model, config)

    # Validate optimizer parameter coverage
    assert_optimizer_matches_trainable_parameters(model_bundle.model, optimizer)

    return TrainingComponents(
        model_bundle=model_bundle,
        data_bundle=data_bundle,
        optimizer=optimizer,
        device=device,
    )
