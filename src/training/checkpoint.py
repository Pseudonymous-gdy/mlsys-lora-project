'''
Checkpoint save, reload, and verification.

Handles saving final inference artifacts (Full FT or LoRA),
reloading them, and verifying reload compatibility.
'''

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from training.config import ExperimentConfig


# ============================================================================
# Checkpoint dataclasses
# ============================================================================


@dataclass(frozen=True)
class CheckpointInfo:
    """Immutable container for checkpoint metadata."""
    path: Path
    method: str
    size_bytes: int
    size_mb: float
    reload_verified: bool


# ============================================================================
# Directory size
# ============================================================================


def directory_size_bytes(path: Path) -> int:
    """
    Recursively sum regular file sizes in a directory.

    Responsibilities:
    - recursively sum regular file sizes
    - do not follow unrelated symlinks
    - reject missing directories
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {path}")

    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            if filepath.is_file() and not filepath.is_symlink():
                total += filepath.stat().st_size

    return total


# ============================================================================
# Save checkpoint
# ============================================================================


def save_final_checkpoint(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    path: Path,
) -> CheckpointInfo:
    """
    Save the final inference artifact atomically.

    Responsibilities:
    - create temporary sibling directory
    - save appropriate artifacts (Full FT or LoRA)
    - save tokenizer metadata
    - atomically rename to final path
    - compute directory size
    - return initial CheckpointInfo with reload_verified=False

    Full FT behavior:
        model.save_pretrained(...)
        tokenizer.save_pretrained(...)

    LoRA behavior:
        peft_model.save_pretrained(...)
        tokenizer.save_pretrained(...)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary sibling directory
    tmp_dir = tempfile.mkdtemp(
        dir=path.parent,
        prefix=".checkpoint_",
    )
    tmp_path = Path(tmp_dir)

    try:
        # Save model (handles both Full FT and LoRA via model.save_pretrained)
        model.save_pretrained(str(tmp_path))

        # Save tokenizer
        tokenizer.save_pretrained(str(tmp_path))

        # Atomically rename to final path
        if path.exists():
            shutil.rmtree(path)
        tmp_path.rename(path)

        # Compute directory size
        size_bytes = directory_size_bytes(path)
        size_mb = size_bytes / (1024 * 1024)

        return CheckpointInfo(
            path=path,
            method=config.method.name,
            size_bytes=size_bytes,
            size_mb=size_mb,
            reload_verified=False,
        )

    except Exception:
        # Clean up temporary directory on failure
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise


# ============================================================================
# Reload checkpoint
# ============================================================================


def reload_final_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> PreTrainedModel:
    """
    Reload a saved checkpoint for inference.

    Full FT responsibilities:
    - load saved full model
    - use configured dtype
    - move to selected device

    LoRA responsibilities:
    1. load pinned base model
    2. load saved PEFT adapter
    3. return combined model
    """
    checkpoint_path = Path(checkpoint_path)
    method = config.method.name

    # Resolve dtype
    if config.training.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if method == "full_ft":
        # Load full model directly
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            torch_dtype=dtype,
            trust_remote_code=config.model.trust_remote_code,
        )
        model = model.to(device)

    elif method == "lora":
        # Import PEFT for LoRA reload
        from peft import PeftModel

        # Load base model with pinned revision
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            revision=config.model.revision,
            torch_dtype=dtype,
            trust_remote_code=config.model.trust_remote_code,
        )
        base_model = base_model.to(device)

        # Load PEFT adapter from checkpoint
        model = PeftModel.from_pretrained(
            base_model,
            str(checkpoint_path),
        )

    else:
        raise ValueError(f"Unknown method for checkpoint reload: {method}")

    return model


# ============================================================================
# Verify reload
# ============================================================================


def verify_checkpoint_reload(
    config: ExperimentConfig,
    checkpoint_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> PreTrainedModel:
    """
    Reload final artifact and verify minimal generation compatibility.

    Responsibilities:
    - reload final artifact
    - set evaluation mode
    - verify minimal generation-compatible forward path can be prepared
    - return reloaded model for evaluation
    """
    model = reload_final_checkpoint(config, checkpoint_path, device)
    model.eval()

    # Verify minimal forward path with a dummy input
    dummy_input = tokenizer(
        "Verify checkpoint reload.",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        _ = model(**dummy_input, use_cache=False)

    return model
