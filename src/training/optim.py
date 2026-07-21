'''
Optimizer construction module.

Builds AdamW optimizer for trainable parameters.
'''

from __future__ import annotations

from typing import Any

import torch

from methods.common import (
    assert_has_trainable_parameters,
    assert_optimizer_matches_trainable_parameters,
)
from training.config import ExperimentConfig


def build_optimizer(model: Any, config: ExperimentConfig) -> torch.optim.Optimizer:
    """
    Build the AdamW optimizer for trainable parameters.

    Responsibilities:
    - collect only parameters with requires_grad=True
    - reject an empty trainable set
    - construct torch.optim.AdamW
    - use configured learning rate and weight decay
    - avoid adding frozen parameters
    - call assert_optimizer_matches_trainable_parameters

    The first implementation does not add a learning-rate scheduler.
    """
    # Collect only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Reject empty trainable set
    if not trainable_params:
        raise ValueError(
            "Cannot build optimizer: no trainable parameters found. "
            "Ensure the model has parameters with requires_grad=True."
        )

    # Build AdamW optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Validate optimizer parameter coverage
    assert_optimizer_matches_trainable_parameters(model, optimizer)

    return optimizer
