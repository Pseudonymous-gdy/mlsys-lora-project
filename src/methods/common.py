'''
This module defines common model-parameter utilities used by Full FT,
LoRA, tests, and result reporting.

'''

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterStats:
    """Immutable container for model parameter statistics."""
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_fraction: float

    def __post_init__(self):
        if self.trainable_parameters + self.frozen_parameters != self.total_parameters:
            raise ValueError(
                f"Parameter counts do not add up: "
                f"trainable({self.trainable_parameters}) + "
                f"frozen({self.frozen_parameters}) != "
                f"total({self.total_parameters})"
            )
        if not (0.0 <= self.trainable_fraction <= 1.0):
            raise ValueError(
                f"Trainable fraction must be between 0 and 1, "
                f"got {self.trainable_fraction}"
            )

def count_parameters(model) -> ParameterStats:
    """
    Responsibility:
    - iterate through model.parameters()
    - count all parameter elements
    - count parameters where requires_grad=True

    Args:
        model: The model to analyze.

    Returns:
        ParameterStats: A dataclass containing parameter counts and trainable fraction.
    """
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_parameters = total_parameters - trainable_parameters
    trainable_fraction = trainable_parameters / total_parameters if total_parameters > 0 else 0.0

    return ParameterStats(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        frozen_parameters=frozen_parameters,
        trainable_fraction=trainable_fraction
    )

def get_trainable_parameter_names(model) -> tuple[str, ...]:
    '''
    Responsibilities:

    - return the names of all trainable parameters;
    - preserve model traversal order;
    - support diagnostic logs and LoRA validation.

    Args:
        model: The model to analyze.
    
    Returns:
        tuple[str, ...]: A tuple containing the names of all trainable parameters.
    '''
    return tuple(name for name, param in model.named_parameters() if param.requires_grad)

def assert_has_trainable_parameters(model) -> None:
    '''
    Responsibilities:

    - raise ValueError if no parameters require gradients;
    - prevent optimizer construction with an empty parameter list.

    Args:
        model: The model to analyze.
    '''
    if not any(param.requires_grad for param in model.parameters()):
        raise ValueError("The model has no trainable parameters (requires_grad=True).")

def assert_optimizer_matches_trainable_parameters(model, optimizer) -> None:
    '''
    Responsibilities:

    - collect parameter identities in optimizer parameter groups;
    - collect parameter identities where requires_grad=True;
    - raise an error if a trainable parameter is missing;
    - raise an error if a frozen parameter was accidentally added.

    Args:
        model: The model to analyze.
        optimizer: The optimizer to check against the model's trainable parameters.
    '''
    # Collect parameter identities in optimizer parameter groups
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group['params']}

    # Collect parameter identities where requires_grad=True
    trainable_param_ids = {id(p) for p in model.parameters() if p.requires_grad}

    # Raise an error if a trainable parameter is missing
    if not trainable_param_ids.issubset(optimizer_param_ids):
        raise ValueError("The optimizer is missing some trainable parameters.")

    # Raise an error if a frozen parameter was accidentally added
    if not optimizer_param_ids.issubset(trainable_param_ids):
        raise ValueError("The optimizer contains frozen parameters that should not be optimized.")

