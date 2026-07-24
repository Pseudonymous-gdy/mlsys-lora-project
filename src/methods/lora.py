'''
This module injects and validates LoRA adapters.
'''

from dataclasses import dataclass

from .common import (
    ParameterStats,
    assert_has_trainable_parameters,
    count_parameters,
    get_trainable_parameter_names,
)


@dataclass
class LoraMethodConfig:
    '''The canonical method configuration is defined in training.config, 
    but this module may import it for type checking.'''
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    bias: str

def resolve_lora_target_modules(model, requested_modules) -> tuple[str, ...]:
    '''
    Responsibilities:

    - verify that every configured target-module suffix exists in the model;
    - return the validated target-module tuple;
    - reject an empty configuration;
    - reject silently ignored target modules;
    - include available matching module names in the error message.

    The model architecture must not be guessed at runtime without recording the choice.

    The initial YAML should explicitly define:

    method:
        target_modules:
            - q_proj
            - k_proj
            - v_proj
            - o_proj
    
    Additional MLP targets may be introduced only as a documented experimental change.

    Args:
        model: The model to analyze.
        requested_modules: A tuple of target-module suffixes to validate.
    
    Returns:
        tuple[str, ...]: A tuple of validated target-module suffixes.
    '''
    if not requested_modules:
        raise ValueError("LoRA target_modules configuration is empty. Please specify at least one target module.")

    # Collect all module names in the model
    all_module_names = {name for name, _ in model.named_modules()}

    # Find which requested modules actually exist as suffixes
    validated = []
    for requested in requested_modules:
        matches = [name for name in all_module_names if name.endswith(requested)]
        if not matches:
            available = sorted({name.split('.')[-1] for name in all_module_names})
            raise ValueError(
                f"LoRA target module '{requested}' not found in model. "
                f"Available module suffixes include: {available[:20]}..."
            )
        validated.append(requested)

    if not validated:
        raise ValueError(
            f"No requested LoRA target modules found in model. "
            f"Requested: {requested_modules}"
        )

    return tuple(validated)


def validate_lora_trainability(model) -> None:
    '''
    Responsibilities:

    - ensure at least one trainable parameter exists;
    - ensure trainable parameters are adapter parameters;
    - ensure base-model parameters remain frozen;
    - raise a descriptive error containing unexpected trainable names.

    This validation must run before optimizer construction.

    Args:
        model: The PEFT-wrapped model to validate.
    '''
    assert_has_trainable_parameters(model)

    trainable_names = get_trainable_parameter_names(model)

    # Check that trainable parameters are adapter parameters
    unexpected_trainable = [
        name for name in trainable_names
        if 'lora' not in name.lower() and 'adapter' not in name.lower()
    ]

    if unexpected_trainable:
        raise ValueError(
            f"Found non-adapter trainable parameters: {unexpected_trainable[:5]}... "
            "Base model parameters should be frozen when using LoRA."
        )


def configure_lora(model, config) -> tuple[object, ParameterStats]:
    '''
    Responsibilities:

    1. validate rank, alpha, dropout, and targets;
    2. create peft.LoraConfig;
    3. call get_peft_model;
    4. validate trainability;
    5. return the wrapped model and parameter statistics.

    The initial LoRA configuration must use:
    task_type = CAUSAL_LM
    inference_mode = false

    Args:
        model: The base model to wrap with LoRA.
        config: LoraMethodConfig with LoRA parameters.

    Returns:
        tuple[object, ParameterStats]: The PEFT-wrapped model and parameter statistics.
    '''
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA experiments. Install with: pip install peft"
        ) from e

    # Validate configuration
    if config.rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {config.rank}")
    if config.alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {config.alpha}")
    if not (0 <= config.dropout < 1):
        raise ValueError(f"LoRA dropout must be in [0, 1), got {config.dropout}")
    if not config.target_modules:
        raise ValueError("LoRA target_modules must be non-empty")

    # Validate target modules exist in model
    validated_targets = resolve_lora_target_modules(model, config.target_modules)

    # Create LoRA configuration
    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(validated_targets),
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    # Apply LoRA to model
    peft_model = get_peft_model(model, lora_config)

    # Validate that LoRA was applied correctly
    validate_lora_trainability(peft_model)

    # Count parameters after LoRA injection
    param_stats = count_parameters(peft_model)

    return peft_model, param_stats
    
    
