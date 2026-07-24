'''
This module exposes the method-selection API for training configuration.
'''

from .common import ParameterStats
from .full_ft import configure_full_finetuning
from .lora import LoraMethodConfig, configure_lora


def configure_training_method(model, config) -> tuple[object, ParameterStats]:
    '''
    Responsibilities:

    - dispatch on config.method.name;
    - call configure_full_finetuning for full_ft;
    - call configure_lora for lora;
    - reject unknown method names;
    - return a consistently shaped result.

    No other module should manually branch between Full FT and LoRA setup.

    Args:
        model: The base model to configure.
        config: ExperimentConfig with method configuration.

    Returns:
        tuple[object, ParameterStats]: The configured model and parameter statistics.
    '''
    method_name = config.method.name

    if method_name == "full_ft":
        param_stats = configure_full_finetuning(model)
        return model, param_stats
    elif method_name == "lora":
        lora_config = LoraMethodConfig(
            rank=config.method.rank,
            alpha=config.method.alpha,
            dropout=config.method.dropout,
            target_modules=config.method.target_modules,
            bias=config.method.bias,
        )
        return configure_lora(model, lora_config)
    else:
        raise ValueError(
            f"Unknown training method: '{method_name}'. "
            "Supported methods: 'full_ft', 'lora'"
        )


__all__ = [
    "ParameterStats",
    "LoraMethodConfig",
    "configure_training_method",
    "configure_full_finetuning",
    "configure_lora",
]