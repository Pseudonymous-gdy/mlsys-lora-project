'''
This module is responsible only for configuring and validating Full Fine-tuning.
'''

from .common import ParameterStats, assert_has_trainable_parameters, count_parameters


def configure_full_finetuning(model) -> ParameterStats:
    '''
    Responsibilities:

    - set requires_grad=True for all model parameters;
    - return final parameter statistics;
    - perform no optimizer construction;
    - perform no training.

    Args:
        model: The model to configure for full fine-tuning.
    
    Returns:
        ParameterStats: A dataclass containing parameter counts and trainable 
        fraction.
    '''
    for param in model.parameters():
        param.requires_grad = True
    
    validate_full_finetuning(model)

    return count_parameters(model)

def validate_full_finetuning(model) -> None:
    '''
    Responsibilities:

    - assert that all expected model parameters are trainable;
    - raise a descriptive error listing the first frozen parameters found;
    - call assert_has_trainable_parameters.

    Args:
        model: The model to validate for full fine-tuning.
    '''
    for param in model.parameters():
        if not param.requires_grad:
            raise AssertionError("Found frozen parameter: {}".format(param))

    assert_has_trainable_parameters(model)