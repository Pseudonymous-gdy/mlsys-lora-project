'''
Training configuration module.

This module is the only module allowed to parse YAML.
'''

from .checkpoint import (
    CheckpointInfo,
    directory_size_bytes,
    reload_final_checkpoint,
    save_final_checkpoint,
    verify_checkpoint_reload,
)
from .config import (
    EvaluationConfig,
    ExperimentConfig,
    ExperimentIdentityConfig,
    MethodConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
    config_to_dict,
    load_experiment_config,
    validate_experiment_config,
    write_resolved_config,
)
from .engine import TrainerEngine
from .experiment import (
    ExperimentRunner,
    RunPaths,
    build_run_id,
    build_run_paths,
    collect_run_metadata,
    evaluate_model,
)
from .optim import build_optimizer
from .results import (
    EvaluationResult,
    ExperimentResult,
    StopReason,
    TrainingResult,
    experiment_result_to_dict,
    validate_experiment_result,
    write_json_atomic,
)
from .setup import (
    DataBundle,
    ModelBundle,
    TrainingComponents,
    build_data_bundle,
    build_model_bundle,
    build_training_components,
    load_base_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
    set_global_seed,
)

__all__ = [
    # Config
    "ExperimentConfig",
    "ExperimentIdentityConfig",
    "ModelConfig",
    "MethodConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "OutputConfig",
    "load_experiment_config",
    "validate_experiment_config",
    "config_to_dict",
    "write_resolved_config",
    # Setup
    "ModelBundle",
    "DataBundle",
    "TrainingComponents",
    "set_global_seed",
    "resolve_device",
    "resolve_dtype",
    "load_tokenizer",
    "load_base_model",
    "build_model_bundle",
    "build_data_bundle",
    "build_training_components",
    # Optimizer
    "build_optimizer",
    # Engine
    "TrainerEngine",
    # Results
    "StopReason",
    "TrainingResult",
    "EvaluationResult",
    "ExperimentResult",
    "experiment_result_to_dict",
    "validate_experiment_result",
    "write_json_atomic",
    # Checkpoint
    "CheckpointInfo",
    "directory_size_bytes",
    "save_final_checkpoint",
    "reload_final_checkpoint",
    "verify_checkpoint_reload",
    # Experiment
    "RunPaths",
    "build_run_id",
    "build_run_paths",
    "collect_run_metadata",
    "evaluate_model",
    "ExperimentRunner",
]
