'''
Training configuration module.

This module is the only module allowed to parse YAML.
'''

from .config import (
    ExperimentConfig,
    ExperimentIdentityConfig,
    ModelConfig,
    MethodConfig,
    TrainingConfig,
    EvaluationConfig,
    OutputConfig,
    load_experiment_config,
    validate_experiment_config,
    config_to_dict,
    write_resolved_config,
)
from .setup import (
    ModelBundle,
    DataBundle,
    TrainingComponents,
    set_global_seed,
    resolve_device,
    resolve_dtype,
    load_tokenizer,
    load_base_model,
    build_model_bundle,
    build_data_bundle,
    build_training_components,
)
from .optim import build_optimizer
from .engine import TrainerEngine
from .results import (
    StopReason,
    TrainingResult,
    EvaluationResult,
    ExperimentResult,
    experiment_result_to_dict,
    validate_experiment_result,
    write_json_atomic,
)
from .checkpoint import (
    CheckpointInfo,
    directory_size_bytes,
    save_final_checkpoint,
    reload_final_checkpoint,
    verify_checkpoint_reload,
)
from .experiment import (
    RunPaths,
    build_run_id,
    build_run_paths,
    collect_run_metadata,
    evaluate_model,
    ExperimentRunner,
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
