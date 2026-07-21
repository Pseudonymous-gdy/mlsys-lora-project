#!/usr/bin/env python3
"""
Train script - lean entry point for running experiments.

Usage:
    python scripts/train.py --config configs/generated/smoke_lora_r16.yaml
    python scripts/train.py --config configs/generated/main_lora.yaml --allow-overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single MLSys LoRA experiment")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect)",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting completed results (development only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Auto-detect repository root if not provided
    repository_root = args.repository_root or Path(__file__).resolve().parent.parent

    # Import training modules
    from src.training.config import load_experiment_config
    from src.training.experiment import ExperimentRunner

    # Load and validate config
    config = load_experiment_config(args.config)

    # Create runner and execute
    runner = ExperimentRunner(config=config, repository_root=repository_root)

    try:
        result = runner.run()
    except ValueError as e:
        if "Completed result already exists" in str(e) and not args.allow_overwrite:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        raise

    # Print result JSON to stdout
    from src.training.results import experiment_result_to_dict
    print(json.dumps(experiment_result_to_dict(result), indent=2))

    # Return appropriate exit code
    if result.status == "completed":
        return 0
    elif result.status == "oom":
        return 2
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
