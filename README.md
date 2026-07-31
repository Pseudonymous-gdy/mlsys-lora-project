# MLSys LoRA Project

Team: Daoyuan GUO, Zhaokai LIANG

Purpose
 
Compare LoRA adaptation with full fine-tuning and measure system-level
trade-offs (peak GPU memory, training throughput, epoch time, checkpoint
size) while keeping experiments feasible on HPC resources.

Large artifacts (logs, results, etc.) are not tracked
in this repository.

Quick Links

- Code: repository root
- Configs: `configs/` and `configs/generated/`
- Results: `results/`
- Docs: `docs/`

Prerequisites

- Python 3.10+ (see `requirements.txt`)
- A cluster environment with SLURM for full training runs

Quickstart

Clone the repo and run tests locally:

```bash
git clone https://github.com/Pseudonymous-gdy/mlsys-lora-project.git
cd mlsys-lora-project
# Run unit & integration tests
python -m pytest -q
```

Run generated experiment configs on an HPC cluster (example):

```bash
PROJECT_ENV=/path/to/python/env \
INCLUDE_REGEX='regex-to-include' \
EXCLUDE_REGEX='regex-to-exclude' \
GPUS_PER_JOB=1 \
TIME_LIMIT="04:00:00" \
bash slurm/run_generated_configs.sbatch
```

The SLURM script discovers `configs/generated/` automatically; it does not
assume fixed filenames.

Resuming or re-running experiments

Re-run the same command to resume incomplete runs. Default behavior:

- `completed` -> skip
- `OOM` -> skip
- `failed` -> retry
- `partial` -> retry
- `missing` -> run

Force rerun all generated configurations:

```bash
FORCE_RERUN=1 \
GPUS_PER_JOB=4 \
PROJECT_ENV=/path/to/python/env \
bash slurm/run_generated_configs.sbatch
```

Selecting a subset (e.g., LR sweep):

```bash
INCLUDE_REGEX="hyperparameter_tuning" \
GPUS_PER_JOB=1 \
PROJECT_ENV=/path/to/python/env \
bash slurm/run_generated_configs.sbatch
```

Analysis and utilities

Rebuild the learning-rate selection tables from stored results:

```bash
PYTHONPATH=src python -m analysis.tuning_table results \
  --markdown reports/learning_rate_selection.md \
  --json reports/learning_rate_selection.json
```

List all generated result files:

```bash
find results -mindepth 2 -maxdepth 2 -name result.json -print | sort
```

Print summary metrics for every run:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("results").glob("*/result.json")):
    result = json.loads(path.read_text(encoding="utf-8"))
    print(
        path.parent.name,
        "status=", result.get("status"),
        "peak_memory_gb=", result.get("peak_memory_gb"),
        "peak_reserved_memory_gb=", result.get("peak_reserved_memory_gb"),
        "tokens_per_second=", result.get("tokens_per_second"),
        "training_time_seconds=", result.get("training_time_seconds"),
    )
PY
```

Local development workflow

The CPU-side pipeline can be developed and validated on macOS or Linux. Do
not install CUDA-specific cluster requirements on macOS. Example local
workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt

# Unit and integration tests (no model weights required)
python -m pytest -q
ruff check src tests scripts

# Small smoke test using a real tokenizer (adjust HF cache path if needed)
HF_HOME="$HOME/.cache/mlsys-lora-project/huggingface" \
  python scripts/smoke_data.py --samples 50

# Materialize per-run experiment configurations
python scripts/generate_sweep_configs.py
```

Notes

- Local runs prepare data, score existing model completions, generate
  experiment configs, and aggregate results. They do not perform full FT or
  LoRA training with GPU measurements; those require the cluster.
- Dataset and model revisions are pinned in `configs/base.yaml` to ensure
  reproducibility across local and cluster runs.

Documentation

See [docs/experiment_protocol.md](docs/experiment_protocol.md) for the data
contract, fairness controls, result schema, and handoff procedure.

Contributing

- Develop on feature branches and open a Pull Request for review.
- Do not commit datasets, model weights, checkpoints, caches, logs,
  secrets, or local environment files.

Requirements

Install runtime dependencies listed in `requirements.txt`.

License

This repository contains project code and configs. Add license information
here if desired.

