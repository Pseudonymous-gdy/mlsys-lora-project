#!/bin/bash
#SBATCH --job-name=train_one
#SBATCH --partition=debug
#SBATCH --output=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project/logs/train_one_%j.out
#SBATCH --error=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project/logs/train_one_%j.err
#SBATCH --time=00:29:59
#SBATCH --gres=gpu:a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

module purge
module load slurm
module load cuda/12.8

export PROJECT_ROOT=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project
export DATA_ROOT=/hpc2hdd/home/dsaa4012_002/jhaidata/mlsys-lora-project
export PROJECT_ENV=/hpc2hdd/home/dsaa4012_002/.conda/envs/project

export PATH="$PROJECT_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

export HF_HOME="$DATA_ROOT/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TMPDIR="$DATA_ROOT/tmp"

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HF_HUB_CACHE" \
  "$TMPDIR" \
  "$PROJECT_ROOT/results" \
  "$PROJECT_ROOT/checkpoints"

cd "$PROJECT_ROOT"

if [[ $# -lt 1 ]]; then
  echo \
    "Usage: sbatch train_one.sbatch CONFIG_PATH [ALLOW_OVERWRITE]" \
    >&2
  exit 1
fi

CONFIG_PATH="$1"
ALLOW_OVERWRITE="${2:-0}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config does not exist: $CONFIG_PATH" >&2
  exit 1
fi

case "$ALLOW_OVERWRITE" in
  0|1)
    ;;
  *)
    echo \
      "ERROR: ALLOW_OVERWRITE must be 0 or 1, got: $ALLOW_OVERWRITE" \
      >&2
    exit 1
    ;;
esac

RUN_ID="$(
  python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import sys
from pathlib import Path

from training.config import load_experiment_config
from training.experiment import build_run_paths

config_path = Path(sys.argv[1])
repository_root = Path(sys.argv[2])

config = load_experiment_config(config_path)
paths = build_run_paths(config, repository_root)

print(paths.run_id)
PY
)"

echo "===== Hostname ====="
hostname

echo "===== Date ====="
date



echo "===== Environment ====="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "PROJECT_ENV=$PROJECT_ENV"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "RUN_ID=$RUN_ID"
echo "ALLOW_OVERWRITE=$ALLOW_OVERWRITE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

which python
python --version
python -m pip -V

echo "===== CUDA ====="
nvidia-smi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable inside the training allocation."
    )

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"Expected one visible GPU, "
        f"found {torch.cuda.device_count()}."
    )

if not torch.cuda.is_bf16_supported():
    raise RuntimeError(
        "BF16 is not supported by the selected GPU."
    )

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory GiB:",
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3,
)
PY

train_args=(
  --config "$CONFIG_PATH"
  --repository-root "$PROJECT_ROOT"
)

if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
  train_args+=(--allow-overwrite)
fi

echo "===== Starting Experiment ====="
echo "Command:"
printf ' %q' \
  python \
  "$PROJECT_ROOT/scripts/train.py" \
  "${train_args[@]}"
echo

set +e
python "$PROJECT_ROOT/scripts/train.py" \
  "${train_args[@]}"
exit_code=$?
set -e

echo "===== Experiment Finished ====="
echo "RUN_ID=$RUN_ID"
echo "Exit code=$exit_code"
echo "Result path=$PROJECT_ROOT/results/$RUN_ID/result.json"
date

exit "$exit_code"