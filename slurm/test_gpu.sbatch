#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=debug
#SBATCH --output=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project/logs/test_gpu_%j.out
#SBATCH --error=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project/logs/test_gpu_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

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

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HF_HUB_CACHE" \
  "$TMPDIR"

cd "$PROJECT_ROOT"

echo "===== Hostname ====="
hostname
date

echo "===== Git ====="
git status --short --branch
git rev-parse HEAD

echo "===== Environment ====="
echo "PROJECT_ENV=$PROJECT_ENV"
echo "PATH=$PATH"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

which python
python --version
python -m pip -V

echo "===== nvidia-smi ====="
nvidia-smi

echo "===== Torch CUDA/BF16 Test ====="
python - <<'PY'
import sys

import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available inside the Slurm allocation."
    )

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"Expected exactly one visible GPU, "
        f"found {torch.cuda.device_count()}."
    )

device = torch.device("cuda")
properties = torch.cuda.get_device_properties(device)

print("GPU:", torch.cuda.get_device_name(device))
print(
    "GPU memory GiB:",
    properties.total_memory / 1024**3,
)
print(
    "BF16 supported:",
    torch.cuda.is_bf16_supported(),
)

if not torch.cuda.is_bf16_supported():
    raise RuntimeError(
        "The selected GPU does not support BF16."
    )

x = torch.randn(
    1024,
    1024,
    device=device,
    dtype=torch.bfloat16,
)
y = x @ x

torch.cuda.synchronize(device)

print("BF16 matmul shape:", tuple(y.shape))
print(
    "max memory allocated MiB:",
    torch.cuda.max_memory_allocated(device)
    / 1024**2,
)

print("GPU_SMOKE_PASS")
PY