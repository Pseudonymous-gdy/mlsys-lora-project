#!/bin/bash
#SBATCH --job-name=gsm8k_data
#SBATCH --partition=i64m512u
#SBATCH --output=logs/gsm8k_data_%j.out
#SBATCH --error=logs/gsm8k_data_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module purge
module load slurm

export PROJECT_ENV=/hpc2hdd/home/dsaa4012_002/.conda/envs/project
export PATH="$PROJECT_ENV/bin:$PATH"

export MLSYS_ROOT=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project
export MLSYS_DATA=/hpc2hdd/home/dsaa4012_002/jhaidata/mlsys-lora-project

export HF_HOME="$MLSYS_DATA/hf_cache"
export HF_DATASETS_CACHE="$MLSYS_DATA/hf_cache/datasets"
export TRANSFORMERS_CACHE="$MLSYS_DATA/hf_cache"
export TMPDIR="$MLSYS_DATA/tmp"

# 继续使用此前 HPC 上设置的 Hugging Face 镜像。
export HF_ENDPOINT=https://hf-mirror.com

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$MLSYS_ROOT/src:${PYTHONPATH:-}"

mkdir -p \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$TMPDIR" \
    "$MLSYS_DATA/processed" \
    "$MLSYS_ROOT/logs"

cd "$MLSYS_ROOT"

echo "===== Host ====="
hostname
date

echo "===== Git ====="
git status --short --branch
git rev-parse HEAD

echo "===== Python ====="
which python
python --version
python -m pip -V

echo "===== Package versions ====="
python - <<'PY'
import torch
import datasets
import transformers

print("torch:", torch.__version__)
print("datasets:", datasets.__version__)
print("transformers:", transformers.__version__)
PY

echo "===== GSM8K unit tests ====="
python -m pytest -q tests/test_gsm8k.py

echo "===== Real GSM8K + Qwen tokenizer smoke test ====="
python scripts/smoke_data.py \
    --model Qwen/Qwen3.5-0.8B \
    --samples 50 \
    --max-length 512 \
    --seed 42

echo "===== DATA_SMOKE_PASS ====="
