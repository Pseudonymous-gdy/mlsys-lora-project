#!/bin/bash
#SBATCH --job-name=gsm8k_data
#SBATCH --partition=i64m512u
#SBATCH --output=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project/logs/gsm8k_data_%j.out
#SBATCH --error=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project/logs/gsm8k_data_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module purge
module load slurm

export PROJECT_ENV=/hpc2hdd/home/dsaa4012_002/.conda/envs/project
export MLSYS_ROOT=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project
export MLSYS_DATA=/hpc2hdd/home/dsaa4012_002/jhaidata/mlsys-lora-project

export PATH="$PROJECT_ENV/bin:$PATH"
export PYTHONPATH="$MLSYS_ROOT/src:${PYTHONPATH:-}"

export HF_HOME="$MLSYS_DATA/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TMPDIR="$MLSYS_DATA/tmp"

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HF_HUB_CACHE" \
  "$TMPDIR" \
  "$MLSYS_DATA/processed"

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
import datasets
import torch
import transformers

print("torch:", torch.__version__)
print("datasets:", datasets.__version__)
print("transformers:", transformers.__version__)
PY

echo "===== GSM8K unit tests ====="
python -m pytest -q tests/test_gsm8k.py

echo "===== Real GSM8K + Qwen tokenizer smoke ====="
python scripts/smoke_data.py \
  --model Qwen/Qwen3.5-0.8B \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --samples 50 \
  --max-length 512 \
  --seed 42

echo "===== DATA_SMOKE_PASS ====="