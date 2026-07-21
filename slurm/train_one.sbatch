#!/bin/bash
#SBATCH --job-name=train_one
#SBATCH --partition=a40
#SBATCH --output=logs/train_one_%j.out
#SBATCH --error=logs/train_one_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e

# ============================================================================
# Module setup
# ============================================================================

module purge
module load slurm
module load cuda/12.8

# ============================================================================
# Environment setup
# ============================================================================

export PROJECT_ROOT=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project
export PROJECT_ENV=/hpc2hdd/home/dsaa4012_002/.conda/envs/project

export PATH="$PROJECT_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/src"

export HF_HOME="$PROJECT_ROOT/data/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TMPDIR="$PROJECT_ROOT/data/tmp"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_ROOT"

# ============================================================================
# System information
# ============================================================================

echo "===== Hostname ====="
hostname

echo "===== Date ====="
date

echo "===== Git Info ====="
git log --oneline -1 || echo "Not a git repository"
git status --short || echo "Git status unavailable"

echo "===== Environment ====="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PROJECT_ENV=$PROJECT_ENV"
echo "PATH=$PATH"
which python
python --version

echo "===== CUDA ====="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi unavailable"

echo "===== GPU Info ====="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
" || echo "PyTorch CUDA unavailable"

# ============================================================================
# Run experiment
# ============================================================================

echo "===== Starting Experiment ====="
echo "Config: $1"

python "$PROJECT_ROOT/scripts/train.py" \
  --config "$1" \
  --repository-root "$PROJECT_ROOT"

exit_code=$?

echo "===== Experiment Finished ====="
echo "Exit code: $exit_code"

exit $exit_code
