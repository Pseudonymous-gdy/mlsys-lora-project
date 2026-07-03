#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=debug
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e

module purge
module load slurm
module load cuda/12.8

export PROJECT_ENV=/hpc2hdd/home/dsaa4012_002/.conda/envs/project
export PATH=$PROJECT_ENV/bin:$PATH

export MLSYS_ROOT=$HOME/mlsys-lora-project
export MLSYS_DATA=$HOME/jhaidata/mlsys-lora-project
export HF_HOME=$HOME/jhaidata/mlsys-lora-project/hf_cache
export TRANSFORMERS_CACHE=$HOME/jhaidata/mlsys-lora-project/hf_cache
export HF_DATASETS_CACHE=$HOME/jhaidata/mlsys-lora-project/hf_cache/datasets
export TMPDIR=$HOME/jhaidata/mlsys-lora-project/tmp
export HF_ENDPOINT=https://hf-mirror.com

cd $MLSYS_ROOT

echo "===== Hostname ====="
hostname

echo "===== Environment ====="
echo "PROJECT_ENV=$PROJECT_ENV"
echo "PATH=$PATH"
which python
python --version
python -m pip -V

echo "===== CUDA_VISIBLE_DEVICES ====="
echo $CUDA_VISIBLE_DEVICES

echo "===== nvidia-smi ====="
nvidia-smi

echo "===== Torch CUDA Test ====="
python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside Slurm GPU job.")

print("gpu:", torch.cuda.get_device_name(0))

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("matmul ok:", y.shape)
print("max memory allocated MB:", torch.cuda.max_memory_allocated() / 1024 / 1024)
PY
