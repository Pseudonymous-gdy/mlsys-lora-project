#!/bin/bash
# ============================================================================
# Smoke test submission script
#
# Submits LoRA and Full FT smoke tests sequentially.
# Full FT is only submitted after LoRA smoke test is accepted.
# ============================================================================

set -e

PROJECT_ROOT=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project
SLURM_SCRIPT="$PROJECT_ROOT/slurm/train_one.sbatch"

echo "===== Submitting Smoke Tests ====="
echo "Date: $(date)"
echo "Project: $PROJECT_ROOT"

# ============================================================================
# 1. Submit LoRA-16 smoke test
# ============================================================================

echo ""
echo "----- Submitting LoRA-16 smoke test -----"
LORA_CONFIG="$PROJECT_ROOT/configs/generated/smoke_lora.yaml"

if [[ ! -f "$LORA_CONFIG" ]]; then
  echo "ERROR: LoRA smoke config not found: $LORA_CONFIG"
  exit 1
fi

LORA_JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT" "$LORA_CONFIG")
echo "LoRA smoke test submitted with job ID: $LORA_JOB_ID"

# ============================================================================
# 2. Wait for LoRA job to be accepted
# ============================================================================

echo "Waiting for LoRA job to be accepted..."
sleep 5

# Check if job is in the queue
if squeue -j "$LORA_JOB_ID" &>/dev/null; then
  echo "LoRA job $LORA_JOB_ID is in the queue."
else
  echo "WARNING: LoRA job $LORA_JOB_ID may have already completed or failed."
fi

# ============================================================================
# 3. Submit Full FT smoke test
# ============================================================================

echo ""
echo "----- Submitting Full FT smoke test -----"
FULLFT_CONFIG="$PROJECT_ROOT/configs/generated/smoke_full_ft.yaml"

if [[ ! -f "$FULLFT_CONFIG" ]]; then
  echo "ERROR: Full FT smoke config not found: $FULLFT_CONFIG"
  exit 1
fi

FULLFT_JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT" "$FULLFT_CONFIG")
echo "Full FT smoke test submitted with job ID: $FULLFT_JOB_ID"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "===== Smoke Test Summary ====="
echo "LoRA job ID:    $LORA_JOB_ID"
echo "Full FT job ID: $FULLFT_JOB_ID"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo ""
echo "Check logs after completion:"
echo "  logs/train_one_${LORA_JOB_ID}.out"
echo "  logs/train_one_${LORA_JOB_ID}.err"
echo "  logs/train_one_${FULLFT_JOB_ID}.out"
echo "  logs/train_one_${FULLFT_JOB_ID}.err"
echo ""
echo "Check results after completion:"
echo "  results/smoke_lora/"
echo "  results/smoke_full_ft/"
echo ""
echo "Expected artifacts per run:"
echo "  resolved_config.yaml"
echo "  metadata.json"
echo "  predictions.jsonl"
echo "  result.json"
echo "  checkpoints/<run_id>/final/"
