#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project
SLURM_SCRIPT="$PROJECT_ROOT/slurm/train_one.sbatch"

LORA_CONFIG="$PROJECT_ROOT/configs/generated/smoke_lora.yaml"
FULLFT_CONFIG="$PROJECT_ROOT/configs/generated/smoke_full_ft.yaml"

LORA_RUN_ID=smoke_lora_lora_r16_l512_mb1_seed42
FULLFT_RUN_ID=smoke_full_ft_full_ft_l512_mb1_seed42

ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

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

cd "$PROJECT_ROOT"

mkdir -p \
  "$PROJECT_ROOT/logs" \
  "$PROJECT_ROOT/results" \
  "$PROJECT_ROOT/checkpoints"

for required_file in \
  "$SLURM_SCRIPT" \
  "$LORA_CONFIG" \
  "$FULLFT_CONFIG"
do
  if [[ ! -f "$required_file" ]]; then
    echo "ERROR: Missing file: $required_file" >&2
    exit 1
  fi
done

branch="$(
  git branch --show-current
)"

if [[ "$branch" != "dguo" ]]; then
  echo \
    "ERROR: Expected branch dguo, found: $branch" \
    >&2
  exit 1
fi

if ! git diff --quiet; then
  echo "ERROR: Working tree has tracked modifications." >&2
  git diff --stat >&2
  exit 1
fi

if ! git diff --cached --quiet; then
  echo "ERROR: Staging area has uncommitted changes." >&2
  git diff --cached --stat >&2
  exit 1
fi

local_head="$(
  git rev-parse HEAD
)"
remote_head="$(
  git rev-parse origin/dguo
)"

if [[ "$local_head" != "$remote_head" ]]; then
  echo "ERROR: Local HEAD differs from origin/dguo." >&2
  echo "Local:  $local_head" >&2
  echo "Remote: $remote_head" >&2
  exit 1
fi

echo "===== Smoke Submission Preflight ====="
date
echo "Branch: $branch"
echo "Commit: $local_head"
echo "Allow overwrite: $ALLOW_OVERWRITE"

sha256sum \
  "$LORA_CONFIG" \
  "$FULLFT_CONFIG"

echo
echo "===== Submitting LoRA Smoke ====="

lora_job_raw="$(
  sbatch \
    --parsable \
    "$SLURM_SCRIPT" \
    "$LORA_CONFIG" \
    "$ALLOW_OVERWRITE"
)"

LORA_JOB_ID="${lora_job_raw%%;*}"

echo "LoRA job ID: $LORA_JOB_ID"

echo
echo "===== Submitting Full FT Smoke ====="

fullft_job_raw="$(
  sbatch \
    --parsable \
    --dependency="afterok:${LORA_JOB_ID}" \
    "$SLURM_SCRIPT" \
    "$FULLFT_CONFIG" \
    "$ALLOW_OVERWRITE"
)"

FULLFT_JOB_ID="${fullft_job_raw%%;*}"

echo "Full FT job ID: $FULLFT_JOB_ID"
echo "Dependency: afterok:$LORA_JOB_ID"

echo
echo "===== Smoke Submission Summary ====="
echo "LoRA:"
echo "  job: $LORA_JOB_ID"
echo "  result: results/$LORA_RUN_ID/result.json"
echo "  checkpoint: checkpoints/$LORA_RUN_ID/final/"
echo
echo "Full FT:"
echo "  job: $FULLFT_JOB_ID"
echo "  result: results/$FULLFT_RUN_ID/result.json"
echo "  checkpoint: checkpoints/$FULLFT_RUN_ID/final/"
echo
echo "Monitor:"
echo "  squeue -j ${LORA_JOB_ID},${FULLFT_JOB_ID}"
echo
echo "Accounting:"
echo "  sacct -j ${LORA_JOB_ID},${FULLFT_JOB_ID} \\"
echo "    --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES"
echo
echo "Logs:"
echo "  logs/train_one_${LORA_JOB_ID}.out"
echo "  logs/train_one_${LORA_JOB_ID}.err"
echo "  logs/train_one_${FULLFT_JOB_ID}.out"
echo "  logs/train_one_${FULLFT_JOB_ID}.err"