#!/usr/bin/env bash
#
# Dynamically run every valid YAML config currently present under
# configs/generated/.
#
# IMPORTANT:
# - Run this file with bash, not directly with sbatch.
# - The submission mode requests 1, 2, or 4 GPUs in one Slurm allocation.
# - Each config remains a single-GPU training job.
# - Multiple configs are packed concurrently into the allocation.
#
# Examples:
#
#   # Use one A40
#   GPUS_PER_JOB=1 \
#   bash slurm/run_generated_configs.sbatch
#
#   # Use two A40 GPUs
#   GPUS_PER_JOB=2 \
#   bash slurm/run_generated_configs.sbatch
#
#   # Use four A40 GPUs
#   GPUS_PER_JOB=4 \
#   bash slurm/run_generated_configs.sbatch
#
#   # Only configs whose relative path matches "lora|rank"
#   INCLUDE_REGEX='lora|rank' \
#   GPUS_PER_JOB=4 \
#   bash slurm/run_generated_configs.sbatch
#
#   # Exclude smoke configs
#   EXCLUDE_REGEX='smoke' \
#   GPUS_PER_JOB=4 \
#   bash slurm/run_generated_configs.sbatch
#
#   # Force rerun everything
#   FORCE_RERUN=1 \
#   GPUS_PER_JOB=4 \
#   bash slurm/run_generated_configs.sbatch
#
#   # Dry-run manifest generation and submission command
#   DRY_RUN=1 \
#   GPUS_PER_JOB=4 \
#   bash slurm/run_generated_configs.sbatch
#
#SBATCH --job-name=mlsys_generated
#SBATCH --output=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project/logs/generated_%j.out
#SBATCH --error=/hpc2hdd/home/dsaa4012_002/mlsys-lora-project/logs/generated_%j.err

set -euo pipefail


# ============================================================================
# Paths
# ============================================================================

SCRIPT_PATH="$(
    readlink -f "${BASH_SOURCE[0]}"
)"

SCRIPT_DIR="$(
    dirname "$SCRIPT_PATH"
)"

DEFAULT_PROJECT_ROOT="$(
    cd "$SCRIPT_DIR/.."
    pwd -P
)"

export PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"

export CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/configs/generated}"

export DATA_ROOT="${DATA_ROOT:-\
/hpc2hdd/home/dsaa4012_002/jhaidata/mlsys-lora-project}"

export PROJECT_ENV="${PROJECT_ENV:-\
/hpc2hdd/home/dsaa4012_002/.conda/envs/project}"


# ============================================================================
# Slurm resource settings
# ============================================================================

export PARTITION="${PARTITION:-emergency_gpua40}"
export GPU_TYPE="${GPU_TYPE:-a40}"

# Allowed values are deliberately restricted to the requested modes.
export GPUS_PER_JOB="${GPUS_PER_JOB:-4}"

export CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
export MEM_GB_PER_GPU="${MEM_GB_PER_GPU:-48}"

# One packed job may run many configs. Increase this when the local generated
# set is large. The A40 partition currently permits up to seven days.
export TIME_LIMIT="${TIME_LIMIT:-1-00:30:00}"

export CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"


# ============================================================================
# Config discovery
# ============================================================================

# Regex applied to each path relative to CONFIG_DIR.
# Empty means include everything.
export INCLUDE_REGEX="${INCLUDE_REGEX:-}"
export EXCLUDE_REGEX="${EXCLUDE_REGEX:-}"

# Search recursively below configs/generated.
export RECURSIVE_CONFIG_SEARCH="${RECURSIVE_CONFIG_SEARCH:-1}"


# ============================================================================
# Existing-result behavior
# ============================================================================

# completed result:
#   1 = skip
#   0 = overwrite and rerun
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

# oom result:
#   1 = keep it as an experimental outcome and skip
#   0 = overwrite and rerun
export SKIP_OOM="${SKIP_OOM:-1}"

# failed result:
#   1 = overwrite and retry
#   0 = retain and skip
export RETRY_FAILED="${RETRY_FAILED:-1}"

# Overrides all skip/retry settings.
export FORCE_RERUN="${FORCE_RERUN:-0}"

# OOM is normally a valid result for capacity sweeps.
# Set to 1 when any OOM should fail the packed Slurm job.
export FAIL_ON_OOM="${FAIL_ON_OOM:-0}"


# ============================================================================
# Reproducibility and submission behavior
# ============================================================================

export DRY_RUN="${DRY_RUN:-0}"


# ============================================================================
# Utility functions
# ============================================================================

fail() {
    echo "ERROR: $*" >&2
    exit 1
}


require_boolean() {
    local name="$1"
    local value="$2"

    case "$value" in
        0|1)
            ;;
        *)
            fail "$name must be 0 or 1, got: $value"
            ;;
    esac
}


require_positive_integer() {
    local name="$1"
    local value="$2"

    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        fail "$name must be a positive integer, got: $value"
    fi
}


setup_python_environment() {
    export PATH="$PROJECT_ENV/bin:$PATH"
    export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

    export HF_HOME="$DATA_ROOT/hf_cache"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
    export TMPDIR="$DATA_ROOT/tmp"

    export TOKENIZERS_PARALLELISM=false
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

    mkdir -p \
        "$PROJECT_ROOT/logs" \
        "$PROJECT_ROOT/results" \
        "$PROJECT_ROOT/checkpoints" \
        "$HF_HOME" \
        "$HF_DATASETS_CACHE" \
        "$HF_HUB_CACHE" \
        "$TMPDIR"

    cd "$PROJECT_ROOT"
}


require_internal_real_directory() {
    local directory="$1"
    local resolved

    if [[ -L "$directory" ]]; then
        fail "$directory must not be a symbolic link"
    fi

    mkdir -p "$directory"

    resolved="$(
        readlink -f "$directory"
    )"

    case "$resolved" in
        "$PROJECT_ROOT"/*)
            ;;
        *)
            fail \
                "$directory resolves outside PROJECT_ROOT: $resolved"
            ;;
    esac
}


validate_settings() {
    case "$GPUS_PER_JOB" in
        1|2|4)
            ;;
        *)
            fail \
                "GPUS_PER_JOB must be 1, 2, or 4; got $GPUS_PER_JOB"
            ;;
    esac

    require_positive_integer \
        "CPUS_PER_GPU" \
        "$CPUS_PER_GPU"

    require_positive_integer \
        "MEM_GB_PER_GPU" \
        "$MEM_GB_PER_GPU"

    require_boolean \
        "RECURSIVE_CONFIG_SEARCH" \
        "$RECURSIVE_CONFIG_SEARCH"

    require_boolean \
        "SKIP_COMPLETED" \
        "$SKIP_COMPLETED"

    require_boolean \
        "SKIP_OOM" \
        "$SKIP_OOM"

    require_boolean \
        "RETRY_FAILED" \
        "$RETRY_FAILED"

    require_boolean \
        "FORCE_RERUN" \
        "$FORCE_RERUN"

    require_boolean \
        "FAIL_ON_OOM" \
        "$FAIL_ON_OOM"

    require_boolean \
        "DRY_RUN" \
        "$DRY_RUN"
}


# ============================================================================
# Submission mode
# ============================================================================

generate_config_manifest() {
    local snapshot_directory="$1"

    python - \
        "$CONFIG_DIR" \
        "$PROJECT_ROOT" \
        "$snapshot_directory" \
        "$INCLUDE_REGEX" \
        "$EXCLUDE_REGEX" \
        "$RECURSIVE_CONFIG_SEARCH" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

from training.config import load_experiment_config
from training.experiment import build_run_paths


config_dir = Path(sys.argv[1]).resolve()
repository_root = Path(sys.argv[2]).resolve()
snapshot_dir = Path(sys.argv[3]).resolve()

include_pattern = sys.argv[4]
exclude_pattern = sys.argv[5]
recursive = sys.argv[6] == "1"


if not config_dir.is_dir():
    raise RuntimeError(
        f"Config directory does not exist: {config_dir}"
    )


def discover_configs() -> list[Path]:
    if recursive:
        candidates = [
            *config_dir.rglob("*.yaml"),
            *config_dir.rglob("*.yml"),
        ]
    else:
        candidates = [
            *config_dir.glob("*.yaml"),
            *config_dir.glob("*.yml"),
        ]

    unique = {
        path.resolve()
        for path in candidates
        if path.is_file()
    }

    return sorted(
        unique,
        key=lambda path: str(path.relative_to(config_dir)),
    )


def selected(path: Path) -> bool:
    relative = str(path.relative_to(config_dir))

    if (
        include_pattern
        and re.search(include_pattern, relative) is None
    ):
        return False

    if (
        exclude_pattern
        and re.search(exclude_pattern, relative) is not None
    ):
        return False

    return True


config_paths = [
    path
    for path in discover_configs()
    if selected(path)
]

if not config_paths:
    raise RuntimeError(
        "No generated config matched the current selection."
    )


snapshot_config_dir = snapshot_dir / "configs"
snapshot_config_dir.mkdir(
    parents=True,
    exist_ok=True,
)

entries: list[dict[str, object]] = []
seen_run_ids: dict[str, str] = {}


for index, source_path in enumerate(config_paths):
    relative_path = source_path.relative_to(config_dir)

    config = load_experiment_config(source_path)
    run_paths = build_run_paths(
        config,
        repository_root,
    )

    run_id = run_paths.run_id

    if run_id in seen_run_ids:
        raise RuntimeError(
            "Two generated configs produce the same run_id:\n"
            f"  run_id: {run_id}\n"
            f"  first:  {seen_run_ids[run_id]}\n"
            f"  second: {relative_path}"
        )

    seen_run_ids[run_id] = str(relative_path)

    safe_name = str(relative_path).replace("/", "__")

    snapshot_path = (
        snapshot_config_dir
        / f"{index:04d}_{safe_name}"
    )

    shutil.copy2(
        source_path,
        snapshot_path,
    )

    digest = hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()

    entries.append(
        {
            "index": index,
            "source_path": str(source_path),
            "relative_path": str(relative_path),
            "snapshot_path": str(snapshot_path),
            "sha256": digest,
            "run_id": run_id,
        }
    )


manifest_path = snapshot_dir / "manifest.json"

manifest_path.write_text(
    json.dumps(
        {
            "config_dir": str(config_dir),
            "repository_root": str(repository_root),
            "count": len(entries),
            "entries": entries,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


summary_path = snapshot_dir / "manifest.txt"

summary_lines = [
    f"config_dir={config_dir}",
    f"count={len(entries)}",
    "",
]

for entry in entries:
    summary_lines.append(
        f"{entry['index']:04d}\t"
        f"{entry['run_id']}\t"
        f"{entry['relative_path']}\t"
        f"{entry['sha256']}"
    )

summary_lines.append("")

summary_path.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)


print(manifest_path)
print(f"CONFIG_COUNT={len(entries)}", file=sys.stderr)
PY
}


submission_mode() {
    validate_settings

    module purge
    module load slurm
    module load "$CUDA_MODULE"

    setup_python_environment

    require_internal_real_directory \
        "$PROJECT_ROOT/results"

    require_internal_real_directory \
        "$PROJECT_ROOT/checkpoints"



    local timestamp
    timestamp="$(
        date +%Y%m%d_%H%M%S
    )"

    export SNAPSHOT_DIRECTORY="$PROJECT_ROOT/logs/config_manifests/${timestamp}"

    mkdir -p "$SNAPSHOT_DIRECTORY"

    echo "===== Config Discovery ====="
    echo "Project root:      $PROJECT_ROOT"
    echo "Config directory:  $CONFIG_DIR"
    echo "Include regex:     ${INCLUDE_REGEX:-<none>}"
    echo "Exclude regex:     ${EXCLUDE_REGEX:-<none>}"
    echo "Recursive search:  $RECURSIVE_CONFIG_SEARCH"

    export CONFIG_MANIFEST="$(
        generate_config_manifest \
            "$SNAPSHOT_DIRECTORY"
    )"

    export CONFIG_MANIFEST_SHA256="$(
        sha256sum "$CONFIG_MANIFEST" |
            awk '{print $1}'
    )"

    echo
    cat "$SNAPSHOT_DIRECTORY/manifest.txt"

    local total_cpus
    local total_memory_gb

    total_cpus=$((CPUS_PER_GPU * GPUS_PER_JOB))
    total_memory_gb=$((MEM_GB_PER_GPU * GPUS_PER_JOB))

    export MLSYS_GENERATED_MODE=worker

    local output_path
    local error_path

    output_path="$PROJECT_ROOT/logs/generated_%j.out"
    error_path="$PROJECT_ROOT/logs/generated_%j.err"

    local -a submission=(
        sbatch
        --parsable
        --job-name="mlsys_generated"
        --partition="$PARTITION"
        --nodes=1
        --gres="gpu:${GPU_TYPE}:${GPUS_PER_JOB}"
        --cpus-per-task="$total_cpus"
        --mem="${total_memory_gb}G"
        --time="$TIME_LIMIT"
        --output="$output_path"
        --error="$error_path"
        --export=ALL
        "$SCRIPT_PATH"
    )

    echo
    echo "===== Slurm Submission ====="
    echo "Partition:       $PARTITION"
    echo "GPU request:     gpu:${GPU_TYPE}:${GPUS_PER_JOB}"
    echo "Parallel configs: $GPUS_PER_JOB"
    echo "CPUs:            $total_cpus"
    echo "Memory:          ${total_memory_gb}G"
    echo "Time limit:      $TIME_LIMIT"

    printf ' %q' "${submission[@]}"
    echo

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY_RUN_COMPLETE"
        return 0
    fi

    local raw_job_id
    local job_id

    raw_job_id="$(
        "${submission[@]}"
    )"

    job_id="${raw_job_id%%;*}"

    echo
    echo "JOB_ID=$job_id"
    echo "Manifest=$CONFIG_MANIFEST"
    echo
    echo "Monitor:"
    echo "  squeue -j $job_id"
    echo
    echo "Accounting:"
    echo "  sacct -j $job_id \\"
    echo "    --format=JobID,JobName,Partition,State,ExitCode,Elapsed,AllocTRES"
    echo
    echo "Main logs:"
    echo "  logs/generated_${job_id}.out"
    echo "  logs/generated_${job_id}.err"
}


# ============================================================================
# Worker mode
# ============================================================================

worker_mode() {
    validate_settings

    module purge
    module load slurm
    module load "$CUDA_MODULE"

    setup_python_environment

    [[ -f "${CONFIG_MANIFEST:-}" ]] ||
        fail "CONFIG_MANIFEST is missing"

    [[ -n "${CONFIG_MANIFEST_SHA256:-}" ]] ||
        fail "CONFIG_MANIFEST_SHA256 is missing"

    local actual_manifest_sha
    actual_manifest_sha="$(
        sha256sum "$CONFIG_MANIFEST" |
            awk '{print $1}'
    )"

    [[ "$actual_manifest_sha" == "$CONFIG_MANIFEST_SHA256" ]] ||
        fail \
            "Config manifest changed after submission"

    require_internal_real_directory \
        "$PROJECT_ROOT/results"

    require_internal_real_directory \
        "$PROJECT_ROOT/checkpoints"

    export PACKED_LOG_DIRECTORY="$PROJECT_ROOT/logs/generated_runs/"\
"${SLURM_JOB_ID}"

    mkdir -p "$PACKED_LOG_DIRECTORY"

    echo "===== Packed Worker ====="
    echo "Date:             $(date --iso-8601=seconds)"
    echo "Host:             $(hostname)"
    echo "Job ID:           $SLURM_JOB_ID"
    echo "Manifest:         $CONFIG_MANIFEST"
    echo "GPUs requested:   $GPUS_PER_JOB"
    echo "GPU type:         $GPU_TYPE"
    echo "CPUs per GPU:     $CPUS_PER_GPU"
    echo "CUDA visible:     ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "Per-config logs:  $PACKED_LOG_DIRECTORY"

    nvidia-smi \
        --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader

    python - <<'PY'
import os

import torch

expected = int(os.environ["GPUS_PER_JOB"])
actual = torch.cuda.device_count()

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("visible GPU count:", actual)

if actual != expected:
    raise RuntimeError(
        "Visible GPU count does not match allocation: "
        f"expected {expected}, found {actual}"
    )

for index in range(actual):
    properties = torch.cuda.get_device_properties(index)

    print(
        f"GPU {index}: "
        f"{properties.name}, "
        f"{properties.total_memory / 1024**3:.2f} GiB"
    )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "The allocated GPU does not support BF16."
        )
PY

    python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

repository_root = Path(
    os.environ["PROJECT_ROOT"]
).resolve()

manifest_path = Path(
    os.environ["CONFIG_MANIFEST"]
).resolve()

log_directory = Path(
    os.environ["PACKED_LOG_DIRECTORY"]
).resolve()

gpu_type = os.environ["GPU_TYPE"]
gpu_count = int(os.environ["GPUS_PER_JOB"])
cpus_per_gpu = int(os.environ["CPUS_PER_GPU"])

skip_completed = os.environ["SKIP_COMPLETED"] == "1"
skip_oom = os.environ["SKIP_OOM"] == "1"
retry_failed = os.environ["RETRY_FAILED"] == "1"
force_rerun = os.environ["FORCE_RERUN"] == "1"
fail_on_oom = os.environ["FAIL_ON_OOM"] == "1"

train_script = (
    repository_root / "scripts/train.py"
)

summary_path = (
    log_directory / "summary.json"
)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

manifest = json.loads(
    manifest_path.read_text(encoding="utf-8")
)

entries: list[dict[str, Any]] = manifest["entries"]

if not entries:
    raise RuntimeError(
        "Config manifest contains no entries."
    )


for entry in entries:
    snapshot_path = Path(entry["snapshot_path"])

    if not snapshot_path.is_file():
        raise RuntimeError(
            f"Missing config snapshot: {snapshot_path}"
        )

    actual_digest = hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()

    if actual_digest != entry["sha256"]:
        raise RuntimeError(
            "Config snapshot changed after submission: "
            f"{snapshot_path}"
        )


# ---------------------------------------------------------------------------
# Result handling
# ---------------------------------------------------------------------------

def read_result_status(
    run_id: str,
) -> str:
    result_path = (
        repository_root
        / "results"
        / run_id
        / "result.json"
    )

    if not result_path.exists():
        return "missing"

    try:
        data = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return "invalid"

    status = data.get("status")

    if status in {
        "completed",
        "oom",
        "failed",
    }:
        return str(status)

    return "invalid"


def choose_action(
    run_id: str,
) -> tuple[str, bool]:
    """
    Return:
        (action, allow_overwrite)

    action:
        run
        skip_completed
        skip_oom
        skip_failed
    """
    status = read_result_status(run_id)

    result_directory = (
        repository_root
        / "results"
        / run_id
    )

    checkpoint_directory = (
        repository_root
        / "checkpoints"
        / run_id
    )

    partial = (
        result_directory.exists()
        or checkpoint_directory.exists()
    )

    if force_rerun:
        return "run", True

    if status == "completed":
        if skip_completed:
            return "skip_completed", False

        return "run", True

    if status == "oom":
        if skip_oom:
            return "skip_oom", False

        return "run", True

    if status == "failed":
        if retry_failed:
            return "run", True

        return "skip_failed", False

    if status == "invalid":
        return "run", True

    if status == "missing" and partial:
        return "run", True

    return "run", False


# ---------------------------------------------------------------------------
# Parallel scheduler
# ---------------------------------------------------------------------------

task_queue: queue.Queue[dict[str, Any]] = (
    queue.Queue()
)

for entry in entries:
    task_queue.put(entry)


results: list[dict[str, Any]] = []
results_lock = threading.Lock()
print_lock = threading.Lock()


def safe_filename(
    value: str,
) -> str:
    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )


def synchronized_print(
    message: str,
) -> None:
    with print_lock:
        print(
            message,
            flush=True,
        )


def record_result(
    record: dict[str, Any],
) -> None:
    with results_lock:
        results.append(record)

        summary_path.write_text(
            json.dumps(
                {
                    "job_id": os.environ.get(
                        "SLURM_JOB_ID"
                    ),
                    "total_configs": len(entries),
                    "records": sorted(
                        results,
                        key=lambda item: item["index"],
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def run_config(
    worker_slot: int,
    entry: dict[str, Any],
) -> None:
    index = int(entry["index"])
    run_id = str(entry["run_id"])
    config_path = Path(
        entry["snapshot_path"]
    )

    action, allow_overwrite = (
        choose_action(run_id)
    )

    if action != "run":
        synchronized_print(
            f"[slot={worker_slot}] "
            f"[{index}] {run_id}: {action}"
        )

        record_result(
            {
                "index": index,
                "run_id": run_id,
                "source_config": entry[
                    "relative_path"
                ],
                "action": action,
                "outcome": action,
                "return_code": None,
                "final_status": (
                    read_result_status(run_id)
                ),
                "duration_seconds": 0.0,
            }
        )

        return

    filename_prefix = (
        f"{index:04d}_"
        f"{safe_filename(run_id)}"
    )

    stdout_path = (
        log_directory
        / f"{filename_prefix}.out"
    )

    stderr_path = (
        log_directory
        / f"{filename_prefix}.err"
    )

    command = [
        "srun",
        "--exclusive",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus_per_gpu}",
        f"--gres=gpu:{gpu_type}:1",
        "env",
        "PYTHONUNBUFFERED=1",
        f"OMP_NUM_THREADS={cpus_per_gpu}",
        sys.executable,
        str(train_script),
        "--config",
        str(config_path),
        "--repository-root",
        str(repository_root),
    ]

    if allow_overwrite:
        command.append(
            "--allow-overwrite"
        )

    synchronized_print(
        f"[slot={worker_slot}] "
        f"[{index}] START {run_id}\n"
        f"  config={entry['relative_path']}\n"
        f"  overwrite={allow_overwrite}\n"
        f"  stdout={stdout_path}\n"
        f"  stderr={stderr_path}"
    )

    started = time.monotonic()

    with (
        stdout_path.open(
            "w",
            encoding="utf-8",
        ) as stdout_handle,
        stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr_handle,
    ):
        completed = subprocess.run(
            command,
            cwd=repository_root,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            env=os.environ.copy(),
        )

    duration = time.monotonic() - started
    final_status = read_result_status(
        run_id
    )

    if (
        completed.returncode == 0
        and final_status == "completed"
    ):
        outcome = "completed"

    elif final_status == "oom":
        outcome = "oom"

    elif final_status == "failed":
        outcome = "failed"

    else:
        outcome = "inconsistent"

    synchronized_print(
        f"[slot={worker_slot}] "
        f"[{index}] END {run_id}: "
        f"return_code={completed.returncode}, "
        f"status={final_status}, "
        f"outcome={outcome}, "
        f"seconds={duration:.1f}"
    )

    record_result(
        {
            "index": index,
            "run_id": run_id,
            "source_config": entry[
                "relative_path"
            ],
            "action": "run",
            "outcome": outcome,
            "return_code": completed.returncode,
            "final_status": final_status,
            "duration_seconds": duration,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
    )


def worker(
    worker_slot: int,
) -> None:
    while True:
        try:
            entry = task_queue.get_nowait()
        except queue.Empty:
            return

        try:
            run_config(
                worker_slot,
                entry,
            )
        except BaseException as error:
            synchronized_print(
                f"[slot={worker_slot}] "
                f"COORDINATOR ERROR: "
                f"{type(error).__name__}: {error}"
            )

            record_result(
                {
                    "index": int(entry["index"]),
                    "run_id": str(entry["run_id"]),
                    "source_config": entry[
                        "relative_path"
                    ],
                    "action": "run",
                    "outcome": "coordinator_error",
                    "return_code": None,
                    "final_status": (
                        read_result_status(
                            str(entry["run_id"])
                        )
                    ),
                    "duration_seconds": None,
                    "error_type": (
                        type(error).__name__
                    ),
                    "error_message": str(error),
                }
            )
        finally:
            task_queue.task_done()


threads = [
    threading.Thread(
        target=worker,
        args=(slot,),
        name=f"gpu-worker-{slot}",
        daemon=False,
    )
    for slot in range(gpu_count)
]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

records = sorted(
    results,
    key=lambda item: item["index"],
)

counts: dict[str, int] = {}

for record in records:
    outcome = str(record["outcome"])
    counts[outcome] = (
        counts.get(outcome, 0) + 1
    )

summary = {
    "job_id": os.environ.get(
        "SLURM_JOB_ID"
    ),
    "total_configs": len(entries),
    "completed_records": len(records),
    "counts": counts,
    "records": records,
}

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(
    "===== Packed Run Summary =====",
    flush=True,
)

print(
    json.dumps(
        counts,
        indent=2,
    ),
    flush=True,
)

print(
    f"Summary file: {summary_path}",
    flush=True,
)


hard_failure_outcomes = {
    "failed",
    "inconsistent",
    "coordinator_error",
}

hard_failures = sum(
    count
    for outcome, count in counts.items()
    if outcome in hard_failure_outcomes
)

oom_count = counts.get(
    "oom",
    0,
)

if hard_failures > 0:
    raise SystemExit(1)

if fail_on_oom and oom_count > 0:
    raise SystemExit(2)

raise SystemExit(0)
PY
}


# ============================================================================
# Entry point
# ============================================================================

if [[ "${MLSYS_GENERATED_MODE:-}" == "worker" ]]; then
    worker_mode
else
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        fail \
            "Do not call this script directly with sbatch. " \
            "Run it with bash so submission options can be generated."
    fi

    submission_mode
fi