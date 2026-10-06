#!/bin/bash
# Submit the David AI E2E pipeline as one SLURM array task per node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANT="${VARIANT:-wav}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the shared raw-session directory}"
WORK_DIR="${WORK_DIR:?Set WORK_DIR to the shared output directory}"
MFA_ENV="${MFA_ENV:?Set MFA_ENV to an environment visible on compute nodes}"
MFA_ROOT_DIR="${MFA_ROOT_DIR:-$HOME/MFA_models}"
SESSIONS_FILE="${SESSIONS_FILE:-}"

NUM_NODES="${NUM_NODES:-1}"
MAX_CONCURRENT_NODES="${MAX_CONCURRENT_NODES:-$NUM_NODES}"
CPUS_PER_NODE="${CPUS_PER_NODE:-64}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-16}"
MFA_NUM_JOBS="${MFA_NUM_JOBS:-2}"
SEG_EXTRACT_WORKERS="${SEG_EXTRACT_WORKERS:-8}"
MIX_PREP_WORKERS="${MIX_PREP_WORKERS:-4}"
FFMPEG_TIMEOUT_S="${FFMPEG_TIMEOUT_S:-2400}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MEMORY_PER_NODE="${MEMORY_PER_NODE:-}"
JOB_NAME="${JOB_NAME:-david-ai-${VARIANT}}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs/slurm}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-}"
FFMPEG_BIN="${FFMPEG_BIN:-}"
CONTAINER_MOUNTS_B64=""
if [[ -n "$CONTAINER_MOUNTS" ]]; then
    CONTAINER_MOUNTS_B64="$(printf "%s" "$CONTAINER_MOUNTS" | base64 | tr -d "\n")"
fi

case "$VARIANT" in
    opus | wav) ;;
    *)
        echo "ERROR: VARIANT must be 'opus' or 'wav', got: $VARIANT" >&2
        exit 2
        ;;
esac
for value in "$NUM_NODES" "$MAX_CONCURRENT_NODES" "$CPUS_PER_NODE" \
    "$WORKERS_PER_NODE" "$MFA_NUM_JOBS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: node/thread counts must be positive integers" >&2
        exit 2
    fi
done
for path in "$DATA_ROOT" "$WORK_DIR" "$MFA_ENV" "$MFA_ROOT_DIR" "$TUTORIAL_ROOT"; do
    if [[ "$path" != /* ]]; then
        echo "ERROR: cluster paths must be absolute: $path" >&2
        exit 2
    fi
    if [[ "$path" == *,* ]]; then
        echo "ERROR: cluster paths cannot contain commas: $path" >&2
        exit 2
    fi
done
if [[ -n "$SESSIONS_FILE" ]]; then
    if [[ "$SESSIONS_FILE" != /* || "$SESSIONS_FILE" == *,* ]]; then
        echo "ERROR: SESSIONS_FILE must be an absolute path without commas" >&2
        exit 2
    fi
    if [[ ! -f "$SESSIONS_FILE" ]]; then
        echo "ERROR: SESSIONS_FILE does not exist: $SESSIONS_FILE" >&2
        exit 1
    fi
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is not available" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: DATA_ROOT does not exist: $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -x "$MFA_ENV/bin/python" || ! -x "$MFA_ENV/bin/mfa" ]]; then
    echo "ERROR: MFA_ENV must contain executable python and mfa: $MFA_ENV" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$WORK_DIR"

# Export only task-required values. Do not propagate the submit shell environment.
EXPORTS="VARIANT=$VARIANT"
EXPORTS+=",DATA_ROOT=$DATA_ROOT,WORK_DIR=$WORK_DIR"
EXPORTS+=",MFA_ENV=$MFA_ENV,MFA_ROOT_DIR=$MFA_ROOT_DIR"
EXPORTS+=",TUTORIAL_ROOT=$TUTORIAL_ROOT"
EXPORTS+=",SHARD_COUNT=$NUM_NODES"
EXPORTS+=",WORKERS_PER_NODE=$WORKERS_PER_NODE,MFA_NUM_JOBS=$MFA_NUM_JOBS"
EXPORTS+=",SEG_EXTRACT_WORKERS=$SEG_EXTRACT_WORKERS,MIX_PREP_WORKERS=$MIX_PREP_WORKERS"
EXPORTS+=",FFMPEG_TIMEOUT_S=$FFMPEG_TIMEOUT_S"
EXPORTS+=",CONTAINER_IMAGE=$CONTAINER_IMAGE,CONTAINER_MOUNTS_B64=$CONTAINER_MOUNTS_B64"
EXPORTS+=",FFMPEG_BIN=$FFMPEG_BIN"
[[ -n "$SESSIONS_FILE" ]] && EXPORTS+=",SESSIONS_FILE=$SESSIONS_FILE"

SBATCH_ARGS=(
    --job-name "$JOB_NAME"
    --nodes 1
    --ntasks 1
    --cpus-per-task "$CPUS_PER_NODE"
    --time "$TIME_LIMIT"
    --array "0-$((NUM_NODES - 1))%$MAX_CONCURRENT_NODES"
    --output "$LOG_DIR/${JOB_NAME}_%A_%a.out"
    --error "$LOG_DIR/${JOB_NAME}_%A_%a.err"
    --export "$EXPORTS"
)
[[ -n "$SLURM_ACCOUNT" ]] && SBATCH_ARGS+=(--account "$SLURM_ACCOUNT")
[[ -n "$SLURM_PARTITION" ]] && SBATCH_ARGS+=(--partition "$SLURM_PARTITION")
[[ -n "$MEMORY_PER_NODE" ]] && SBATCH_ARGS+=(--mem "$MEMORY_PER_NODE")

echo "Submitting $NUM_NODES-node $VARIANT pipeline"
echo "Per node: CPUs=$CPUS_PER_NODE session_workers=$WORKERS_PER_NODE MFA_jobs=$MFA_NUM_JOBS"
echo "Input:  $DATA_ROOT"
echo "Output: $WORK_DIR"

sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_node.sh"
