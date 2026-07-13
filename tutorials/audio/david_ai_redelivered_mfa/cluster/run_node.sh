#!/bin/bash
# Execute one deterministic session shard on one SLURM node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${VARIANT:?Missing VARIANT}"
: "${DATA_ROOT:?Missing DATA_ROOT}"
: "${WORK_DIR:?Missing WORK_DIR}"
: "${MFA_ENV:?Missing MFA_ENV}"
: "${MFA_ROOT_DIR:?Missing MFA_ROOT_DIR}"
: "${TUTORIAL_ROOT:?Missing TUTORIAL_ROOT}"
: "${SHARD_COUNT:?Missing SHARD_COUNT}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array task}"

SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-16}"
MFA_NUM_JOBS="${MFA_NUM_JOBS:-2}"
SEG_EXTRACT_WORKERS="${SEG_EXTRACT_WORKERS:-8}"
MIX_PREP_WORKERS="${MIX_PREP_WORKERS:-4}"
FFMPEG_TIMEOUT_S="${FFMPEG_TIMEOUT_S:-2400}"
SESSIONS_FILE="${SESSIONS_FILE:-}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_MOUNTS_B64="${CONTAINER_MOUNTS_B64:-}"
CONTAINER_MOUNTS=""
if [[ -n "$CONTAINER_MOUNTS_B64" ]]; then
    CONTAINER_MOUNTS="$(printf "%s" "$CONTAINER_MOUNTS_B64" | base64 --decode)"
fi
FFMPEG_BIN="${FFMPEG_BIN:-}"

if [[ -n "$CONTAINER_IMAGE" && "${IN_CONTAINER:-0}" != "1" ]]; then
    SRUN_ARGS=(
        --nodes 1
        --ntasks 1
        --container-image "$CONTAINER_IMAGE"
    )
    [[ -n "$CONTAINER_MOUNTS" ]] &&
        SRUN_ARGS+=(--container-mounts "$CONTAINER_MOUNTS")
    exec srun "${SRUN_ARGS[@]}" \
        env IN_CONTAINER=1 bash "$SCRIPT_DIR/run_node.sh"
fi

PIPELINE_DIR="$TUTORIAL_ROOT/$VARIANT"
RUNNER="$PIPELINE_DIR/run_david_ai_mfa_ram_session.sh"
if [[ ! -x "$RUNNER" ]]; then
    echo "ERROR: variant runner is missing or not executable: $RUNNER" >&2
    exit 1
fi
if [[ ! -x "$MFA_ENV/bin/python" || ! -x "$MFA_ENV/bin/mfa" ]]; then
    echo "ERROR: compute node cannot access MFA_ENV=$MFA_ENV" >&2
    exit 1
fi

export PATH="$MFA_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHON="$MFA_ENV/bin/python"
export PYTHONPATH="$(cd "$TUTORIAL_ROOT/../../.." && pwd)"
export MFA_ROOT_DIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

if [[ -n "$FFMPEG_BIN" ]]; then
    if [[ ! -x "$FFMPEG_BIN" ]]; then
        echo "ERROR: FFMPEG_BIN is not executable: $FFMPEG_BIN" >&2
        exit 1
    fi
    export FFMPEG_BIN
    export PATH="$(dirname "$FFMPEG_BIN"):$PATH"
elif [[ ! -x "$MFA_ENV/bin/ffmpeg" ]]; then
    echo "ERROR: ffmpeg is missing from MFA_ENV and FFMPEG_BIN is unset" >&2
    exit 1
fi

SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}"
RAM_DIR="$SCRATCH_ROOT/david_ai_${VARIANT}_${SLURM_JOB_ID}_${SHARD_INDEX}"
rm -rf "$RAM_DIR"
mkdir -p "$RAM_DIR"
cleanup() {
    rm -rf "$RAM_DIR"
}
trap cleanup EXIT

SHARED_MFA_ROOT_DIR="$MFA_ROOT_DIR"
NODE_MFA_ROOT_DIR="$RAM_DIR/model_source"
mkdir -p \
    "$NODE_MFA_ROOT_DIR/pretrained_models/dictionary" \
    "$NODE_MFA_ROOT_DIR/pretrained_models/acoustic" \
    "$NODE_MFA_ROOT_DIR/pretrained_models/g2p"

stage_model() {
    local destination_dir="$1"
    shift
    local candidate
    for candidate in "$@"; do
        if [[ -e "$candidate" ]]; then
            cp -a "$candidate" "$destination_dir/"
            return 0
        fi
    done
    echo "ERROR: no model candidate found: $*" >&2
    return 1
}

stage_model \
    "$NODE_MFA_ROOT_DIR/pretrained_models/dictionary" \
    "$SHARED_MFA_ROOT_DIR/pretrained_models/dictionary/english_us_arpa.dict" \
    "$SHARED_MFA_ROOT_DIR/pretrained_models/dictionary/english_us_arpa.txt"
stage_model \
    "$NODE_MFA_ROOT_DIR/pretrained_models/acoustic" \
    "$SHARED_MFA_ROOT_DIR/pretrained_models/acoustic/english_us_arpa.zip" \
    "$SHARED_MFA_ROOT_DIR/pretrained_models/acoustic/english_us_arpa"
stage_model \
    "$NODE_MFA_ROOT_DIR/pretrained_models/g2p" \
    "$SHARED_MFA_ROOT_DIR/pretrained_models/g2p/english_us_arpa.zip" \
    "$SHARED_MFA_ROOT_DIR/pretrained_models/g2p/english_us_arpa" \
    "$SHARED_MFA_ROOT_DIR/extracted_models/g2p/english_us_arpa_g2p"

export MFA_ROOT_DIR="$NODE_MFA_ROOT_DIR"

echo "[$(date -Is)] Node=$(hostname) job=$SLURM_JOB_ID shard=$SHARD_INDEX/$SHARD_COUNT"
echo "Variant=$VARIANT workers=$WORKERS_PER_NODE MFA_jobs=$MFA_NUM_JOBS"
echo "Scratch=$RAM_DIR"
echo "Shared model source=$SHARED_MFA_ROOT_DIR"
echo "Shard-local model source=$MFA_ROOT_DIR"

env \
    DATA_ROOT="$DATA_ROOT" \
    WORK_DIR="$WORK_DIR" \
    MFA_ENV="$MFA_ENV" \
    MFA_ROOT_DIR="$MFA_ROOT_DIR" \
    PYTHON="$PYTHON" \
    WORKERS="$WORKERS_PER_NODE" \
    MFA_NUM_JOBS="$MFA_NUM_JOBS" \
    SEG_EXTRACT_WORKERS="$SEG_EXTRACT_WORKERS" \
    MIX_PREP_WORKERS="$MIX_PREP_WORKERS" \
    FFMPEG_TIMEOUT_S="$FFMPEG_TIMEOUT_S" \
    RAM_DIR="$RAM_DIR" \
    SHARD_COUNT="$SHARD_COUNT" \
    SHARD_INDEX="$SHARD_INDEX" \
    SESSIONS_FILE="$SESSIONS_FILE" \
    bash "$RUNNER"

echo "[$(date -Is)] Shard $SHARD_INDEX completed"
