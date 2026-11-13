#!/bin/bash
# Execute one Parakeet WER shard on one GPU node.

set -euo pipefail

: "${DATA_ROOT:?Missing DATA_ROOT}"
: "${MFA_WORK_DIR:?Missing MFA_WORK_DIR}"
: "${OUTPUT_DIR:?Missing OUTPUT_DIR}"
: "${ASR_ENV:?Missing ASR_ENV}"
: "${MODEL_CACHE_DIR:?Missing MODEL_CACHE_DIR}"
: "${TUTORIAL_ROOT:?Missing TUTORIAL_ROOT}"
: "${PARAKEET_ROOT:?Missing PARAKEET_ROOT}"
: "${SHARD_COUNT:?Missing SHARD_COUNT}"
: "${SLURM_ARRAY_TASK_ID:?This must run as a SLURM array task}"

SESSIONS_FILE="${SESSIONS_FILE:-}"
PARAKEET_MODEL="${PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v2}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-16}"
WER_THRESHOLD_PCT="${WER_THRESHOLD_PCT:-100}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_MOUNTS_B64="${CONTAINER_MOUNTS_B64:-}"
CONTAINER_MOUNTS=""
[[ -n "$CONTAINER_MOUNTS_B64" ]] &&
    CONTAINER_MOUNTS="$(printf "%s" "$CONTAINER_MOUNTS_B64" | base64 --decode)"

if [[ -n "$CONTAINER_IMAGE" && "${IN_CONTAINER:-0}" != "1" ]]; then
    SRUN_ARGS=(--nodes 1 --ntasks 1 --container-image "$CONTAINER_IMAGE")
    [[ -n "$CONTAINER_MOUNTS" ]] &&
        SRUN_ARGS+=(--container-mounts "$CONTAINER_MOUNTS")
    exec srun "${SRUN_ARGS[@]}" \
        env IN_CONTAINER=1 bash "$PARAKEET_ROOT/cluster/run_node.sh"
fi

if [[ ! -x "$ASR_ENV/bin/python" ]]; then
    echo "ERROR: compute node cannot access ASR_ENV=$ASR_ENV" >&2
    exit 1
fi
for path in "$DATA_ROOT" "$MFA_WORK_DIR/audio_16k_masked" "$MFA_WORK_DIR/textgrids"; do
    if [[ ! -d "$path" ]]; then
        echo "ERROR: missing required directory: $path" >&2
        exit 1
    fi
done

export PATH="$ASR_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHON="$ASR_ENV/bin/python"
export PYTHONPATH="$(cd "$TUTORIAL_ROOT/../../.." && pwd)"
export NEMO_CACHE_DIR="$MODEL_CACHE_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}"
SCRATCH_DIR="$SCRATCH_ROOT/david_ai_parakeet_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
rm -rf "$SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"
cleanup() {
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

echo "[$(date -Is)] Node=$(hostname) shard=$SLURM_ARRAY_TASK_ID/$SHARD_COUNT"
env \
    DATA_ROOT="$DATA_ROOT" \
    MFA_WORK_DIR="$MFA_WORK_DIR" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    SESSIONS_FILE="$SESSIONS_FILE" \
    PARAKEET_MODEL="$PARAKEET_MODEL" \
    MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
    ASR_BATCH_SIZE="$ASR_BATCH_SIZE" \
    ASR_WORKERS=1 \
    WER_THRESHOLD_PCT="$WER_THRESHOLD_PCT" \
    BUILD_LHOTSE=1 \
    SHARD_COUNT="$SHARD_COUNT" \
    SHARD_INDEX="$SLURM_ARRAY_TASK_ID" \
    SCRATCH_DIR="$SCRATCH_DIR" \
    PYTHON="$PYTHON" \
    bash "$PARAKEET_ROOT/run_parakeet_wer.sh"

echo "[$(date -Is)] Parakeet shard $SLURM_ARRAY_TASK_ID completed"
