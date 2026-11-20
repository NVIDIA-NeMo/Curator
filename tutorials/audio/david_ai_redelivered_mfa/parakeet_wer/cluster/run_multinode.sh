#!/bin/bash
# Submit Parakeet WER shards and a dependent Lhotse/distribution merge job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAKEET_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to raw session manifests}"
MFA_WORK_DIR="${MFA_WORK_DIR:?Set MFA_WORK_DIR to completed masked-WAV MFA outputs}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for Parakeet/WER outputs}"
ASR_ENV="${ASR_ENV:?Set ASR_ENV to a shared Parakeet environment}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:?Set MODEL_CACHE_DIR to a shared model cache}"
SESSIONS_FILE="${SESSIONS_FILE:-}"
PARAKEET_MODEL="${PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v2}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-16}"
WER_THRESHOLD_PCT="${WER_THRESHOLD_PCT:-100}"

NUM_NODES="${NUM_NODES:-1}"
MAX_CONCURRENT_NODES="${MAX_CONCURRENT_NODES:-$NUM_NODES}"
CPUS_PER_NODE="${CPUS_PER_NODE:-16}"
GPU_GRES="${GPU_GRES:-gpu:1}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MEMORY_PER_NODE="${MEMORY_PER_NODE:-64G}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
JOB_NAME="${JOB_NAME:-david-ai-parakeet-wer}"
PRELOAD_MODEL="${PRELOAD_MODEL:-1}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-}"

for path in "$DATA_ROOT" "$MFA_WORK_DIR" "$OUTPUT_DIR" "$ASR_ENV" "$MODEL_CACHE_DIR" "$TUTORIAL_ROOT"; do
    if [[ "$path" != /* || "$path" == *,* ]]; then
        echo "ERROR: shared paths must be absolute and contain no commas: $path" >&2
        exit 2
    fi
done
if ! [[ "$NUM_NODES" =~ ^[1-9][0-9]*$ && "$MAX_CONCURRENT_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_NODES and MAX_CONCURRENT_NODES must be positive integers" >&2
    exit 2
fi
if [[ -n "$SESSIONS_FILE" && ! -f "$SESSIONS_FILE" ]]; then
    echo "ERROR: sessions file does not exist: $SESSIONS_FILE" >&2
    exit 1
fi
if [[ -n "$SESSIONS_FILE" && ( "$SESSIONS_FILE" != /* || "$SESSIONS_FILE" == *,* ) ]]; then
    echo "ERROR: SESSIONS_FILE must be absolute and contain no commas" >&2
    exit 2
fi
if [[ ! -x "$ASR_ENV/bin/python" ]]; then
    echo "ERROR: ASR environment is missing python: $ASR_ENV" >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is not available" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR/logs/slurm" "$MODEL_CACHE_DIR"
export PYTHONPATH="$(cd "$TUTORIAL_ROOT/../../.." && pwd)"
export NEMO_CACHE_DIR="$MODEL_CACHE_DIR"

if [[ "$PRELOAD_MODEL" == "1" ]]; then
    "$ASR_ENV/bin/python" - "$PARAKEET_MODEL" <<'PY'
import sys
from nemo.collections.asr.models import ASRModel

ASRModel.from_pretrained(model_name=sys.argv[1], return_model_file=True)
PY
fi

MOUNTS_B64=""
[[ -n "$CONTAINER_MOUNTS" ]] &&
    MOUNTS_B64="$(printf "%s" "$CONTAINER_MOUNTS" | base64 | tr -d "\n")"

EXPORTS="DATA_ROOT=$DATA_ROOT,MFA_WORK_DIR=$MFA_WORK_DIR,OUTPUT_DIR=$OUTPUT_DIR"
EXPORTS+=",ASR_ENV=$ASR_ENV,MODEL_CACHE_DIR=$MODEL_CACHE_DIR"
EXPORTS+=",TUTORIAL_ROOT=$TUTORIAL_ROOT,PARAKEET_ROOT=$PARAKEET_ROOT"
EXPORTS+=",PARAKEET_MODEL=$PARAKEET_MODEL,ASR_BATCH_SIZE=$ASR_BATCH_SIZE"
EXPORTS+=",WER_THRESHOLD_PCT=$WER_THRESHOLD_PCT,SHARD_COUNT=$NUM_NODES"
EXPORTS+=",CONTAINER_IMAGE=$CONTAINER_IMAGE,CONTAINER_MOUNTS_B64=$MOUNTS_B64"
[[ -n "$SESSIONS_FILE" ]] && EXPORTS+=",SESSIONS_FILE=$SESSIONS_FILE"

ARRAY_ARGS=(
    --parsable
    --job-name "$JOB_NAME"
    --nodes 1
    --ntasks 1
    --cpus-per-task "$CPUS_PER_NODE"
    --gres "$GPU_GRES"
    --mem "$MEMORY_PER_NODE"
    --time "$TIME_LIMIT"
    --array "0-$((NUM_NODES - 1))%$MAX_CONCURRENT_NODES"
    --output "$OUTPUT_DIR/logs/slurm/${JOB_NAME}_%A_%a.out"
    --error "$OUTPUT_DIR/logs/slurm/${JOB_NAME}_%A_%a.err"
    --export "$EXPORTS"
)
[[ -n "$SLURM_ACCOUNT" ]] && ARRAY_ARGS+=(--account "$SLURM_ACCOUNT")
[[ -n "$SLURM_PARTITION" ]] && ARRAY_ARGS+=(--partition "$SLURM_PARTITION")

ARRAY_JOB_ID="$(sbatch "${ARRAY_ARGS[@]}" "$SCRIPT_DIR/run_node.sh")"
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"
echo "Submitted Parakeet array job: $ARRAY_JOB_ID"

MERGE_EXPORTS="OUTPUT_DIR=$OUTPUT_DIR,PARAKEET_ROOT=$PARAKEET_ROOT,ASR_ENV=$ASR_ENV"
MERGE_EXPORTS+=",WER_THRESHOLD_PCT=$WER_THRESHOLD_PCT"
MERGE_ARGS=(
    --parsable
    --job-name "${JOB_NAME}-merge"
    --nodes 1
    --ntasks 1
    --cpus-per-task 4
    --mem 32G
    --time 02:00:00
    --dependency "afterok:$ARRAY_JOB_ID"
    --output "$OUTPUT_DIR/logs/slurm/${JOB_NAME}_merge_%j.out"
    --error "$OUTPUT_DIR/logs/slurm/${JOB_NAME}_merge_%j.err"
    --export "$MERGE_EXPORTS"
)
[[ -n "$SLURM_ACCOUNT" ]] && MERGE_ARGS+=(--account "$SLURM_ACCOUNT")
[[ -n "$SLURM_PARTITION" ]] && MERGE_ARGS+=(--partition "$SLURM_PARTITION")

MERGE_JOB_ID="$(sbatch "${MERGE_ARGS[@]}" "$SCRIPT_DIR/merge_outputs.sh")"
MERGE_JOB_ID="${MERGE_JOB_ID%%;*}"
echo "Submitted dependent merge job: $MERGE_JOB_ID"
