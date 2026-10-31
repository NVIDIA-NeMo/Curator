#!/bin/bash
# Run one local Curator/Xenna pipeline with two visible GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to raw session manifests}"
MFA_WORK_DIR="${MFA_WORK_DIR:?Set MFA_WORK_DIR to completed masked-WAV MFA outputs}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for Parakeet/WER outputs}"
SESSIONS_FILE="${SESSIONS_FILE:-}"
PARAKEET_MODEL="${PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v2}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$OUTPUT_DIR/model_cache}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-16}"
WER_THRESHOLD_PCT="${WER_THRESHOLD_PCT:-100}"
GPU_IDS="${GPU_IDS:-0,1}"
PYTHON="${PYTHON:-python3}"
SCRATCH_DIR="${SCRATCH_DIR:-/tmp/david_ai_parakeet_2gpu_$$}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

IFS="," read -r GPU_0 GPU_1 EXTRA_GPUS <<< "$GPU_IDS"
if [[ -z "$GPU_0" || -z "$GPU_1" || -n "${EXTRA_GPUS:-}" || "$GPU_0" == "$GPU_1" ]]; then
    echo "ERROR: GPU_IDS must contain exactly two distinct IDs, for example 0,1" >&2
    exit 2
fi
if [[ ! "$GPU_0" =~ ^[0-9]+$ || ! "$GPU_1" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU IDs must be non-negative integers" >&2
    exit 2
fi
for path in "$DATA_ROOT" "$MFA_WORK_DIR/audio_16k_masked" "$MFA_WORK_DIR/textgrids"; do
    if [[ ! -d "$path" ]]; then
        echo "ERROR: required directory does not exist: $path" >&2
        exit 1
    fi
done
if [[ -n "$SESSIONS_FILE" && ! -f "$SESSIONS_FILE" ]]; then
    echo "ERROR: sessions file does not exist: $SESSIONS_FILE" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$MODEL_CACHE_DIR" "$SCRATCH_DIR" "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT"
export NEMO_CACHE_DIR="$MODEL_CACHE_DIR"

echo "Pre-caching $PARAKEET_MODEL once in $MODEL_CACHE_DIR"
CUDA_VISIBLE_DEVICES="" "$PYTHON" - "$PARAKEET_MODEL" <<'PY'
import sys
from nemo.collections.asr.models import ASRModel

ASRModel.from_pretrained(
    model_name=sys.argv[1],
    return_model_file=True,
)
PY

cleanup() {
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT INT TERM

echo "Starting one Xenna pipeline with visible GPUs $GPU_0,$GPU_1"
env \
    CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" \
    DATA_ROOT="$DATA_ROOT" \
    MFA_WORK_DIR="$MFA_WORK_DIR" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    SESSIONS_FILE="$SESSIONS_FILE" \
    PARAKEET_MODEL="$PARAKEET_MODEL" \
    MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
    ASR_BATCH_SIZE="$ASR_BATCH_SIZE" \
    ASR_WORKERS=2 \
    WER_THRESHOLD_PCT="$WER_THRESHOLD_PCT" \
    SHARD_COUNT=1 \
    SHARD_INDEX=0 \
    SCRATCH_DIR="$SCRATCH_DIR" \
    PYTHON="$PYTHON" \
    bash "$SCRIPT_DIR/run_parakeet_wer.sh" \
    > "$LOG_DIR/local_2gpu.log" 2>&1

"$PYTHON" "$SCRIPT_DIR/analyze_wer_distribution.py" \
    --output-dir "$OUTPUT_DIR" \
    --applied-threshold-pct "$WER_THRESHOLD_PCT"

trap - EXIT INT TERM
cleanup
echo "Two-GPU Parakeet WER pipeline completed successfully"
