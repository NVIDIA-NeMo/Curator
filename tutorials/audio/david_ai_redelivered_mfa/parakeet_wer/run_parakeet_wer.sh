#!/bin/bash
# Run the Parakeet segment-WER pipeline for one deterministic shard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to raw session manifests}"
MFA_WORK_DIR="${MFA_WORK_DIR:?Set MFA_WORK_DIR to the completed masked-WAV MFA output}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for Parakeet/WER manifests}"
SESSIONS_FILE="${SESSIONS_FILE:-}"
PARAKEET_MODEL="${PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v2}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-16}"
ASR_WORKERS="${ASR_WORKERS:-1}"
WER_THRESHOLD_PCT="${WER_THRESHOLD_PCT:-100}"
USE_RECOMMENDED_THRESHOLD="${USE_RECOMMENDED_THRESHOLD:-0}"
BUILD_LHOTSE="${BUILD_LHOTSE:-1}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SCRATCH_DIR="${SCRATCH_DIR:-${SLURM_TMPDIR:-/tmp}/david_ai_parakeet_${SLURM_JOB_ID:-local}_${SHARD_INDEX}}"
PYTHON="${PYTHON:-python3}"

export PYTHONPATH="$REPO_ROOT"
[[ -n "$MODEL_CACHE_DIR" ]] && export NEMO_CACHE_DIR="$MODEL_CACHE_DIR"

COMMAND=(
    "$PYTHON" "$SCRIPT_DIR/run_pipeline.py"
    --data-root "$DATA_ROOT"
    --masked-audio-dir "$MFA_WORK_DIR/audio_16k_masked"
    --textgrid-dir "$MFA_WORK_DIR/textgrids"
    --output-dir "$OUTPUT_DIR"
    --model-name "$PARAKEET_MODEL"
    --asr-batch-size "$ASR_BATCH_SIZE"
    --asr-workers "$ASR_WORKERS"
    --wer-threshold-pct "$WER_THRESHOLD_PCT"
    --scratch-dir "$SCRATCH_DIR"
    --shard-count "$SHARD_COUNT"
    --shard-index "$SHARD_INDEX"
)
[[ -n "$SESSIONS_FILE" ]] && COMMAND+=(--sessions-file "$SESSIONS_FILE")
[[ -n "$MODEL_CACHE_DIR" ]] && COMMAND+=(--model-cache-dir "$MODEL_CACHE_DIR")
[[ "$USE_RECOMMENDED_THRESHOLD" == "1" ]] && COMMAND+=(--use-recommended-threshold)
[[ "$BUILD_LHOTSE" == "0" ]] && COMMAND+=(--no-build-lhotse)

echo "Parakeet model: $PARAKEET_MODEL"
echo "Shard: $SHARD_INDEX/$SHARD_COUNT"
echo "Applied WER threshold: $WER_THRESHOLD_PCT%"
exec "${COMMAND[@]}"
