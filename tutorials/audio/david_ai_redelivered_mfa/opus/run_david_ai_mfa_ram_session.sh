#!/bin/bash
# Strict on-the-fly David AI E2E pipeline.
#
# Every unfinished session starts from raw WAVs + machine_generated_transcript.json:
# normalize in memory -> MFA with base dictionary + runtime G2P -> RTTM ->
# manifest-mask pause-noise Opus -> mixed Opus -> ordinary and FastMSS TextGrids.
# No original per-speaker audio copy, persisted manifests, shared lexicon,
# or partial output cache is read. A validated session.done flag skips completed
# sessions and is written only after every required output passes validation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the raw-session directory}"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/workdir_e2e}"
TEXTGRID_DIR="${TEXTGRID_DIR:-$WORK_DIR/textgrids}"
AUDIO_MIXED_DIR="${AUDIO_MIXED_DIR:-$WORK_DIR/audio_mixed}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
RAM_DIR="${RAM_DIR:-/tmp/david_ai_ram_session_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}}"

MFA_ROOT_DIR="${MFA_ROOT_DIR:-$HOME/MFA_models}"
MFA_DICT_NAME="${MFA_DICT_NAME:-english_us_arpa}"
MFA_G2P="${MFA_G2P:-english_us_arpa}"
MFA_ACOUSTIC="${MFA_ACOUSTIC:-english_us_arpa}"
MFA_NUM_JOBS="${MFA_NUM_JOBS:-2}"
WORKERS="${WORKERS:-4}"
SEGMENT_PADDING="${SEGMENT_PADDING:-0.5}"
RTTM_MERGE_GAP="${RTTM_MERGE_GAP:-0.2}"
OPUS_BITRATE="${OPUS_BITRATE:-32k}"
NUM2WORDS_LANG="${NUM2WORDS_LANG:-en}"
SESSIONS_FILE="${SESSIONS_FILE:-}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" && -x "$CURATOR_ROOT/.venv/bin/python" ]]; then
    PYTHON="$CURATOR_ROOT/.venv/bin/python"
fi
PYTHON="${PYTHON:-python3}"
MFA_ENV="${MFA_ENV:-$HOME/miniconda3/envs/curator_pain_1}"
if [[ -x "$MFA_ENV/bin/mfa" ]]; then
    if [[ -n "${FFMPEG_BIN:-}" && -x "$FFMPEG_BIN" ]]; then
        export PATH="${PATH}:$MFA_ENV/bin"
    else
        export PATH="$MFA_ENV/bin:$PATH"
    fi
fi
export MFA_ROOT_DIR

if [[ -n "${FFMPEG_BIN:-}" && -x "$FFMPEG_BIN" ]]; then
    :
elif ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not on PATH" >&2
    exit 1
fi
if ! command -v mfa >/dev/null 2>&1; then
    echo "ERROR: mfa not on PATH (MFA_ENV=$MFA_ENV)" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root does not exist: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$TEXTGRID_DIR" "$AUDIO_MIXED_DIR" "$RAM_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
LOG_FILE="$LOG_DIR/run_e2e_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

CMD=(
    "$PYTHON" "$SCRIPT_DIR/stage_ram_session_pipeline.py"
    --data-root "$DATA_ROOT"
    --work-dir "$WORK_DIR"
    --audio-mixed-dir "$AUDIO_MIXED_DIR"
    --textgrid-dir "$TEXTGRID_DIR"
    --mfa-dict-name "$MFA_DICT_NAME"
    --mfa-acoustic "$MFA_ACOUSTIC"
    --mfa-g2p "$MFA_G2P"
    --ram-dir "$RAM_DIR"
    --num2words-lang "$NUM2WORDS_LANG"
    --mfa-num-jobs "$MFA_NUM_JOBS"
    --segment-padding "$SEGMENT_PADDING"
    --rttm-merge-gap "$RTTM_MERGE_GAP"
    --opus-bitrate "$OPUS_BITRATE"
    --noise-level 0.0002
    --stitch-ms 5
    --boundary-offset 0.5
    --workers "$WORKERS"
    --shard-count "$SHARD_COUNT"
    --shard-index "$SHARD_INDEX"
)
[[ -n "$SESSIONS_FILE" ]] && CMD+=(--sessions-file "$SESSIONS_FILE")

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ON-THE-FLY E2E START"
echo "DATA_ROOT=$DATA_ROOT"
echo "WORK_DIR=$WORK_DIR"
echo "WORKERS=$WORKERS MFA_NUM_JOBS=$MFA_NUM_JOBS"
echo "MFA dictionary=$MFA_DICT_NAME runtime_g2p=$MFA_G2P"
echo "Pause mask=original manifest boundaries +/-0.5s, noise=0.0002, smoothing=5ms"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ON-THE-FLY E2E DONE"
