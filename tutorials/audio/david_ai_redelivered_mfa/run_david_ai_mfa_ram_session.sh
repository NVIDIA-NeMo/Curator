#!/bin/bash
# David AI redelivered MFA — RAM-by-session pipeline (no normalized JSON manifests).
#
# Each session runs end-to-end in RAM from original audio + transcript:
#   text normalization (in memory) -> 16 kHz Opus per speaker -> MFA (fallback for mix/RTTM)
#   -> session TextGrids + per-recording TextGrids -> Lhotse cuts (MFA word alignment)
#   -> merged Lhotse manifests -> mixed Opus + session RTTM
#
# Persisted outputs only:
#   audio_16k/{speaker}_{session}_postprocessed.opus
#   textgrids/{session_id}.TextGrid (+ _fastmss), {recording_id}_fastmss.TextGrid
#   lhotse/sessions/{session_id}_cuts.jsonl.gz
#   lhotse/david_ai_{recordings,supervisions,cuts,aligned_cuts}.jsonl.gz
#   audio_mixed/{session_id}.opus + {session_id}.rttm
#   lexicon/english_mfa_davidai_eng.dict  (built once from data root transcripts)
#
# Usage:
#   bash run_david_ai_mfa_ram_session.sh
#   WORKERS=16 FORCE=1 bash run_david_ai_mfa_ram_session.sh
#   SESSION=005fe699-c4bf-45fb-ba24-7a452f2e3e20 bash run_david_ai_mfa_ram_session.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/pipeline_ram:$SCRIPT_DIR/lexicon${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="${DATA_ROOT:-/home/ttimofeeva/FastMSS/DavidAI/d12/subser_251spk}"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/workdir_ram_session}"
DONE_DIR="${DONE_DIR:-$WORK_DIR/.done}"
AUDIO_16K_DIR="${AUDIO_16K_DIR:-$WORK_DIR/audio_16k}"
LEXICON_DIR="${LEXICON_DIR:-$WORK_DIR/lexicon}"
TEXTGRID_DIR="${TEXTGRID_DIR:-$WORK_DIR/textgrids}"
AUDIO_MIXED_DIR="${AUDIO_MIXED_DIR:-$WORK_DIR/audio_mixed}"
LHOTSE_DIR="${LHOTSE_DIR:-$WORK_DIR/lhotse}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"

NUM2WORDS_LANG="${NUM2WORDS_LANG:-en}"
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-/home/ttimofeeva/MFA_models}"
MFA_DICT_NAME="${MFA_DICT_NAME:-english_us_arpa}"
MFA_DICT_NAME="${MFA_DICT_NAME:-english_us_arpa}"
MFA_G2P="${MFA_G2P:-english_us_arpa}"
MFA_ACOUSTIC="${MFA_ACOUSTIC:-english_us_arpa}"
MFA_NUM_JOBS="${MFA_NUM_JOBS:-4}"
WORKERS="${WORKERS:-4}"
SEGMENT_PADDING="${SEGMENT_PADDING:-0.5}"
RTTM_MERGE_GAP="${RTTM_MERGE_GAP:-0.2}"
MIX_NOISE_LEVEL="${MIX_NOISE_LEVEL:-0.0002}"
PRESERVE_SPEECH="${PRESERVE_SPEECH:-1}"
MIX_STITCH_MS="${MIX_STITCH_MS:-5}"
MIX_BOUNDARY_INDENT="${MIX_BOUNDARY_INDENT:-0.2}"
LHOTSE_PREFIX="${LHOTSE_PREFIX:-david_ai}"
RAM_DIR="${RAM_DIR:-/dev/shm/david_ai_ram_session}"
MERGE_LHOTSE="${MERGE_LHOTSE:-1}"
FORCE="${FORCE:-0}"
SESSION="${SESSION:-}"
SKIP_LEXICON="${SKIP_LEXICON:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
STAGE_DONE_NAME="${STAGE_DONE_NAME:-ram_session_pipeline}"

PYTHON="${PYTHON:-python3}"
MFA_ENV="${MFA_ENV:-$HOME/miniconda3/envs/curator_pain_1}"
if [[ -x "$CURATOR_ROOT/.venv/bin/python" ]]; then
    PYTHON="$CURATOR_ROOT/.venv/bin/python"
fi
if [[ -x "$MFA_ENV/bin/mfa" ]]; then
    export PATH="$MFA_ENV/bin:$PATH"
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not on PATH" >&2
    exit 1
fi

MERGED_DICT="${MERGED_DICT:-$LEXICON_DIR/english_mfa_davidai_eng.dict}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_ram_session_${RUN_ID}.log"

mkdir -p "$LOG_DIR" "$DONE_DIR" "$AUDIO_16K_DIR" "$LEXICON_DIR" \
    "$TEXTGRID_DIR" "$AUDIO_MIXED_DIR" "$LHOTSE_DIR" "$RAM_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

CMD=(
    "$PYTHON" "$SCRIPT_DIR/pipeline_ram/stage_ram_session_pipeline.py"
    --data-root "$DATA_ROOT"
    --work-dir "$WORK_DIR"
    --audio-16k-dir "$AUDIO_16K_DIR"
    --audio-mixed-dir "$AUDIO_MIXED_DIR"
    --lhotse-dir "$LHOTSE_DIR"
    --textgrid-dir "$TEXTGRID_DIR"
    --lexicon-dir "$LEXICON_DIR"
    --mfa-dict "$MERGED_DICT"
    --mfa-dict-name "$MFA_DICT_NAME"
    --mfa-acoustic "$MFA_ACOUSTIC"
    --mfa-g2p "$MFA_G2P"
    --ram-dir "$RAM_DIR"
    --num2words-lang "$NUM2WORDS_LANG"
    --mfa-num-jobs "$MFA_NUM_JOBS"
    --segment-padding "$SEGMENT_PADDING"
    --rttm-merge-gap "$RTTM_MERGE_GAP"
    --noise-level "$MIX_NOISE_LEVEL"
    --stitch-ms "$MIX_STITCH_MS"
    --boundary-indent "$MIX_BOUNDARY_INDENT"
    --workers "$WORKERS"
    --lhotse-prefix "$LHOTSE_PREFIX"
    --stage-done-name "$STAGE_DONE_NAME"
    --shard-count "$SHARD_COUNT"
    --shard-index "$SHARD_INDEX"
)
[[ "$PRESERVE_SPEECH" == "1" ]] && CMD+=(--preserve-speech) || CMD+=(--no-preserve-speech)
[[ "$FORCE" == "1" ]] && CMD+=(--force)
[[ "$MERGE_LHOTSE" == "1" ]] && CMD+=(--merge-lhotse)
[[ "$SKIP_LEXICON" == "1" ]] && CMD+=(--skip-lexicon)
if [[ -n "$SESSION" ]]; then
    for s in $SESSION; do
        CMD+=(--session "$s")
    done
fi

log "RAM SESSION PIPELINE START: ${CMD[*]}"
"${CMD[@]}"
log "RAM SESSION PIPELINE DONE"
