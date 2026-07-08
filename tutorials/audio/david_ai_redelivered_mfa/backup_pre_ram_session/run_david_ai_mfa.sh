#!/bin/bash
# David AI redelivered MFA pipeline (segment-constrained, English).
#
# Stages:
#   0   manifests + normalize
#   0b  English OOV lexicon
#   1   encode 16 kHz mono Opus
#   2   MFA per segment -> per-session TextGrids + alignments.jsonl
#   3   (legacy) per-recording RTTM from TextGrid — skipped unless WRITE_TEXTGRIDS=1
#   4   Lhotse cutset + session RTTM from alignments (2 workers default)
#   5   (legacy) session RTTM merge — skipped unless WRITE_TEXTGRIDS=1
#   6   mixed session audio (pause=white noise per speaker RTTM, then Opus mix)
#   7   deliverables manifest (16k opus per speaker, mixed opus+rttm, lhotse)
#
# Data layout:
#   ${DATA_ROOT}/${session_id}/${speaker_id}_postprocessed.wav
#   ${DATA_ROOT}/${session_id}/machine_generated_transcript.json
#
# Usage:
#   bash run_david_ai_mfa.sh
#   STAGE=3 bash run_david_ai_mfa.sh              # stages 3-6
#   STAGE=5 STAGE_END=5 bash run_david_ai_mfa.sh  # stage 5 only
#   SESSION=005fe699-c4bf-45fb-ba24-7a452f2e3e20 bash run_david_ai_mfa.sh
#   FORCE=1 STAGE=3 STAGE_END=3 bash run_david_ai_mfa.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/home/ttimofeeva/FastMSS/DavidAI/d12/subser_251spk}"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/workdir}"
DONE_DIR="${DONE_DIR:-$WORK_DIR/.done}"
MANIFESTS_DIR="${MANIFESTS_DIR:-$WORK_DIR/manifests}"
AUDIO_16K_DIR="${AUDIO_16K_DIR:-$WORK_DIR/audio_16k}"
LEXICON_DIR="${LEXICON_DIR:-$WORK_DIR/lexicon}"
TEXTGRID_DIR="${TEXTGRID_DIR:-$WORK_DIR/textgrids}"
RTTM_DIR="${RTTM_DIR:-$WORK_DIR/rttm}"
RTTM_SESSION_DIR="${RTTM_SESSION_DIR:-$WORK_DIR/rttm_sessions}"
AUDIO_MIXED_DIR="${AUDIO_MIXED_DIR:-$WORK_DIR/audio_mixed}"
DELIVERABLES_DIR="${DELIVERABLES_DIR:-$WORK_DIR/deliverables}"
LHOTSE_DIR="${LHOTSE_DIR:-$WORK_DIR/lhotse}"
MFA_TEMP_DIR="${MFA_TEMP_DIR:-$WORK_DIR/mfa_temp}"
MFA_WORKERS_DIR="${MFA_WORKERS_DIR:-$MFA_TEMP_DIR/workers}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"

NUM2WORDS_LANG="${NUM2WORDS_LANG:-en}"

export MFA_ROOT_DIR="${MFA_ROOT_DIR:-/home/ttimofeeva/MFA_models}"
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
ALIGNMENTS_JSONL="${ALIGNMENTS_JSONL:-$WORK_DIR/alignments.jsonl}"
FINAL_WORKERS="${FINAL_WORKERS:-2}"
WRITE_TEXTGRIDS="${WRITE_TEXTGRIDS:-0}"
RAM_DISK="${RAM_DISK:-0}"
RAM_DIR="${RAM_DIR:-}"
STAGE="${STAGE:-0}"
STAGE_END="${STAGE_END:-7}"
FORCE="${FORCE:-0}"
SESSION="${SESSION:-}"

PYTHON="${PYTHON:-python3}"
MFA_ENV="${MFA_ENV:-$HOME/miniconda3/envs/curator_pain_1}"
if [[ -x "$CURATOR_ROOT/.venv/bin/python" ]]; then
    PYTHON="$CURATOR_ROOT/.venv/bin/python"
fi
if [[ -x "$MFA_ENV/bin/mfa" ]]; then
    export PATH="$MFA_ENV/bin:$PATH"
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not on PATH (required for stages 1/2/6)" >&2
    exit 1
fi

MFA_FALLBACK_LOG="${MFA_FALLBACK_LOG:-$LOG_DIR/mfa_segment_fallback.jsonl}"
NORMALIZATION_LOG="${NORMALIZATION_LOG:-$LOG_DIR/normalization.jsonl}"
MERGED_DICT="${MERGED_DICT:-$LEXICON_DIR/english_mfa_davidai_eng.dict}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_david_ai_mfa_${RUN_ID}.log"

mkdir -p "$LOG_DIR" "$DONE_DIR" "$MANIFESTS_DIR" "$AUDIO_16K_DIR" "$LEXICON_DIR" \
    "$TEXTGRID_DIR" "$RTTM_DIR" "$RTTM_SESSION_DIR" "$AUDIO_MIXED_DIR" "$DELIVERABLES_DIR" \
    "$LHOTSE_DIR" "$MFA_TEMP_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

stage_done() {
    [[ -f "$DONE_DIR/$1.done" ]]
}

mark_stage_done() {
    mkdir -p "$DONE_DIR"
    printf 'ok\n' >"$DONE_DIR/$1.done"
}

clear_done_from_stage() {
    [[ -n "$SESSION" ]] && return 0
    local from="$1"
    local -a names=(
        stage0 stage0b stage1 stage2_textgrid stage3_rttm
        stage4_final stage5_rttm_sessions stage6_mixed_audio stage7_deliverables
    )
    local start=0
    case "$from" in
        0) start=0 ;;
        1) start=2 ;;
        2) start=3 ;;
        3) start=4 ;;
        4) start=5 ;;
        5) start=6 ;;
        6) start=7 ;;
        7) start=8 ;;
    esac
    local i
    for ((i=start; i<${#names[@]}; i++)); do
        rm -f "$DONE_DIR/${names[$i]}.done"
    done
}

run_stage() {
    local stage_num="$1"
    local done_name="$2"
    shift 2
    if [[ -z "$SESSION" ]] && [[ "$FORCE" != "1" ]] && stage_done "$done_name"; then
        log "STAGE ${stage_num} SKIP: already done ($done_name)"
        return 0
    fi
    log "STAGE ${stage_num} START: $*"
    "$@"
    if [[ -z "$SESSION" ]]; then
        mark_stage_done "$done_name"
    fi
    log "STAGE ${stage_num} DONE"
}

should_run_stage() {
    local n="$1"
    [[ "$STAGE" -le "$n" && "$STAGE_END" -ge "$n" ]]
}

DONE_ARGS=()
if [[ -z "$SESSION" ]]; then
    DONE_ARGS=(--work-dir "$WORK_DIR")
fi

log "RUN START"
log "LOG_FILE=$LOG_FILE"
log "DATA_ROOT=$DATA_ROOT"
log "WORK_DIR=$WORK_DIR"
log "PYTHON=$PYTHON"
log "MFA_ROOT_DIR=$MFA_ROOT_DIR"
log "STAGE=$STAGE"
log "STAGE_END=$STAGE_END"
log "WORKERS=$WORKERS"
log "MFA_NUM_JOBS=$MFA_NUM_JOBS"
log "ALIGNMENTS_JSONL=$ALIGNMENTS_JSONL"
log "FINAL_WORKERS=$FINAL_WORKERS"
log "DONE_DIR=$DONE_DIR"

cd "$SCRIPT_DIR"

SESSION_ARGS=()
if [[ -n "$SESSION" ]]; then
    SESSION_ARGS=(--session "$SESSION")
fi

FORCE_ARGS=()
if [[ "$FORCE" == "1" ]]; then
    FORCE_ARGS=(--force)
    clear_done_from_stage "$STAGE"
fi

if should_run_stage 0; then
    if [[ -n "$NUM2WORDS_LANG" ]] && ! "$PYTHON" -c "import num2words" 2>/dev/null; then
        log "ERROR: stage 0 needs num2words for digit verbalization (pip install num2words in $PYTHON)"
        exit 1
    fi
    run_stage 0 stage0 \
        "$PYTHON" stage0_build_manifests.py \
        --data-root "$DATA_ROOT" \
        --manifests-dir "$MANIFESTS_DIR" \
        --audio-16k-dir "$AUDIO_16K_DIR" \
        --num2words-lang "$NUM2WORDS_LANG" \
        --normalization-log "$NORMALIZATION_LOG" \
        --workers "$WORKERS" \
        "${FORCE_ARGS[@]}" \
        "${SESSION_ARGS[@]}"
fi

if should_run_stage 0; then
    if ! command -v mfa >/dev/null 2>&1; then
        log "ERROR: mfa not on PATH"
        exit 1
    fi
    run_stage "0b" stage0b \
        "$PYTHON" stage0_build_lexicon.py \
        --manifests-dir "$MANIFESTS_DIR" \
        --lexicon-dir "$LEXICON_DIR" \
        --mfa-dict "$MFA_DICT_NAME" \
        --mfa-g2p "$MFA_G2P" \
        --mfa-root-dir "$MFA_ROOT_DIR" \
        --output-name "english_mfa_davidai_eng.dict"
    log "Merged dictionary: $MERGED_DICT"
fi

if should_run_stage 1; then
    run_stage 1 stage1 \
        "$PYTHON" stage1_resample_audio.py \
        --manifests-dir "$MANIFESTS_DIR" \
        --workers "$WORKERS" \
        "${FORCE_ARGS[@]}"
fi

if should_run_stage 2; then
    if [[ ! -f "$MERGED_DICT" ]]; then
        log "ERROR: missing merged dictionary $MERGED_DICT (run stage 0b first)"
        exit 1
    fi
    if ! command -v mfa >/dev/null 2>&1; then
        log "ERROR: mfa not on PATH"
        exit 1
    fi
    if ! "$PYTHON" -c "import textgrid" 2>/dev/null; then
        log "ERROR: stage 2 needs textgrid (pip install textgrid in $PYTHON)"
        exit 1
    fi
    log "STAGE 2 START: MFA align segments -> session TextGrids + alignments cache"
    if [[ -z "$SESSION" ]] && [[ "$FORCE" != "1" ]] && stage_done stage2_textgrid; then
        log "STAGE 2 SKIP: already done (stage2_textgrid)"
    else
        if [[ "$RAM_DISK" == "1" ]]; then
            STAGE2_ARGS=(
                --manifests-dir "$MANIFESTS_DIR"
                --mfa-dict "$MERGED_DICT"
                --mfa-acoustic "$MFA_ACOUSTIC"
                --textgrid-dir "$TEXTGRID_DIR"
                --alignments-jsonl "$ALIGNMENTS_JSONL"
                --mfa-fallback-log "$MFA_FALLBACK_LOG"
                --num-jobs "$MFA_NUM_JOBS"
                --workers "$WORKERS"
                --segment-padding "$SEGMENT_PADDING"
                "${DONE_ARGS[@]}" --stage-done-name stage2_textgrid
                "${SESSION_ARGS[@]}"
                "${FORCE_ARGS[@]}"
            )
            if [[ -n "$RAM_DIR" ]]; then
                STAGE2_ARGS+=(--ram-dir "$RAM_DIR")
            fi
            if [[ "$WRITE_TEXTGRIDS" == "1" ]]; then
                STAGE2_ARGS+=(--run-rttm --rttm-dir "$RTTM_DIR" --rttm-merge-gap "$RTTM_MERGE_GAP")
            fi
            "$PYTHON" stage2_mfa_align_ramdisk.py "${STAGE2_ARGS[@]}"
            log "STAGE 2 DONE (RAM disk; textgrids: $TEXTGRID_DIR alignments: $ALIGNMENTS_JSONL)"
        else
            STAGE2_ARGS=(
                --manifests-dir "$MANIFESTS_DIR"
                --mfa-dict "$MERGED_DICT"
                --mfa-acoustic "$MFA_ACOUSTIC"
                --textgrid-dir "$TEXTGRID_DIR"
                --alignments-jsonl "$ALIGNMENTS_JSONL"
                --mfa-temp-dir "$MFA_TEMP_DIR"
                --mfa-workers-dir "$MFA_WORKERS_DIR"
                --mfa-fallback-log "$MFA_FALLBACK_LOG"
                --num-jobs "$MFA_NUM_JOBS"
                --workers "$WORKERS"
                --segment-padding "$SEGMENT_PADDING"
                "${DONE_ARGS[@]}" --stage-done-name stage2_textgrid
                "${SESSION_ARGS[@]}"
                "${FORCE_ARGS[@]}"
            )
            if [[ "$WRITE_TEXTGRIDS" == "1" ]]; then
                STAGE2_ARGS+=(--run-rttm --rttm-dir "$RTTM_DIR" --rttm-merge-gap "$RTTM_MERGE_GAP")
            fi
            "$PYTHON" stage2_mfa_align_textgrids.py "${STAGE2_ARGS[@]}"
            log "STAGE 2 DONE (textgrids: $TEXTGRID_DIR alignments: $ALIGNMENTS_JSONL)"
        fi
    fi
fi

if should_run_stage 3 && [[ "$WRITE_TEXTGRIDS" == "1" ]]; then
    if ! "$PYTHON" -c "import textgrid" 2>/dev/null; then
        log "ERROR: stage 3 needs textgrid (pip install textgrid in $PYTHON)"
        exit 1
    fi
    log "STAGE 3 START: RTTM per recording from TextGrid"
    if [[ -z "$SESSION" ]] && [[ "$FORCE" != "1" ]] && stage_done stage3_rttm; then
        log "STAGE 3 SKIP: already done (stage3_rttm)"
    else
        "$PYTHON" stage3_build_recording_rttm.py \
            --manifests-dir "$MANIFESTS_DIR" \
            --textgrid-dir "$TEXTGRID_DIR" \
            --rttm-dir "$RTTM_DIR" \
            --mfa-fallback-log "$MFA_FALLBACK_LOG" \
            --rttm-merge-gap "$RTTM_MERGE_GAP" \
            --workers "$WORKERS" \
            "${DONE_ARGS[@]}" --stage-done-name stage3_rttm \
            "${SESSION_ARGS[@]}" \
            "${FORCE_ARGS[@]}"
        log "STAGE 3 DONE"
    fi
fi

if should_run_stage 4; then
    if ! "$PYTHON" -c "import lhotse" 2>/dev/null; then
        log "ERROR: stage 4 needs lhotse"
        exit 1
    fi
    log "STAGE 4 START: Lhotse cutset + session RTTM (from alignments, workers=$FINAL_WORKERS)"
    if [[ -z "$SESSION" ]] && [[ "$FORCE" != "1" ]] && stage_done stage4_final; then
        log "STAGE 4 SKIP: already done (stage4_final)"
    else
        "$PYTHON" stage4_build_final_outputs.py \
            --manifests-dir "$MANIFESTS_DIR" \
            --alignments-jsonl "$ALIGNMENTS_JSONL" \
            --lhotse-dir "$LHOTSE_DIR" \
            --rttm-mixed-dir "$AUDIO_MIXED_DIR" \
            --prefix "$LHOTSE_PREFIX" \
            --rttm-merge-gap "$RTTM_MERGE_GAP" \
            --workers "$FINAL_WORKERS" \
            "${DONE_ARGS[@]}" --stage-done-name stage4_final \
            "${SESSION_ARGS[@]}" \
            "${FORCE_ARGS[@]}"
        if [[ -z "$SESSION" ]]; then
            mark_stage_done stage5_rttm_sessions
        fi
        log "STAGE 4 DONE"
    fi
fi

if should_run_stage 5 && [[ "$WRITE_TEXTGRIDS" == "1" ]]; then
    log "STAGE 5 START: merge session RTTM"
    if [[ -z "$SESSION" ]] && [[ "$FORCE" != "1" ]] && stage_done stage5_rttm_sessions; then
        log "STAGE 5 SKIP: already done (stage5_rttm_sessions)"
    else
        "$PYTHON" stage5_merge_session_rttm.py \
            --manifests-dir "$MANIFESTS_DIR" \
            --rttm-dir "$RTTM_DIR" \
            --rttm-session-dir "$RTTM_SESSION_DIR" \
            "${DONE_ARGS[@]}" --stage-done-name stage5_rttm_sessions \
            "${SESSION_ARGS[@]}" \
            "${FORCE_ARGS[@]}"
        log "STAGE 5 DONE"
    fi
fi

if should_run_stage 6; then
    log "STAGE 6 START: mixed session audio (pause=white noise, then mix)"
    STAGE6_ARGS=(
        --manifests-dir "$MANIFESTS_DIR"
        --audio-mixed-dir "$AUDIO_MIXED_DIR"
        --alignments-jsonl "$ALIGNMENTS_JSONL"
        --rttm-merge-gap "$RTTM_MERGE_GAP"
        --noise-level "$MIX_NOISE_LEVEL"
        --stitch-ms "$MIX_STITCH_MS"
        --boundary-indent "$MIX_BOUNDARY_INDENT"
        --workers "$WORKERS"
        "${DONE_ARGS[@]}" --stage-done-name stage6_mixed_audio
        "${SESSION_ARGS[@]}"
        "${FORCE_ARGS[@]}"
    )
    if [[ "$PRESERVE_SPEECH" == "1" ]]; then
        STAGE6_ARGS+=(--preserve-speech)
    else
        STAGE6_ARGS+=(--no-preserve-speech)
    fi
    "$PYTHON" stage6_mix_session_audio.py "${STAGE6_ARGS[@]}"
    log "STAGE 6 DONE"
fi

if should_run_stage 7; then
    log "STAGE 7 START: deliverables manifest"
    if [[ -z "$SESSION" ]] && [[ "$FORCE" != "1" ]] && stage_done stage7_deliverables; then
        log "STAGE 7 SKIP: already done (stage7_deliverables)"
    else
        "$PYTHON" stage7_export_deliverables.py \
            --manifests-dir "$MANIFESTS_DIR" \
            --audio-16k-dir "$AUDIO_16K_DIR" \
            --audio-mixed-dir "$AUDIO_MIXED_DIR" \
            --lhotse-dir "$LHOTSE_DIR" \
            --prefix "$LHOTSE_PREFIX" \
            --deliverables-dir "$DELIVERABLES_DIR" \
            "${DONE_ARGS[@]}" --stage-done-name stage7_deliverables \
            "${SESSION_ARGS[@]}" \
            "${FORCE_ARGS[@]}"
        log "STAGE 7 DONE"
    fi
fi

log "RUN COMPLETE"
log "Manifests:      $MANIFESTS_DIR"
log "Audio 16k Opus:  $AUDIO_16K_DIR  (per-speaker)"
log "Lexicon:          $MERGED_DICT"
log "Alignments:       $ALIGNMENTS_JSONL"
log "Mixed audio+RTTM: $AUDIO_MIXED_DIR  ({session_id}.opus + {session_id}.rttm)"
log "Lhotse:           $LHOTSE_DIR/${LHOTSE_PREFIX}_aligned_cuts.jsonl.gz"
log "Deliverables:     $DELIVERABLES_DIR/manifest.jsonl"
log "Audio mixed:      $AUDIO_MIXED_DIR"
log "Done markers:     $DONE_DIR"
