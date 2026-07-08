#!/bin/bash
# David AI redelivered MFA pipeline — SLURM cluster variant (RAM disk).
#
# Full pipeline on the cluster (stages 0..7 of the main orchestrator, RAM_DISK=1):
#   link  symlink source sessions from lustre -> local cluster DATA_ROOT
#   0     manifests + normalized texts
#   0b    whole merged lexicon (dict + G2P for OOV)
#   1     convert audios -> 16 kHz mono Opus (multi-thread)
#   2     MFA align per segment (multi-thread + multi-job) on /dev/shm RAM disk
#   4     Lhotse cutset + session RTTM -> audio_mixed/{session_id}.rttm
#   6     mixed session audio (pause=white noise per speaker, session RTTM) -> .opus
#   7     deliverables manifest (16k opus, mixed opus+rttm, lhotse)
#   (stages 3/5 legacy RTTM merge skipped unless WRITE_TEXTGRIDS=1)
#
# All heavy MFA scratch lives on tmpfs (/dev/shm) and is removed on exit.
# Persisted under WORK_DIR: manifests, lexicon, audio_16k, alignments, textgrids,
# audio_mixed (.opus + .rttm), lhotse, deliverables.
#
# Two ways to launch:
#   1) Self-submit (default): from a login node just run
#        bash run_david_ai_mfa_cluster.sh
#      It builds the sbatch args from env vars below and resubmits itself.
#   2) Manual sbatch:
#        sbatch --account <acct> --partition <part> run_david_ai_mfa_cluster.sh
#   3) Inside an existing allocation (salloc / interactive):
#        LOCAL_RUN=1 bash run_david_ai_mfa_cluster.sh
#
# Common env overrides:
#   SRC_DATA_ROOT   source tree on lustre (default below)
#   CLUSTER_BASE    lustre user root (default: /lustre/fsw/portfolios/nemotron/users/ttimofeeva)
#   WORK_DIR        MFA work/output path (default: $CLUSTER_BASE/david_ai_mfa_workdir)
#   DATA_ROOT       local symlink tree scanned by stage 0 (default: $WORK_DIR/data_links)
#   SLURM_ACCOUNT / SLURM_PARTITION / TIME_LIMIT / CPUS / MEM / JOB_NAME
#   CONTAINER_IMAGE / CONTAINER_MOUNTS   (pyxis/enroot; optional)
#   WORKERS / MFA_NUM_JOBS / FINAL_WORKERS  (default derived from allocated CPUs)
#   MFA_ENV / MFA_ROOT_DIR / PYTHON      (toolchain locations)
#   FORCE=1                              (re-run stages, ignore .done markers)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Source + local paths ----------------------------------------------------
SRC_DATA_ROOT="${SRC_DATA_ROOT:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/DavidAI_2026-05-29_redeliver}"
CLUSTER_BASE="${CLUSTER_BASE:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva}"
WORK_DIR="${WORK_DIR:-$CLUSTER_BASE/david_ai_mfa_workdir}"
DATA_ROOT="${DATA_ROOT:-$WORK_DIR/data_links}"

# --- SLURM knobs (used only for self-submit) ---------------------------------
JOB_NAME="${JOB_NAME:-david_ai_mfa}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS="${CPUS:-64}"
MEM="${MEM:-}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"

# =============================================================================
# Phase A: self-submit to SLURM when not already running inside an allocation.
# =============================================================================
if [[ -z "${SLURM_JOB_ID:-}" && "${LOCAL_RUN:-0}" != "1" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch not found. Run inside an allocation with LOCAL_RUN=1, " \
             "or submit manually." >&2
        exit 1
    fi

    mkdir -p "$WORK_DIR/logs"
    SLURM_LOG="${SLURM_LOG:-$WORK_DIR/logs/${JOB_NAME}_%j.out}"

    SB_ARGS=(
        --job-name "$JOB_NAME"
        --nodes 1
        --ntasks 1
        --cpus-per-task "$CPUS"
        --time "$TIME_LIMIT"
        --output "$SLURM_LOG"
        --export ALL
    )
    [[ -n "$SLURM_ACCOUNT" ]] && SB_ARGS+=(--account "$SLURM_ACCOUNT")
    [[ -n "$SLURM_PARTITION" ]] && SB_ARGS+=(--partition "$SLURM_PARTITION")
    [[ -n "$MEM" ]] && SB_ARGS+=(--mem "$MEM")
    [[ -n "$CONTAINER_IMAGE" ]] && SB_ARGS+=(--container-image "$CONTAINER_IMAGE")
    [[ -n "$CONTAINER_MOUNTS" ]] && SB_ARGS+=(--container-mounts "$CONTAINER_MOUNTS")
    # shellcheck disable=SC2206
    [[ -n "$EXTRA_SBATCH" ]] && SB_ARGS+=($EXTRA_SBATCH)

    echo "Submitting: sbatch ${SB_ARGS[*]} $0"
    exec sbatch "${SB_ARGS[@]}" "$0"
fi

# =============================================================================
# Phase B: runs inside the SLURM allocation (or LOCAL_RUN=1).
# =============================================================================
mkdir -p "$WORK_DIR/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LINK_LOG="$WORK_DIR/logs/link_source_${RUN_ID}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# --- Draco env + conda bootstrap (compute nodes lack login /tmp install) -----
DRACO_ENV="${DRACO_ENV:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/david_ai_mfa_env.sh}"
if [[ -f "$DRACO_ENV" ]]; then
    set +e
    # shellcheck source=/dev/null
    source "$DRACO_ENV"
    set -e
fi

ensure_conda_on_node() {
    local miniconda="${MINICONDA_DIR:-/tmp/miniconda3_tt}"
    local env_name="${CONDA_ENV:-curator_pain_1}"
    if [[ -x "$miniconda/envs/$env_name/bin/python" ]]; then
        export MINICONDA_DIR="$miniconda"
        export MFA_ENV="$miniconda/envs/$env_name"
        export PYTHON="$MFA_ENV/bin/python"
        export PATH="$MFA_ENV/bin:$PATH"
        return 0
    fi
    log "conda env missing on $(hostname) — installing to $miniconda"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip_cache_tt}"
    export TMPDIR="${TMPDIR:-/tmp/tmp_tt}"
    mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
    local installer=/tmp/Miniconda3-latest-Linux-x86_64.sh
    if [[ ! -f "$installer" ]]; then
        curl -fsSL -o "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    fi
    rm -rf "$miniconda"
    bash "$installer" -b -p "$miniconda"
    # shellcheck source=/dev/null
    source "$miniconda/etc/profile.d/conda.sh"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    conda create -y -n "$env_name" python=3.12
    conda activate "$env_name"
    conda install -y -c conda-forge montreal-forced-aligner=3.3.9 ffmpeg
    local curator="${CURATOR_ROOT:-/lustre/fsw/portfolios/nemotron/users/ttimofeeva/Curator}"
    export PYTHONPATH="$curator"
    pip install -q lhotse textgrid num2words hydra-core omegaconf soundfile tqdm pyloudnorm praatio \
        cosmos-xenna pandas pyarrow
    export MINICONDA_DIR="$miniconda"
    export MFA_ENV="$miniconda/envs/$env_name"
    export PYTHON="$MFA_ENV/bin/python"
    export PATH="$MFA_ENV/bin:$PATH"
    log "conda ready: $PYTHON"
}

ensure_conda_on_node

# nemo_curator must be importable for text normalization (stage 0). The lustre
# checkout is slow/incomplete for small files, so extract a package tarball to
# /tmp on the compute node (mirrors ensure_mfa_models) and point PYTHONPATH at it.
ensure_curator_pkg() {
    local tarball="${CURATOR_PKG_TARBALL:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/nemo_curator_pkg.tar.gz}"
    local dest=/tmp/curator_pkg
    if [[ -f "$dest/nemo_curator/__init__.py" ]]; then
        export PYTHONPATH="$dest${PYTHONPATH:+:$PYTHONPATH}"
        return 0
    fi
    if [[ ! -f "$tarball" ]]; then
        log "WARN: curator package tarball not found: $tarball (relying on existing PYTHONPATH)"
        return 0
    fi
    log "Extracting nemo_curator package tarball to $dest"
    rm -rf "$dest"
    mkdir -p "$dest"
    tar -xzf "$tarball" -C "$dest"
    export PYTHONPATH="$dest${PYTHONPATH:+:$PYTHONPATH}"
    log "PYTHONPATH=$PYTHONPATH"
}

# Always ensure runtime deps (idempotent; fast when already present) so a cached
# /tmp conda env from an earlier run still gets nemo_curator's import deps.
ensure_curator_deps() {
    local py="${PYTHON:-python}"
    if "$py" -c "import cosmos_xenna, pandas, pyarrow" 2>/dev/null; then
        return 0
    fi
    log "Installing nemo_curator import deps (cosmos-xenna pandas pyarrow) into $py"
    "$py" -m pip install -q cosmos-xenna pandas pyarrow || log "WARN: dep install failed"
}

ensure_curator_pkg
ensure_curator_deps
export MFA_ENV PYTHON PYTHONPATH PIP_CACHE_DIR TMPDIR MFA_ROOT_DIR CURATOR_ROOT

ensure_mfa_models() {
    local dict="${MFA_ROOT_DIR:-}/pretrained_models/dictionary/english_us_arpa.dict"
    if [[ -f "$dict" ]]; then
        return 0
    fi
    local tarball="${MFA_MODELS_TARBALL:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/MFA_models_draco.tar.gz}"
    if [[ ! -f "$tarball" ]]; then
        log "WARN: MFA tarball not found: $tarball"
        return 0
    fi
    log "Extracting MFA models from tarball to /tmp (avoid lustre small-file unpack hang)"
    rm -rf /tmp/MFA_models
    tar -xzf "$tarball" -C /tmp
    export MFA_ROOT_DIR=/tmp/MFA_models
    log "MFA_ROOT_DIR=$MFA_ROOT_DIR"
}

ensure_mfa_models

# Stage 0 writes ~2 manifest files per session (~190k small files). Lustre fs12
# stalls under that small-file write load, so keep manifests on node-local /tmp
# and archive them to a single lustre tarball at the end. All downstream stages
# run in this same job on this node, so they read the local manifests directly.
MANIFESTS_TARBALL="${MANIFESTS_TARBALL:-$WORK_DIR/manifests.tar.gz}"
export MANIFESTS_DIR="${MANIFESTS_DIR:-/tmp/david_ai_manifests}"
ensure_manifests_local() {
    mkdir -p "$MANIFESTS_DIR"
    local have
    have=$(find "$MANIFESTS_DIR" -maxdepth 1 -name '*_norm.jsonl' 2>/dev/null | wc -l)
    if [[ "$have" -gt 0 ]]; then
        log "MANIFESTS_DIR=$MANIFESTS_DIR ($have norm files already present)"
        return 0
    fi
    if [[ -f "$MANIFESTS_TARBALL" ]]; then
        log "Restoring manifests from $MANIFESTS_TARBALL -> $MANIFESTS_DIR"
        tar -xzf "$MANIFESTS_TARBALL" -C "$MANIFESTS_DIR"
    fi
    log "MANIFESTS_DIR=$MANIFESTS_DIR"
}
ensure_manifests_local

ALLOC_CPUS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-$CPUS}}"
if [[ -z "${MFA_NUM_JOBS:-}" ]]; then
    MFA_NUM_JOBS=$(( ALLOC_CPUS >= 4 ? 4 : ALLOC_CPUS ))
    [[ "$MFA_NUM_JOBS" -lt 1 ]] && MFA_NUM_JOBS=1
fi
if [[ -z "${WORKERS:-}" ]]; then
    WORKERS=$(( ALLOC_CPUS / MFA_NUM_JOBS ))
    [[ "$WORKERS" -lt 1 ]] && WORKERS=1
fi
if [[ -z "${FINAL_WORKERS:-}" ]]; then
    FINAL_WORKERS=$(( WORKERS >= 2 ? 2 : WORKERS ))
fi

log "CLUSTER RUN START (job=${SLURM_JOB_ID:-local})"
log "SRC_DATA_ROOT=$SRC_DATA_ROOT"
log "DATA_ROOT=$DATA_ROOT"
log "WORK_DIR=$WORK_DIR"
log "ALLOC_CPUS=$ALLOC_CPUS WORKERS=$WORKERS MFA_NUM_JOBS=$MFA_NUM_JOBS FINAL_WORKERS=$FINAL_WORKERS"

# --- Stage: link source sessions into the local DATA_ROOT --------------------
if [[ ! -d "$SRC_DATA_ROOT" ]]; then
    log "ERROR: source data root not found: $SRC_DATA_ROOT"
    exit 1
fi

mkdir -p "$DATA_ROOT"
existing_links=$(find "$DATA_ROOT" -maxdepth 1 -type l 2>/dev/null | wc -l)
if [[ "$existing_links" -gt 0 && "${RELINK:-0}" != "1" ]]; then
    log "STAGE link SKIP: $existing_links symlinks already present in $DATA_ROOT (set RELINK=1 to force re-link)"
else
    log "STAGE link START: symlinking sessions -> $DATA_ROOT (log: $LINK_LOG)"
    linked=0
    skipped=0
    missing=0
    : >"$LINK_LOG"
    for session_dir in "$SRC_DATA_ROOT"/*/; do
        [[ -d "$session_dir" ]] || continue
        session_name="$(basename "$session_dir")"
        # Only mirror sessions that actually carry a transcript (matches discover_sessions()).
        if [[ ! -f "$session_dir/machine_generated_transcript.json" ]]; then
            echo "MISSING_TRANSCRIPT $session_name" >>"$LINK_LOG"
            missing=$((missing + 1))
            continue
        fi
        link_path="$DATA_ROOT/$session_name"
        if [[ -L "$link_path" || -e "$link_path" ]]; then
            skipped=$((skipped + 1))
            continue
        fi
        ln -s "${session_dir%/}" "$link_path"
        echo "LINKED $session_name" >>"$LINK_LOG"
        linked=$((linked + 1))
    done
    log "STAGE link DONE: linked=$linked skipped=$skipped missing_transcript=$missing total_target=$(find "$DATA_ROOT" -maxdepth 1 -type l | wc -l)"

    if [[ "$linked" -eq 0 && "$skipped" -eq 0 ]]; then
        log "ERROR: no sessions linked (nothing with machine_generated_transcript.json under $SRC_DATA_ROOT)"
        exit 1
    fi
fi

# --- Stages 0..7 via the main orchestrator (RAM disk for MFA stage 2) --------
export DATA_ROOT WORK_DIR WORKERS MFA_NUM_JOBS FINAL_WORKERS MANIFESTS_DIR
export RAM_DISK=1
export STAGE="${STAGE:-0}"
export STAGE_END="${STAGE_END:-7}"
export FORCE="${FORCE:-0}"
export MFA_ENV="${MFA_ENV:-}"
export PYTHON="${PYTHON:-}"
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export MFA_MODELS_TARBALL="${MFA_MODELS_TARBALL:-}"

log "Handing off to run_david_ai_mfa.sh (RAM_DISK=1, STAGE=$STAGE..$STAGE_END)"
# Under sbatch, SCRIPT_DIR points to the SLURM spool copy (sibling scripts absent).
# Prefer the real tutorial dir from the env file.
RUN_DIR="${MFA_TUTORIAL:-$SCRIPT_DIR}"
if [[ ! -f "$RUN_DIR/run_david_ai_mfa.sh" ]]; then
    log "ERROR: run_david_ai_mfa.sh not found in $RUN_DIR (set MFA_TUTORIAL)"
    exit 1
fi
cd "$RUN_DIR"
bash "$RUN_DIR/run_david_ai_mfa.sh"

# Persist node-local manifests to a single lustre tarball (one big file avoids
# the small-file write stall). Downstream reruns restore from it automatically.
manifest_norm=$(find "$MANIFESTS_DIR" -maxdepth 1 -name '*_norm.jsonl' 2>/dev/null | wc -l)
if [[ "$manifest_norm" -gt 0 ]]; then
    log "Archiving $manifest_norm manifest sessions -> $MANIFESTS_TARBALL"
    if tar -czf "$MANIFESTS_TARBALL.tmp" -C "$MANIFESTS_DIR" . && mv "$MANIFESTS_TARBALL.tmp" "$MANIFESTS_TARBALL"; then
        log "Manifests archived: $MANIFESTS_TARBALL"
    else
        log "WARN: manifest archive failed (left in $MANIFESTS_DIR)"
        rm -f "$MANIFESTS_TARBALL.tmp"
    fi
fi

log "CLUSTER RUN COMPLETE"
log "Manifests:       $MANIFESTS_DIR (node-local) archived to $MANIFESTS_TARBALL"
log "Audio 16k Opus:  $WORK_DIR/audio_16k"
log "Alignments:      $WORK_DIR/alignments.jsonl"
log "Mixed opus+rttm: $WORK_DIR/audio_mixed"
log "Lhotse:          $WORK_DIR/lhotse"
log "Deliverables:    $WORK_DIR/deliverables"
