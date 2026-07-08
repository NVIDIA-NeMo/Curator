#!/bin/bash
# Build MFA lexicon on draco: normalize all transcript text in parallel, run G2P.
#
# Uses existing data_links symlinks from the old cluster job (no linking here).
#
# Launch:
#   bash run_preprocess_lexicon_cluster.sh
#   LOCAL_RUN=1 bash run_preprocess_lexicon_cluster.sh
#   SLURM_ACCOUNT=nemotron_speechprod_asr SLURM_PARTITION=cpu_long CPUS=64 \
#     bash run_preprocess_lexicon_cluster.sh
#
# After this finishes, run the RAM pipeline with SKIP_LEXICON=1:
#   SKIP_LEXICON=1 bash run_david_ai_mfa_ram_session_cluster.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$TUTORIAL_DIR:$TUTORIAL_DIR/pipeline_ram:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CLUSTER_BASE="${CLUSTER_BASE:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva}"
LINK_WORK_DIR="${LINK_WORK_DIR:-$CLUSTER_BASE/david_ai_mfa_workdir}"
DATA_ROOT="${DATA_ROOT:-$LINK_WORK_DIR/data_links}"
WORK_DIR="${WORK_DIR:-$CLUSTER_BASE/david_ai_mfa_ram_workdir}"
LEXICON_DIR="${LEXICON_DIR:-$WORK_DIR/lexicon}"
MANIFESTS_DIR="${MANIFESTS_DIR:-$LINK_WORK_DIR/manifests}"

JOB_NAME="${JOB_NAME:-david_ai_lexicon}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CPUS="${CPUS:-64}"
MEM="${MEM:-}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"

WORKERS="${WORKERS:-}"
NUM2WORDS_LANG="${NUM2WORDS_LANG:-en}"
USE_MANIFESTS="${USE_MANIFESTS:-0}"
RENORMALIZE="${RENORMALIZE:-0}"

if [[ -z "${SLURM_JOB_ID:-}" && "${LOCAL_RUN:-0}" != "1" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch not found. Use LOCAL_RUN=1 inside an allocation." >&2
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
    # shellcheck disable=SC2206
    [[ -n "$EXTRA_SBATCH" ]] && SB_ARGS+=($EXTRA_SBATCH)
    echo "Submitting: sbatch ${SB_ARGS[*]} $0"
    exec sbatch "${SB_ARGS[@]}" "$0"
fi

mkdir -p "$WORK_DIR/logs" "$LEXICON_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

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

ensure_curator_pkg() {
    local tarball="${CURATOR_PKG_TARBALL:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/nemo_curator_pkg.tar.gz}"
    local dest=/tmp/curator_pkg
    if [[ -f "$dest/nemo_curator/__init__.py" ]]; then
        export PYTHONPATH="$dest${PYTHONPATH:+:$PYTHONPATH}"
        return 0
    fi
    if [[ ! -f "$tarball" ]]; then
        log "WARN: curator package tarball not found: $tarball"
        return 0
    fi
    log "Extracting nemo_curator package tarball to $dest"
    rm -rf "$dest"
    mkdir -p "$dest"
    tar -xzf "$tarball" -C "$dest"
    export PYTHONPATH="$dest${PYTHONPATH:+:$PYTHONPATH}"
}

ensure_curator_deps() {
    local py="${PYTHON:-python}"
    if "$py" -c "import cosmos_xenna, pandas, pyarrow" 2>/dev/null; then
        return 0
    fi
    log "Installing runtime deps into $py"
    "$py" -m pip install -q cosmos-xenna pandas pyarrow num2words || log "WARN: dep install failed"
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
    log "Extracting MFA models to /tmp"
    rm -rf /tmp/MFA_models
    tar -xzf "$tarball" -C /tmp
    export MFA_ROOT_DIR=/tmp/MFA_models
}

ensure_mfa_models

ALLOC_CPUS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-$CPUS}}"
if [[ -z "$WORKERS" ]]; then
    WORKERS="$ALLOC_CPUS"
    [[ "$WORKERS" -lt 1 ]] && WORKERS=1
fi

log "LEXICON PREPROCESS START (job=${SLURM_JOB_ID:-local})"
log "DATA_ROOT=$DATA_ROOT LEXICON_DIR=$LEXICON_DIR WORKERS=$WORKERS"

if [[ "$USE_MANIFESTS" == "1" ]]; then
    if [[ ! -d "$MANIFESTS_DIR" ]]; then
        log "ERROR: manifests dir not found: $MANIFESTS_DIR"
        exit 1
    fi
    INPUT_ARGS=(--manifests-dir "$MANIFESTS_DIR")
    [[ "$RENORMALIZE" == "1" ]] && INPUT_ARGS+=(--renormalize)
    log "Source: existing manifests under $MANIFESTS_DIR"
else
    if [[ ! -d "$DATA_ROOT" ]]; then
        log "ERROR: data_links not found: $DATA_ROOT"
        exit 1
    fi
    link_count=$(find "$DATA_ROOT" -maxdepth 1 -type l 2>/dev/null | wc -l)
    log "Source: $link_count symlinks under $DATA_ROOT"
    INPUT_ARGS=(--data-root "$DATA_ROOT")
fi

CMD=(
    "$PYTHON" "$SCRIPT_DIR/preprocess_build_lexicon.py"
    "${INPUT_ARGS[@]}"
    --lexicon-dir "$LEXICON_DIR"
    --workers "$WORKERS"
    --num2words-lang "$NUM2WORDS_LANG"
    --words-out "$LEXICON_DIR/all_words.txt"
)

log "Running: ${CMD[*]}"
"${CMD[@]}"
log "LEXICON PREPROCESS DONE -> $LEXICON_DIR/english_mfa_davidai_eng.dict"
