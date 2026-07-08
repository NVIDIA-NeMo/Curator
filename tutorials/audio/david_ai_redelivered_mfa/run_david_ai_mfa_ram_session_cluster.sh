#!/bin/bash
# David AI RAM-by-session pipeline — SLURM cluster variant.
#
# No normalized JSON manifests: each session is processed in RAM from source
# audio + transcript (norm -> 16k -> MFA w/ fallback for mix/RTTM -> Lhotse MFA-only -> mix + RTTM).
#
# Launch:
#   bash run_david_ai_mfa_ram_session_cluster.sh
#   LOCAL_RUN=1 bash run_david_ai_mfa_ram_session_cluster.sh
#   SLURM_ACCOUNT=nemotron_speechprod_asr SLURM_PARTITION=cpu_long CPUS=64 \
#     bash run_david_ai_mfa_ram_session_cluster.sh
#
# Container (same pyxis image as the lexicon punct-repair try). Draco's sbatch
# rejects --container-image, so we submit a plain batch job and enter the image
# with `srun --container-image` inside the allocation. MFA is not in the image,
# so the prebuilt lustre conda env (setup_draco_cluster.sh) is reused via mounts:
#   SLURM_ACCOUNT=... SLURM_PARTITION=... bash run_david_ai_mfa_ram_session_cluster.sh
#   RAM_USE_CONTAINER=0 ...   # disable container, provision conda on the node

set -euo pipefail

# Under sbatch/srun this script may execute from a spool copy; RAM_SCRIPT_DIR
# (exported at submit) keeps sibling .py/.sh lookups pointing at the tutorial dir.
SCRIPT_DIR="${RAM_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

SRC_DATA_ROOT="${SRC_DATA_ROOT:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/DavidAI_2026-05-29_redeliver}"
CLUSTER_BASE="${CLUSTER_BASE:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva}"
# Outputs from the RAM pipeline (separate from the old multi-stage workdir).
WORK_DIR="${WORK_DIR:-$CLUSTER_BASE/david_ai_mfa_ram_workdir}"
# Reuse symlinks created by run_david_ai_mfa_cluster.sh (no linking in this script).
LINK_WORK_DIR="${LINK_WORK_DIR:-$CLUSTER_BASE/david_ai_mfa_workdir}"
DATA_ROOT="${DATA_ROOT:-$LINK_WORK_DIR/data_links}"

JOB_NAME="${JOB_NAME:-david_ai_ram_session}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
CPUS="${CPUS:-64}"
MEM="${MEM:-}"
# Pyxis container: entered via `srun --container-image` inside the batch job
# (sbatch --container-image is unsupported on Draco). Default ON to mirror the
# previous punct-repair try; set RAM_USE_CONTAINER=0 to run on bare conda.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/users/pzelasko/containers/nemo-25.11-pytorch2.9-automodel-23apr26.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12}"
RAM_USE_CONTAINER="${RAM_USE_CONTAINER:-1}"
# Packed conda env lives on lustre, but is unpacked to node-local /tmp for each
# job. Unpacking millions of conda files directly to lustre can hang.
MINICONDA_DIR="${MINICONDA_DIR:-/tmp/miniconda3_tt}"
CONTAINER_MINICONDA="${CONTAINER_MINICONDA:-$MINICONDA_DIR}"
CONDA_ENV_TARBALL="${CONDA_ENV_TARBALL:-$CLUSTER_BASE/curator_pain_1_draco.tar.gz}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"
# Multi-node scaling: submit a SLURM array of RAM_ARRAY_COUNT tasks (1 node each).
# Each task processes a disjoint shard (session_index % count == task_id), so the
# tasks never redo each other's sessions. Set to 1 for a single-node run.
RAM_ARRAY_COUNT="${RAM_ARRAY_COUNT:-1}"

if [[ -z "${SLURM_JOB_ID:-}" && "${LOCAL_RUN:-0}" != "1" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch not found. Use LOCAL_RUN=1 inside an allocation." >&2
        exit 1
    fi
    mkdir -p "$WORK_DIR/logs"
    SB_ARGS=(
        --job-name "$JOB_NAME"
        --nodes 1
        --ntasks 1
        --cpus-per-task "$CPUS"
        --time "$TIME_LIMIT"
        --export ALL
    )
    if [[ "$RAM_ARRAY_COUNT" -gt 1 ]]; then
        # One node per shard; %A=array job id, %a=task id (=shard index).
        SLURM_LOG="${SLURM_LOG:-$WORK_DIR/logs/${JOB_NAME}_%A_%a.out}"
        SB_ARGS+=(--array "0-$((RAM_ARRAY_COUNT - 1))" --output "$SLURM_LOG")
    else
        SLURM_LOG="${SLURM_LOG:-$WORK_DIR/logs/${JOB_NAME}_%j.out}"
        SB_ARGS+=(--output "$SLURM_LOG")
    fi
    [[ -n "$SLURM_ACCOUNT" ]] && SB_ARGS+=(--account "$SLURM_ACCOUNT")
    [[ -n "$SLURM_PARTITION" ]] && SB_ARGS+=(--partition "$SLURM_PARTITION")
    [[ -n "$MEM" ]] && SB_ARGS+=(--mem "$MEM")
    # Container is entered via srun inside the job (see below), NOT via sbatch.
    # --export ALL forwards CONTAINER_IMAGE/MOUNTS/RAM_USE_CONTAINER intact.
    export RAM_SCRIPT_DIR="$SCRIPT_DIR"
    export CONTAINER_IMAGE CONTAINER_MOUNTS RAM_USE_CONTAINER CONTAINER_MINICONDA RAM_ARRAY_COUNT
    # shellcheck disable=SC2206
    [[ -n "$EXTRA_SBATCH" ]] && SB_ARGS+=($EXTRA_SBATCH)
    echo "Submitting: sbatch ${SB_ARGS[*]} $0"
    exec sbatch "${SB_ARGS[@]}" "$0"
fi

# Inside the batch allocation: enter the pyxis container once via srun, then
# re-exec this script with IN_CONTAINER=1 so this block is skipped on pass 2.
if [[ "${RAM_USE_CONTAINER:-0}" == "1" && -n "${CONTAINER_IMAGE:-}" \
      && "${IN_CONTAINER:-0}" != "1" && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Entering container: $CONTAINER_IMAGE"
    exec srun --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$CONTAINER_MOUNTS" \
        bash -lc "IN_CONTAINER=1 RAM_SCRIPT_DIR='$SCRIPT_DIR' MINICONDA_DIR='${CONTAINER_MINICONDA}' CLUSTER_BASE='$CLUSTER_BASE' RAM_ARRAY_COUNT='${RAM_ARRAY_COUNT}' SLURM_ARRAY_TASK_ID='${SLURM_ARRAY_TASK_ID:-}' bash '$SCRIPT_DIR/run_david_ai_mfa_ram_session_cluster.sh'"
fi

mkdir -p "$WORK_DIR/logs"

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
    local env_python="$miniconda/envs/$env_name/bin/python"
    if [[ -x "$env_python" ]]; then
        export MINICONDA_DIR="$miniconda"
        export MFA_ENV="$miniconda/envs/$env_name"
        export PYTHON="$MFA_ENV/bin/python"
        export PATH="$MFA_ENV/bin:$PATH"
        log "Using existing conda env: $PYTHON"
        return 0
    fi
    if [[ ! -f "$CONDA_ENV_TARBALL" ]]; then
        log "ERROR: packed conda env not found: $CONDA_ENV_TARBALL"
        exit 1
    fi
    log "Extracting packed conda env to node-local $miniconda/envs/$env_name"
    rm -rf "$miniconda/envs/$env_name"
    mkdir -p "$miniconda/envs/$env_name"
    tar -xzf "$CONDA_ENV_TARBALL" -C "$miniconda/envs/$env_name"
    if [[ -x "$miniconda/envs/$env_name/bin/conda-unpack" ]]; then
        "$miniconda/envs/$env_name/bin/conda-unpack" || true
    fi
    if [[ ! -x "$env_python" ]]; then
        log "ERROR: extracted env missing python: $env_python"
        exit 1
    fi
    export MINICONDA_DIR="$miniconda"
    export MFA_ENV="$miniconda/envs/$env_name"
    export PYTHON="$MFA_ENV/bin/python"
    export PATH="$MFA_ENV/bin:$PATH"
    log "Using node-local conda env: $PYTHON"
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
    if "$py" -c "import cosmos_xenna, pandas, pyarrow, lhotse" 2>/dev/null; then
        return 0
    fi
    log "Installing runtime deps into $py"
    "$py" -m pip install -q cosmos-xenna pandas pyarrow lhotse num2words || log "WARN: dep install failed"
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

install_unglue_repairs() {
    local src="${UNGLUE_REPAIRS_SRC:-$SCRIPT_DIR/workdir/lexicon/unglue_repairs_heuristic.tsv}"
    export LEXICON_DIR="${LEXICON_DIR:-$LINK_WORK_DIR/lexicon}"
    mkdir -p "$LEXICON_DIR"
    if [[ -f "$src" ]]; then
        cp -f "$src" "$LEXICON_DIR/unglue_repairs_heuristic.tsv"
        cp -f "$src" "$LEXICON_DIR/unglue_repairs.tsv"
        log "Installed $(wc -l < "$LEXICON_DIR/unglue_repairs.tsv") unglue repairs -> $LEXICON_DIR/unglue_repairs.tsv"
    elif [[ -f "$LEXICON_DIR/unglue_repairs.tsv" || -f "$LEXICON_DIR/unglue_repairs_heuristic.tsv" ]]; then
        log "Using existing unglue repairs in $LEXICON_DIR"
    else
        log "WARN: no unglue repairs in $LEXICON_DIR (copy with copy_unglue_repairs_to_draco.sh)"
    fi
}

install_unglue_repairs

ALLOC_CPUS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-$CPUS}}"
if [[ -z "${MFA_NUM_JOBS:-}" ]]; then
    MFA_NUM_JOBS=$(( ALLOC_CPUS >= 4 ? 4 : ALLOC_CPUS ))
    [[ "$MFA_NUM_JOBS" -lt 1 ]] && MFA_NUM_JOBS=1
fi
if [[ -z "${WORKERS:-}" ]]; then
    WORKERS=$(( ALLOC_CPUS / MFA_NUM_JOBS ))
    [[ "$WORKERS" -lt 1 ]] && WORKERS=1
fi

log "CLUSTER RAM SESSION START (job=${SLURM_JOB_ID:-local})"
log "DATA_ROOT=$DATA_ROOT (existing data_links, no link stage)"
log "WORK_DIR=$WORK_DIR LINK_WORK_DIR=$LINK_WORK_DIR"
log "ALLOC_CPUS=$ALLOC_CPUS WORKERS=$WORKERS MFA_NUM_JOBS=$MFA_NUM_JOBS"

if [[ ! -d "$DATA_ROOT" ]]; then
    log "ERROR: data_links not found: $DATA_ROOT"
    log "Run the old cluster link stage first, or set DATA_ROOT to an existing symlink tree."
    exit 1
fi

link_count=$(find "$DATA_ROOT" -maxdepth 1 -type l 2>/dev/null | wc -l)
log "Using $link_count session symlinks under $DATA_ROOT"

# Shard config for multi-node array runs. In an array job each task owns one
# shard (index = SLURM_ARRAY_TASK_ID). Per-shard stage-done markers avoid tasks
# overwriting each other's completion flag. Lhotse merging is deferred to a
# final single run (see note below), so array tasks never race the global files.
if [[ "$RAM_ARRAY_COUNT" -gt 1 ]]; then
    export SHARD_COUNT="$RAM_ARRAY_COUNT"
    export SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
    export STAGE_DONE_NAME="ram_session_pipeline_shard${SHARD_INDEX}"
    export MERGE_LHOTSE="${MERGE_LHOTSE:-0}"
    log "SHARD $SHARD_INDEX/$SHARD_COUNT (stage-done=$STAGE_DONE_NAME, merge=$MERGE_LHOTSE)"
else
    export SHARD_COUNT=1 SHARD_INDEX=0 STAGE_DONE_NAME="ram_session_pipeline"
    export MERGE_LHOTSE="${MERGE_LHOTSE:-1}"
fi

export DATA_ROOT WORK_DIR RAM_DIR="${RAM_DIR:-/dev/shm/david_ai_ram_session}"
export LEXICON_DIR WORKERS MFA_NUM_JOBS FORCE="${FORCE:-0}" SESSION="${SESSION:-}"
export SKIP_LEXICON="${SKIP_LEXICON:-1}" MFA_G2P="${MFA_G2P:-english_us_arpa}" MFA_DICT_NAME="${MFA_DICT_NAME:-english_us_arpa}"
export SHARD_COUNT SHARD_INDEX STAGE_DONE_NAME

bash "$SCRIPT_DIR/run_david_ai_mfa_ram_session.sh"
log "CLUSTER RAM SESSION DONE"
