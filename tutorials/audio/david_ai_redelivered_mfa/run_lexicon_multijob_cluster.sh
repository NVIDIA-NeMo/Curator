#!/bin/bash
# Multi-job MFA lexicon build on draco:
#   1) punct array  - parallel punctuation repair (re-normalize from text_raw)
#                     runs INSIDE a pyxis container (no conda install)
#   2) prepare      - unglue repair map + frequency unglue + G2P shards (conda)
#   3) g2p array    - parallel mfa g2p per shard (SLURM array, conda + MFA)
#   4) merge        - combine shard pronunciations into final dictionary (conda)
#
# Submit from login node:
#   REPAIR_SHARDS=32 G2P_SHARDS=16 bash run_lexicon_multijob_cluster.sh
#
# Disable the container for punct repair (fall back to conda):
#   PUNCT_USE_CONTAINER=0 bash run_lexicon_multijob_cluster.sh
#
# Or run one phase inside an allocation:
#   LEXICON_MODE=punct_repair LOCAL_RUN=1 SLURM_ARRAY_TASK_ID=0 bash run_lexicon_multijob_cluster.sh
#   LEXICON_MODE=prepare LOCAL_RUN=1 bash run_lexicon_multijob_cluster.sh
#   LEXICON_MODE=g2p LOCAL_RUN=1 SLURM_ARRAY_TASK_ID=0 bash run_lexicon_multijob_cluster.sh

set -euo pipefail

# When SLURM runs an array/batch task it executes a copy of this script from
# the job spool dir, so ${BASH_SOURCE[0]} no longer sits next to the sibling
# .py files. submit_chain exports LEXICON_SCRIPT_DIR so tasks find them.
SCRIPT_DIR="${LEXICON_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

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
CPUS="${CPUS:-16}"
MEM="${MEM:-}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"

REPAIR_SHARDS="${REPAIR_SHARDS:-32}"
G2P_SHARDS="${G2P_SHARDS:-16}"
NUM2WORDS_LANG="${NUM2WORDS_LANG:-en}"
UNGLUE_MAX_FREQ="${UNGLUE_MAX_FREQ:-5}"
USE_MANIFESTS="${USE_MANIFESTS:-1}"
LEXICON_MODE="${LEXICON_MODE:-submit}"

# Punctuation repair runs inside a pyxis container (no per-node conda install).
# The container ships python3 + num2words; nemo_curator comes from the package
# tarball extracted to node-local /tmp, and a tiny cosmos_xenna shim satisfies
# the nemo_curator import. MFA is NOT in the container, so g2p/merge stay on conda.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/users/pzelasko/containers/nemo-25.11-pytorch2.9-automodel-23apr26.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12}"
CONTAINER_SHIM="${CONTAINER_SHIM:-$CLUSTER_BASE/container_pylibs}"
PUNCT_USE_CONTAINER="${PUNCT_USE_CONTAINER:-1}"
PUNCT_ARRAY_THROTTLE="${PUNCT_ARRAY_THROTTLE:-4}"
PUNCT_CPUS="${PUNCT_CPUS:-2}"
PUNCT_MEM="${PUNCT_MEM:-32G}"
# Optional: submit only a subset of punct shards (e.g. "2,5,6,8-23"); empty = 0..N-1.
PUNCT_ARRAY="${PUNCT_ARRAY:-}"
# Optional: submit only the punct array and skip prepare/g2p/merge (for retries).
PUNCT_ONLY="${PUNCT_ONLY:-0}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Install/activate the conda env on this node. Guarded by a per-node flock so
# concurrent array tasks on the same node don't collide during the install.
_provision_conda() {
    local miniconda="$1" env_name="$2"
    if [[ -x "$miniconda/envs/$env_name/bin/python" ]]; then
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
    pip install -q lhotse textgrid num2words hydra-core omegaconf soundfile tqdm pyloudnorm praatio \
        cosmos-xenna pandas pyarrow
    log "conda install done: $miniconda/envs/$env_name"
}

ensure_conda_on_node() {
    local miniconda="${MINICONDA_DIR:-/tmp/miniconda3_tt}"
    local env_name="${CONDA_ENV:-curator_pain_1}"
    if [[ ! -x "$miniconda/envs/$env_name/bin/python" ]]; then
        mkdir -p "$(dirname "$miniconda")"
        (
            flock 9
            _provision_conda "$miniconda" "$env_name"
        ) 9>"${miniconda}.lock"
    fi
    export MINICONDA_DIR="$miniconda"
    export MFA_ENV="$miniconda/envs/$env_name"
    export PYTHON="$MFA_ENV/bin/python"
    export PATH="$MFA_ENV/bin:$PATH"
}

# Race-safe extraction of a tarball into a destination shared across array
# tasks on the same node. Uses flock so only one task extracts; the rest wait
# and reuse the result. $3 is a sentinel file that marks a completed extract.
extract_once() {
    local tarball="$1" dest="$2" sentinel="$3"
    [[ -f "$tarball" ]] || return 0
    mkdir -p "$dest"
    local lock="${dest%/}.lock"
    (
        flock 9
        if [[ ! -f "$sentinel" ]]; then
            rm -rf "$dest"
            mkdir -p "$dest"
            tar -xzf "$tarball" -C "$dest"
            touch "$sentinel"
        fi
    ) 9>"$lock"
}

setup_env() {
    # First (optional) arg: "mfa" if this phase needs the MFA acoustic models.
    local need_mfa="${1:-}"

    DRACO_ENV="${DRACO_ENV:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/david_ai_mfa_env.sh}"
    if [[ -f "$DRACO_ENV" ]]; then
        set +e
        # shellcheck source=/dev/null
        source "$DRACO_ENV"
        set -e
    fi

    ensure_conda_on_node

    local tarball="${CURATOR_PKG_TARBALL:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/nemo_curator_pkg.tar.gz}"
    local dest=/tmp/curator_pkg
    if [[ ! -f "$dest/nemo_curator/__init__.py" ]]; then
        extract_once "$tarball" "$dest" "$dest/.extract.done"
    fi
    if [[ -f "$dest/nemo_curator/__init__.py" ]]; then
        export PYTHONPATH="$dest${PYTHONPATH:+:$PYTHONPATH}"
    fi

    if [[ "$need_mfa" == "mfa" ]]; then
        local dict="${MFA_ROOT_DIR:-}/pretrained_models/dictionary/english_us_arpa.dict"
        if [[ ! -f "$dict" ]]; then
            local mfa_tar="${MFA_MODELS_TARBALL:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/MFA_models_draco.tar.gz}"
            if [[ -f "$mfa_tar" ]]; then
                # Tarball has a top-level MFA_models/ dir -> extract with -C /tmp.
                local sentinel=/tmp/MFA_models/.extract.done
                (
                    flock 9
                    if [[ ! -f "$sentinel" ]]; then
                        rm -rf /tmp/MFA_models
                        tar -xzf "$mfa_tar" -C /tmp
                        touch "$sentinel"
                    fi
                ) 9>/tmp/MFA_models.lock
                export MFA_ROOT_DIR=/tmp/MFA_models
            fi
        fi
    fi
}

input_args() {
    if [[ "$USE_MANIFESTS" == "1" ]]; then
        INPUT_ARGS=(--manifests-dir "$MANIFESTS_DIR")
    else
        INPUT_ARGS=(--data-root "$DATA_ROOT")
    fi
}

submit_chain() {
    mkdir -p "$WORK_DIR/logs" "$LEXICON_DIR/repair_shards"
    local punct_log="$WORK_DIR/logs/${JOB_NAME}_punct_%A_%a.out"
    local prep_log="$WORK_DIR/logs/${JOB_NAME}_prepare_%j.out"
    local g2p_log="$WORK_DIR/logs/${JOB_NAME}_g2p_%A_%a.out"
    local merge_log="$WORK_DIR/logs/${JOB_NAME}_merge_%j.out"

    local repair_max=$((REPAIR_SHARDS - 1))
    local punct_export="ALL,LEXICON_MODE=punct_repair,LEXICON_SCRIPT_DIR=$SCRIPT_DIR,REPAIR_SHARDS=$REPAIR_SHARDS,USE_MANIFESTS=$USE_MANIFESTS,MANIFESTS_DIR=$MANIFESTS_DIR,LEXICON_DIR=$LEXICON_DIR,WORK_DIR=$WORK_DIR,NUM2WORDS_LANG=$NUM2WORDS_LANG"
    local prep_export="ALL,LEXICON_MODE=prepare,LEXICON_SCRIPT_DIR=$SCRIPT_DIR,G2P_SHARDS=$G2P_SHARDS,USE_MANIFESTS=$USE_MANIFESTS,MANIFESTS_DIR=$MANIFESTS_DIR,LEXICON_DIR=$LEXICON_DIR,WORK_DIR=$WORK_DIR,UNGLUE_MAX_FREQ=$UNGLUE_MAX_FREQ,NUM2WORDS_LANG=$NUM2WORDS_LANG"

    # Pyxis --container-image is an srun option, not an sbatch one. We submit a
    # normal batch job and launch the container via srun inside run_punct_repair.
    # NOTE: CONTAINER_MOUNTS contains commas, which clash with sbatch --export's
    # separator, so we pass it via a ';'-encoded var (decoded in the job) instead.
    if [[ "$PUNCT_USE_CONTAINER" == "1" && -n "$CONTAINER_IMAGE" ]]; then
        local mounts_enc="${CONTAINER_MOUNTS//,/;}"
        punct_export="$punct_export,IN_CONTAINER=1,CONTAINER_IMAGE=$CONTAINER_IMAGE,CONTAINER_MOUNTS_ENC=$mounts_enc,CONTAINER_SHIM=$CONTAINER_SHIM,CLUSTER_BASE=$CLUSTER_BASE"
        log "punct_repair will run in container: $CONTAINER_IMAGE"
    fi

    local punct_mem_args=()
    if [[ -n "$PUNCT_MEM" ]]; then
        punct_mem_args=(--mem "$PUNCT_MEM")
    fi
    local punct_array="${PUNCT_ARRAY:-0-${repair_max}}"
    if [[ -n "$PUNCT_ARRAY_THROTTLE" && "$PUNCT_ARRAY_THROTTLE" -gt 0 ]]; then
        punct_array="${punct_array}%${PUNCT_ARRAY_THROTTLE}"
    fi

    local punct_job
    punct_job=$(sbatch --parsable \
        --job-name "${JOB_NAME}_punct" \
        --array="$punct_array" \
        --nodes 1 --ntasks 1 --cpus-per-task "$PUNCT_CPUS" \
        --time "$TIME_LIMIT" --output "$punct_log" \
        --export "$punct_export" \
        ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
        ${SLURM_PARTITION:+--partition "$SLURM_PARTITION"} \
        ${punct_mem_args[@]+"${punct_mem_args[@]}"} \
        ${MEM:+--mem "$MEM"} \
        "$0")

    if [[ "$PUNCT_ONLY" == "1" ]]; then
        log "Submitted punct-only array: punct=$punct_job (array=$punct_array)"
        return 0
    fi

    local prep_job
    prep_job=$(sbatch --parsable \
        --job-name "${JOB_NAME}_prepare" \
        --dependency "afterok:${punct_job}" \
        --nodes 1 --ntasks 1 --cpus-per-task "$CPUS" \
        --time "$TIME_LIMIT" --output "$prep_log" \
        --export "$prep_export" \
        ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
        ${SLURM_PARTITION:+--partition "$SLURM_PARTITION"} \
        ${MEM:+--mem "$MEM"} \
        "$0")

    local array_max=$((G2P_SHARDS - 1))
    local g2p_job
    g2p_job=$(sbatch --parsable \
        --job-name "${JOB_NAME}_g2p" \
        --dependency "afterok:${prep_job}" \
        --array="0-${array_max}" \
        --nodes 1 --ntasks 1 --cpus-per-task 4 \
        --time "$TIME_LIMIT" --output "$g2p_log" \
        --export "ALL,LEXICON_MODE=g2p,LEXICON_SCRIPT_DIR=$SCRIPT_DIR,G2P_SHARDS=$G2P_SHARDS,LEXICON_DIR=$LEXICON_DIR,WORK_DIR=$WORK_DIR" \
        ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
        ${SLURM_PARTITION:+--partition "$SLURM_PARTITION"} \
        ${MEM:+--mem "$MEM"} \
        "$0")

    local merge_job
    merge_job=$(sbatch --parsable \
        --job-name "${JOB_NAME}_merge" \
        --dependency "afterok:${g2p_job}" \
        --nodes 1 --ntasks 1 --cpus-per-task 2 \
        --time "01:00:00" --output "$merge_log" \
        --export "ALL,LEXICON_MODE=merge,LEXICON_SCRIPT_DIR=$SCRIPT_DIR,LEXICON_DIR=$LEXICON_DIR,WORK_DIR=$WORK_DIR" \
        ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
        ${SLURM_PARTITION:+--partition "$SLURM_PARTITION"} \
        "$0")

    log "Submitted lexicon multi-job chain:"
    log "  punct_array=$punct_job (0-${repair_max})  prepare=$prep_job  g2p_array=$g2p_job (0-${array_max})  merge=$merge_job"
}

run_punct_repair_in_container() {
    local shard="$1"
    local tarball="${CURATOR_PKG_TARBALL:-$CLUSTER_BASE/nemo_curator_pkg.tar.gz}"
    # Decode ';'-encoded mounts (see submit_chain) back to comma-separated.
    local mounts="${CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12}"
    if [[ -n "${CONTAINER_MOUNTS_ENC:-}" ]]; then
        mounts="${CONTAINER_MOUNTS_ENC//;/,}"
    fi
    # Inner script runs inside the pyxis container. To avoid sustained small-file
    # I/O against the (congested) lustre manifests dir, we:
    #   1) list this shard's files, 2) bulk-copy them to node-local /tmp with a
    #   single sequential tar stream, 3) repair locally on /tmp, 4) bulk-copy
    #   only the changed files back with one tar stream, 5) write the .done marker.
    local stage="/tmp/punct_stage_${shard}"
    local inner
    inner=$(cat <<INNER
set -euo pipefail
if [[ ! -f /tmp/curator_pkg/nemo_curator/__init__.py ]]; then
    rm -rf /tmp/curator_pkg
    mkdir -p /tmp/curator_pkg
    tar -xzf "$tarball" -C /tmp/curator_pkg
fi
export PYTHONPATH="/tmp/curator_pkg:$CONTAINER_SHIM"

STAGE="$stage"
FILELIST=\$STAGE/shard_files.txt
CHANGED=\$STAGE/changed.txt
STAGE_IN=\$STAGE/in
rm -rf "\$STAGE"
mkdir -p "\$STAGE_IN"

echo "[stage] listing shard $shard files"
python3 "$SCRIPT_DIR/repair_glued_oov_manifests.py" \
    --manifests-dir "$MANIFESTS_DIR" \
    --lexicon-dir "$LEXICON_DIR" \
    --repair-mode punctuation \
    --shard-count "$REPAIR_SHARDS" \
    --shard-index "$shard" \
    --list-shard-files > "\$FILELIST"

NFILES=\$(wc -l < "\$FILELIST" | tr -d ' ')
echo "[stage] shard $shard has \$NFILES files; bulk-copying lustre -> \$STAGE_IN"
if [[ "\$NFILES" -gt 0 ]]; then
    tar -C "$MANIFESTS_DIR" -cf - -T "\$FILELIST" | tar -C "\$STAGE_IN" -xf -
fi

echo "[stage] repairing locally on \$STAGE_IN"
python3 "$SCRIPT_DIR/repair_glued_oov_manifests.py" \
    --manifests-dir "\$STAGE_IN" \
    --lexicon-dir "$LEXICON_DIR" \
    --num2words-lang "$NUM2WORDS_LANG" \
    --repair-mode punctuation \
    --changed-list "\$CHANGED" \
    --skip-done-marker

NCHG=0
[[ -f "\$CHANGED" ]] && NCHG=\$(wc -l < "\$CHANGED" | tr -d ' ')
echo "[stage] \$NCHG files changed; copying back to lustre"
if [[ "\$NCHG" -gt 0 ]]; then
    tar -C "\$STAGE_IN" -cf - -T "\$CHANGED" | tar -C "$MANIFESTS_DIR" -xf -
fi

mkdir -p "$LEXICON_DIR/repair_shards"
printf 'ok\n' > "$LEXICON_DIR/repair_shards/punctuation_shard_\$(printf '%03d' $shard).done"
rm -rf "\$STAGE"
echo "[stage] shard $shard complete"
INNER
)
    srun --container-image="$CONTAINER_IMAGE" --container-mounts="$mounts" \
        bash -lc "$inner"
}

run_punct_repair() {
    local shard="${SLURM_ARRAY_TASK_ID:-0}"
    log "PUNCT REPAIR SHARD $shard/$REPAIR_SHARDS START (job=${SLURM_JOB_ID:-local})"
    if [[ "${IN_CONTAINER:-0}" == "1" ]]; then
        run_punct_repair_in_container "$shard"
    else
        setup_env
        "$PYTHON" "$SCRIPT_DIR/repair_glued_oov_manifests.py" \
            --manifests-dir "$MANIFESTS_DIR" \
            --lexicon-dir "$LEXICON_DIR" \
            --num2words-lang "$NUM2WORDS_LANG" \
            --repair-mode punctuation \
            --shard-count "$REPAIR_SHARDS" \
            --shard-index "$shard"
    fi
    log "PUNCT REPAIR SHARD $shard DONE"
}

run_prepare() {
    setup_env
    mkdir -p "$LEXICON_DIR" "$WORK_DIR/logs"
    input_args
    log "LEXICON PREPARE START (job=${SLURM_JOB_ID:-local})"
    log "MANIFESTS_DIR=$MANIFESTS_DIR LEXICON_DIR=$LEXICON_DIR G2P_SHARDS=$G2P_SHARDS"

    log "Pass 1: build unglue repair map from punctuation-repaired text_norm"
    "$PYTHON" "$SCRIPT_DIR/stage0_build_lexicon.py" \
        "${INPUT_ARGS[@]}" \
        --lexicon-dir "$LEXICON_DIR" \
        --num2words-lang "$NUM2WORDS_LANG" \
        --unglue-max-freq "$UNGLUE_MAX_FREQ" \
        --skip-g2p \
        --g2p-shard-count 0 \
        --no-renormalize-from-raw

    if [[ -f "$LEXICON_DIR/unglue_repairs.tsv" ]]; then
        log "Pass 2: frequency unglue repair on manifests"
        "$PYTHON" "$SCRIPT_DIR/repair_glued_oov_manifests.py" \
            --manifests-dir "$MANIFESTS_DIR" \
            --lexicon-dir "$LEXICON_DIR" \
            --num2words-lang "$NUM2WORDS_LANG" \
            --repair-mode unglue
    fi

    log "Pass 3: final OOV list + $G2P_SHARDS G2P shards"
    "$PYTHON" "$SCRIPT_DIR/stage0_build_lexicon.py" \
        "${INPUT_ARGS[@]}" \
        --lexicon-dir "$LEXICON_DIR" \
        --num2words-lang "$NUM2WORDS_LANG" \
        --unglue-max-freq "$UNGLUE_MAX_FREQ" \
        --skip-g2p \
        --g2p-shard-count "$G2P_SHARDS" \
        --no-renormalize-from-raw

    log "LEXICON PREPARE DONE"
}

run_g2p() {
    setup_env mfa
    local shard="${SLURM_ARRAY_TASK_ID:-0}"
    log "LEXICON G2P SHARD $shard START (job=${SLURM_JOB_ID:-local})"
    "$PYTHON" "$SCRIPT_DIR/stage0_build_lexicon.py" \
        --lexicon-dir "$LEXICON_DIR" \
        --g2p-shard-index "$shard"
    log "LEXICON G2P SHARD $shard DONE"
}

run_merge() {
    setup_env mfa
    log "LEXICON MERGE START (job=${SLURM_JOB_ID:-local})"
    "$PYTHON" "$SCRIPT_DIR/stage0_build_lexicon.py" \
        --lexicon-dir "$LEXICON_DIR" \
        --merge-g2p-only
    log "LEXICON MERGE DONE -> $LEXICON_DIR/english_mfa_davidai_eng.dict"
}

if [[ "$LEXICON_MODE" == "submit" && -z "${SLURM_JOB_ID:-}" && "${LOCAL_RUN:-0}" != "1" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch not found. Use LOCAL_RUN=1 or set LEXICON_MODE=punct_repair|prepare|g2p|merge" >&2
        exit 1
    fi
    submit_chain
    exit 0
fi

case "$LEXICON_MODE" in
    punct_repair) run_punct_repair ;;
    prepare) run_prepare ;;
    g2p) run_g2p ;;
    merge) run_merge ;;
    *)
        echo "Unknown LEXICON_MODE=$LEXICON_MODE (use submit|punct_repair|prepare|g2p|merge)" >&2
        exit 1
        ;;
esac
