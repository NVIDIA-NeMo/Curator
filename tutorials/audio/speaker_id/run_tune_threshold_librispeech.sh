#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Launcher for tune_threshold_librispeech.py on a 2-GPU box.
#
# Phases:
#   1. extract  -- run two `run_pipeline.py --direct` processes in parallel,
#                  one pinned to each GPU, each over half of the
#                  manifest/tar shard range.  Both write into the same
#                  $WORK/embeddings/ folder (per-shard files don't collide
#                  because each shard becomes embeddings_<shard_id>.npz).
#   2. merge    -- a single CPU `--merge` call over $WORK/embeddings/ to
#                  produce embeddings_merged.npz.  We invoke run_pipeline.py
#                  directly so the merge step doesn't trigger any GPU init.
#   3. labels   -- CPU.  Joins manifest speaker labels with the merged
#                  embeddings.
#   4. tune     -- CPU.  2-D sweep over (AHC threshold) x (BIRCH cosine
#                  floor) and reports the best.
#
# Usage:
#   ./run_tune_threshold_librispeech.sh                # run all phases
#   ./run_tune_threshold_librispeech.sh extract        # only the GPU phase
#   ./run_tune_threshold_librispeech.sh merge          # only the merge step
#   ./run_tune_threshold_librispeech.sh labels         # only the join phase
#   ./run_tune_threshold_librispeech.sh tune           # only the sweep
#   ./run_tune_threshold_librispeech.sh extract merge  # multiple phases
#
# Override any default via env var, e.g.:
#   WORK_DIR=/scratch/lstune NUM_SHARDS=512 BATCH_SIZE=128 \
#   AHC_RANGE=0.20,0.50,0.01 BIRCH_RANGE=0.75,0.95,0.05 \
#       ./run_tune_threshold_librispeech.sh
#
# Cluster auto-detection
# ----------------------
# This script targets two NVIDIA clusters:
#
#   * CS-OCI-ORD    (hostnames containing "cs-oci")
#       DATA_ROOT = /lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train
#   * DRACO-OCI-IAD (hostnames containing "draco-oci")
#       DATA_ROOT = /lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train
#
# The cluster is picked by substring-matching the local hostname.  Override
# with CLUSTER=cs-oci-ord|draco-oci-iad or by exporting DATA_ROOT directly.

set -euo pipefail

# --------------------------------------------------------------------------
# Config (override via env)
# --------------------------------------------------------------------------

# LibriSpeech tarred dataset.
#
# raw_sharded_manifests/ holds the per-shard manifests whose audio_filepath
# values are exactly the tar member names of the matching audio_<sid>.tar.
# Built once on each cluster from tarred_audio_manifest.json via:
#
#   python -c "import json,os; \
#     IN='${DATA_ROOT}/tarred_audio_manifest.json'; \
#     OUT='${MANIFEST_DIR}'; os.makedirs(OUT, exist_ok=True); \
#     hs={}; \
#     [hs.setdefault(json.loads(l)['shard_id'], \
#         open(os.path.join(OUT,f'manifest_{json.loads(l)[\"shard_id\"]}.json'),'w')) \
#      .write(l) for l in open(IN)]"

# Resolve DATA_ROOT from CLUSTER (env var) or hostname auto-detect.
_CS_OCI_ORD_ROOT="/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train"
_DRACO_OCI_IAD_ROOT="/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train"

if [[ -z "${DATA_ROOT:-}" ]]; then
    _CLUSTER="${CLUSTER:-}"
    if [[ -z "${_CLUSTER}" ]]; then
        _HOST="$(hostname -f 2>/dev/null || hostname)"
        case "${_HOST}" in
            *cs-oci*)    _CLUSTER="cs-oci-ord"   ;;
            *draco-oci*) _CLUSTER="draco-oci-iad" ;;
        esac
    fi
    case "${_CLUSTER}" in
        cs-oci-ord)
            DATA_ROOT="${_CS_OCI_ORD_ROOT}"
            ;;
        draco-oci-iad)
            DATA_ROOT="${_DRACO_OCI_IAD_ROOT}"
            ;;
        "")
            echo "ERROR: could not auto-detect cluster from hostname '$(hostname -f 2>/dev/null || hostname)'." >&2
            echo "       This launcher is intended to run on CS-OCI-ORD or DRACO-OCI-IAD." >&2
            echo "       Set CLUSTER=cs-oci-ord or CLUSTER=draco-oci-iad, or export DATA_ROOT=/path/to/tarred_train." >&2
            exit 1
            ;;
        *)
            echo "ERROR: unknown CLUSTER='${_CLUSTER}'.  Use cs-oci-ord or draco-oci-iad, or export DATA_ROOT directly." >&2
            exit 1
            ;;
    esac
    echo "[config] auto-detected cluster: ${_CLUSTER}"
fi

MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/raw_sharded_manifests}"
MANIFEST_PREFIX="${MANIFEST_PREFIX:-manifest_}"
TAR_PREFIX="${TAR_PREFIX:-audio_}"
NUM_SHARDS="${NUM_SHARDS:-512}"          # total shards (0..NUM_SHARDS-1)

# Output / scratch.  Defaults to a per-user scratch dir on the cluster; both
# CS-OCI-ORD and DRACO-OCI-IAD provide /lustre/.../scratch.  Override with
# WORK_DIR=/path/to/dir.
WORK_DIR="${WORK_DIR:-/tmp/${USER}/librispeech_threshold_tune}"

# GPU plan
GPUS="${GPUS:-0,1}"                       # comma-separated CUDA device ids
BATCH_SIZE="${BATCH_SIZE:-128}"           # per-GPU inference batch
MODEL_NAME="${MODEL_NAME:-nvidia/speakerverification_en_titanet_large}"
MAX_CUTS="${MAX_CUTS:-}"                  # blank => use everything

# Sweep ranges (passed verbatim to --ahc_thresholds / --birch_thresholds)
AHC_RANGE="${AHC_RANGE:-0.20,0.50,0.01}"
BIRCH_RANGE="${BIRCH_RANGE:-0.75,0.95,0.05}"
LINKAGE="${LINKAGE:-average}"
EMB_NORM="${EMB_NORM:-center_global}"     # center_global | none

# Path to the Curator source tree (used as PYTHONPATH so `import nemo_curator`
# works without a pip install).  Auto-detect by walking up from this script
# until we find a directory containing nemo_curator/.  Override by exporting
# CURATOR_REPO=/path/to/Curator before running.
if [[ -z "${CURATOR_REPO:-}" ]]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [[ "${_here}" != "/" && ! -d "${_here}/nemo_curator" ]]; do
        _here="$(dirname "${_here}")"
    done
    if [[ -d "${_here}/nemo_curator" ]]; then
        CURATOR_REPO="${_here}"
    fi
fi

# Python interpreter.  Must be Python >= 3.10 with nemo_toolkit + lhotse +
# scikit-learn + torch installed (e.g. an `nemo` conda/venv on the cluster).
# Override by exporting PYTHON=/path/to/python before running.
PYTHON="${PYTHON:-python3}"

# --------------------------------------------------------------------------
# Resolve script paths
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_PY="${SCRIPT_DIR}/tune_threshold_librispeech.py"
RUN_PIPELINE_PY="${SCRIPT_DIR}/run_pipeline.py"

for f in "${TUNE_PY}" "${RUN_PIPELINE_PY}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required script not found: ${f}" >&2
        exit 1
    fi
done

if [[ -z "${CURATOR_REPO:-}" || ! -d "${CURATOR_REPO}/nemo_curator" ]]; then
    echo "ERROR: could not locate the Curator source tree." >&2
    echo "       Set CURATOR_REPO=/path/to/Curator (a directory containing nemo_curator/)." >&2
    exit 1
fi

# Make `import nemo_curator` work without installing the package.
export PYTHONPATH="${CURATOR_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

# One-time sanity check: can the chosen Python actually import the GPU stage?
if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1
from nemo_curator.stages.audio.speaker_id.speaker_embedding_lhotse import SpeakerEmbeddingLhotseStage  # noqa: F401
PY
then
    echo "ERROR: '${PYTHON}' cannot import nemo_curator's SpeakerEmbeddingLhotseStage." >&2
    echo "       PYTHONPATH=${PYTHONPATH}" >&2
    echo "       Activate a Python>=3.10 env with nemo_toolkit + lhotse installed and re-run:" >&2
    echo "         PYTHON=\$(which python) ./run_tune_threshold_librispeech.sh" >&2
    "${PYTHON}" -c "from nemo_curator.stages.audio.speaker_id.speaker_embedding_lhotse import SpeakerEmbeddingLhotseStage" || true
    exit 1
fi

# --------------------------------------------------------------------------
# Shard range helpers
# --------------------------------------------------------------------------

# Build a NeMo brace-pattern for shards $1..$2 inclusive.
nemo_pattern() {
    local prefix="$1" suffix="$2" lo="$3" hi="$4"
    if [[ "${lo}" == "${hi}" ]]; then
        printf '%s/%s%s%s' "${MANIFEST_DIR%%/}" "${prefix}" "${lo}" "${suffix}"
    else
        printf '%s/%s_OP_%s..%s_CL_%s' "${MANIFEST_DIR%%/}" "${prefix}" "${lo}" "${hi}" "${suffix}"
    fi
}

manifest_pattern() {  # $1=lo $2=hi
    nemo_pattern "${MANIFEST_PREFIX}" ".json" "$1" "$2"
}
tar_pattern() {       # $1=lo $2=hi
    # tar files live next to the manifests' parent (DATA_ROOT), not in MANIFEST_DIR
    local lo="$1" hi="$2"
    if [[ "${lo}" == "${hi}" ]]; then
        printf '%s/%s%s.tar' "${DATA_ROOT%%/}" "${TAR_PREFIX}" "${lo}"
    else
        printf '%s/%s_OP_%s..%s_CL_.tar' "${DATA_ROOT%%/}" "${TAR_PREFIX}" "${lo}" "${hi}"
    fi
}

full_manifest_pattern() {
    manifest_pattern 0 "$((NUM_SHARDS - 1))"
}

# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------

EMB_DIR="${WORK_DIR}/embeddings"
LOG_DIR="${WORK_DIR}/logs"
MERGED_NPZ="${EMB_DIR}/embeddings_merged.npz"
LABELS_NPZ="${WORK_DIR}/labels.npz"

phase_extract() {
    mkdir -p "${EMB_DIR}" "${LOG_DIR}"

    # Split GPUs into an array
    IFS=',' read -ra GPU_ARR <<< "${GPUS}"
    local n_gpu="${#GPU_ARR[@]}"
    if (( n_gpu < 1 )); then
        echo "ERROR: GPUS env var is empty" >&2
        exit 1
    fi

    # Split shards across GPUs as evenly as possible.
    # Partition [0, NUM_SHARDS) into n_gpu contiguous ranges.
    local total="${NUM_SHARDS}"
    local base=$(( total / n_gpu ))
    local extra=$(( total % n_gpu ))   # first $extra GPUs get one extra shard

    local pids=()
    local start=0
    local i
    for (( i=0; i<n_gpu; i++ )); do
        local count=$(( base + (i < extra ? 1 : 0) ))
        if (( count <= 0 )); then continue; fi
        local end=$(( start + count - 1 ))
        local gpu="${GPU_ARR[i]}"

        local mp tp
        mp="$(manifest_pattern "${start}" "${end}")"
        tp="$(tar_pattern      "${start}" "${end}")"

        local log="${LOG_DIR}/extract_gpu${gpu}_shards${start}-${end}.log"
        echo "[extract] GPU ${gpu}: shards ${start}..${end} (${count} shards)"
        echo "          manifest: ${mp}"
        echo "          tar     : ${tp}"
        echo "          log     : ${log}"

        local extra_args=()
        if [[ -n "${MAX_CUTS}" ]]; then
            extra_args+=(--max_cuts "${MAX_CUTS}")
        fi

        # Each GPU runs its own `extract` subcommand.  We bypass the wrapper's
        # own merge step here so the two GPUs don't both try to merge in
        # parallel; we run a single merge in phase_merge after both finish.
        # The wrapper's `extract` subcommand only invokes
        # run_pipeline.py --direct internally if the merged file is missing,
        # so to avoid that race we call run_pipeline.py --direct directly.
        CUDA_VISIBLE_DEVICES="${gpu}" \
        nohup "${PYTHON}" "${RUN_PIPELINE_PY}" \
            --direct \
            --input_manifest "${mp}" \
            --input_tar      "${tp}" \
            --lhotse_mode    nemo_tarred \
            --output_dir     "${EMB_DIR}" \
            --batch_size     "${BATCH_SIZE}" \
            --model_name     "${MODEL_NAME}" \
            "${extra_args[@]}" \
            > "${log}" 2>&1 &
        pids+=("$!")
        start=$(( end + 1 ))
    done

    echo "[extract] Launched ${#pids[@]} GPU workers (pids=${pids[*]}). Waiting..."

    # Wait for all GPU workers; fail loudly if any non-zero.
    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "[extract] ERROR: worker pid ${pid} failed" >&2
            fail=1
        fi
    done
    if (( fail )); then
        echo "[extract] One or more GPU workers failed; see ${LOG_DIR}/" >&2
        exit 1
    fi

    # Sanity check: number of shard files written.
    local n_emb
    n_emb=$(find "${EMB_DIR}" -maxdepth 1 -name 'embeddings_*.npz' \
            -not -name 'embeddings_merged.npz' | wc -l)
    echo "[extract] Done. ${n_emb} per-shard .npz files in ${EMB_DIR}"
}

phase_merge() {
    if [[ -f "${MERGED_NPZ}" ]]; then
        echo "[merge] ${MERGED_NPZ} already exists, skipping (delete to force)."
        return
    fi
    mkdir -p "${LOG_DIR}"
    local log="${LOG_DIR}/merge.log"
    echo "[merge] -> ${MERGED_NPZ} (log: ${log})"
    "${PYTHON}" "${RUN_PIPELINE_PY}" \
        --merge \
        --output_dir "${EMB_DIR}" \
        2>&1 | tee "${log}"
    if [[ ! -f "${MERGED_NPZ}" ]]; then
        echo "[merge] ERROR: merged file not produced" >&2
        exit 1
    fi
}

phase_labels() {
    mkdir -p "${LOG_DIR}"
    local log="${LOG_DIR}/labels.log"
    echo "[labels] manifest: $(full_manifest_pattern)"
    echo "[labels] log     : ${log}"
    "${PYTHON}" "${TUNE_PY}" labels \
        --work_dir       "${WORK_DIR}" \
        --manifest_glob  "$(full_manifest_pattern)" \
        2>&1 | tee "${log}"
    if [[ ! -f "${LABELS_NPZ}" ]]; then
        echo "[labels] ERROR: labels.npz not produced" >&2
        exit 1
    fi
}

phase_tune() {
    mkdir -p "${LOG_DIR}"
    local log="${LOG_DIR}/tune.log"
    echo "[tune] AHC range  : ${AHC_RANGE}"
    echo "[tune] BIRCH range: ${BIRCH_RANGE}"
    echo "[tune] linkage    : ${LINKAGE}, embedding_norm: ${EMB_NORM}"
    echo "[tune] log        : ${log}"
    "${PYTHON}" "${TUNE_PY}" tune \
        --work_dir                  "${WORK_DIR}" \
        --ahc_thresholds            "${AHC_RANGE}" \
        --birch_thresholds          "${BIRCH_RANGE}" \
        --linkage                   "${LINKAGE}" \
        --embedding_normalization   "${EMB_NORM}" \
        2>&1 | tee "${log}"
}

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

print_config() {
    cat <<EOF
==========================================================================
  tune_threshold_librispeech launcher
--------------------------------------------------------------------------
  DATA_ROOT       : ${DATA_ROOT}
  MANIFEST_DIR    : ${MANIFEST_DIR}
  NUM_SHARDS      : ${NUM_SHARDS}
  WORK_DIR        : ${WORK_DIR}
  GPUS            : ${GPUS}
  BATCH_SIZE      : ${BATCH_SIZE}
  MODEL_NAME      : ${MODEL_NAME}
  MAX_CUTS        : ${MAX_CUTS:-<all>}
  AHC_RANGE       : ${AHC_RANGE}
  BIRCH_RANGE     : ${BIRCH_RANGE}
  LINKAGE         : ${LINKAGE}
  EMB_NORM        : ${EMB_NORM}
  CURATOR_REPO    : ${CURATOR_REPO}
  PYTHON          : ${PYTHON}
==========================================================================
EOF
}

main() {
    print_config

    local phases=("$@")
    if [[ "${#phases[@]}" -eq 0 ]]; then
        phases=(extract merge labels tune)
    fi

    local p
    for p in "${phases[@]}"; do
        case "${p}" in
            extract) phase_extract ;;
            merge)   phase_merge   ;;
            labels)  phase_labels  ;;
            tune)    phase_tune    ;;
            all)
                phase_extract
                phase_merge
                phase_labels
                phase_tune
                ;;
            *)
                echo "ERROR: unknown phase '${p}'  (allowed: extract|merge|labels|tune|all)" >&2
                exit 1
                ;;
        esac
    done

    echo
    echo "Done. Best threshold + per-cell metrics:"
    echo "  ${WORK_DIR}/best_threshold.json"
    echo "  ${WORK_DIR}/tuning_results.csv"
}

main "$@"
