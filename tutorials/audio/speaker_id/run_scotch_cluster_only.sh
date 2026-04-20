#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Cluster-only launcher for SCOTCH-v1.large_scale (CS-OCI-ORD / DRACO-OCI-IAD).
#
# Inputs:
#   * embeddings_merged.npz        produced by run_pipeline.py --direct + --merge
#   * raw_sharded_manifests/       per-shard input manifests
#
# Outputs (in $OUTPUT_DIR):
#   * manifest_<sid>.json          input manifest + speaker_label + confidence_score
#   * clusters_summary.jsonl       flat one-row-per-utterance index
#   * cluster_config.json          SCOTCH sidecar (PARAM_TUNE.md 3)
#
# Usage:
#   ./run_scotch_cluster_only.sh                     # uses cluster defaults
#   PRESET=librispeech-2026-04 \
#       ./run_scotch_cluster_only.sh                 # explicit preset
#   THRESHOLD=0.55 MIN_CLUSTER_SIZE=10 \
#       ./run_scotch_cluster_only.sh                 # override individual knobs
#
# Cluster auto-detection
# ----------------------
#   * CS-OCI-ORD    (hostnames containing "cs-oci")
#       DATA_ROOT = /lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train
#   * DRACO-OCI-IAD (hostnames containing "draco-oci")
#       DATA_ROOT = /lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train
#
# Override with CLUSTER=cs-oci-ord|draco-oci-iad or by exporting DATA_ROOT
# directly.  See PARAM_TUNE.md for the full SCOTCH parameter rationale and
# the analysis plots backing the librispeech-2026-04 preset.

set -euo pipefail

# --------------------------------------------------------------------------
# Cluster + path resolution
# --------------------------------------------------------------------------

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
            echo "       This launcher targets CS-OCI-ORD or DRACO-OCI-IAD." >&2
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

# Per-cluster work / scratch.  Override with WORK_DIR=/path/to/scratch.
WORK_DIR="${WORK_DIR:-/tmp/${USER}/scotch_librispeech}"

# Where the merged embeddings live.  By default we re-use the layout
# emitted by run_tune_threshold_librispeech.sh so you can chain:
#   ./run_tune_threshold_librispeech.sh extract merge
#   ./run_scotch_cluster_only.sh
MERGED_NPZ="${MERGED_NPZ:-${WORK_DIR}/embeddings/embeddings_merged.npz}"

# Where the per-shard input manifests live (one manifest_<sid>.json per
# audio_<sid>.tar).  Built once from tarred_audio_manifest.json --
# see run_tune_threshold_librispeech.sh for the recipe.
MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/raw_sharded_manifests}"

# Where to write annotated manifests + cluster_config.json + summary.
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/scotch_speaker_clustering_results}"

# SCOTCH preset.  Defined in
# nemo_curator.stages.audio.speaker_id.clustering.cluster_config.PRESETS.
# librispeech-2026-04 is the calibration documented in PARAM_TUNE.md.
PRESET="${PRESET:-librispeech-2026-04}"

# Optional per-knob overrides.  Leave blank to use the preset's value.
THRESHOLD="${THRESHOLD:-}"
LINKAGE="${LINKAGE:-}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-}"
BIRCH_COSINE_FLOOR="${BIRCH_COSINE_FLOOR:-}"
BRANCHING_FACTOR="${BRANCHING_FACTOR:-}"
PARTIAL_FIT_BATCH="${PARTIAL_FIT_BATCH:-}"
ASSIGN_TILE="${ASSIGN_TILE:-}"
EMBEDDING_NORMALIZATION="${EMBEDDING_NORMALIZATION:-}"
NO_CONFIDENCE="${NO_CONFIDENCE:-0}"  # set to 1 to skip confidence scoring

# --------------------------------------------------------------------------
# Curator + Python resolution (same scheme as run_tune_threshold_librispeech.sh)
# --------------------------------------------------------------------------

if [[ -z "${CURATOR_REPO:-}" ]]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [[ "${_here}" != "/" && ! -d "${_here}/nemo_curator" ]]; do
        _here="$(dirname "${_here}")"
    done
    if [[ -d "${_here}/nemo_curator" ]]; then
        CURATOR_REPO="${_here}"
    fi
fi

PYTHON="${PYTHON:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCOTCH_PY="${SCRIPT_DIR}/scotch_cluster_and_annotate.py"

if [[ ! -f "${SCOTCH_PY}" ]]; then
    echo "ERROR: required script not found: ${SCOTCH_PY}" >&2
    exit 1
fi

if [[ -z "${CURATOR_REPO:-}" || ! -d "${CURATOR_REPO}/nemo_curator" ]]; then
    echo "ERROR: could not locate the Curator source tree." >&2
    echo "       Set CURATOR_REPO=/path/to/Curator (a directory containing nemo_curator/)." >&2
    exit 1
fi

# Make `import nemo_curator` work without installing the package.
export PYTHONPATH="${CURATOR_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------

if [[ ! -f "${MERGED_NPZ}" ]]; then
    echo "ERROR: merged embeddings not found: ${MERGED_NPZ}" >&2
    echo "       Run extract + merge first, e.g.:" >&2
    echo "         ./run_tune_threshold_librispeech.sh extract merge" >&2
    echo "       or override with MERGED_NPZ=/path/to/embeddings_merged.npz" >&2
    exit 1
fi

if [[ ! -d "${MANIFEST_DIR}" ]]; then
    echo "ERROR: manifest dir not found: ${MANIFEST_DIR}" >&2
    echo "       Override with MANIFEST_DIR=/path/to/raw_sharded_manifests" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# --------------------------------------------------------------------------
# Build argv with optional overrides
# --------------------------------------------------------------------------

declare -a override_args=()
[[ -n "${THRESHOLD}"               ]] && override_args+=(--threshold              "${THRESHOLD}")
[[ -n "${LINKAGE}"                 ]] && override_args+=(--linkage                "${LINKAGE}")
[[ -n "${MIN_CLUSTER_SIZE}"        ]] && override_args+=(--min_cluster_size       "${MIN_CLUSTER_SIZE}")
[[ -n "${BIRCH_COSINE_FLOOR}"      ]] && override_args+=(--birch_cosine_floor     "${BIRCH_COSINE_FLOOR}")
[[ -n "${BRANCHING_FACTOR}"        ]] && override_args+=(--branching_factor       "${BRANCHING_FACTOR}")
[[ -n "${PARTIAL_FIT_BATCH}"       ]] && override_args+=(--partial_fit_batch      "${PARTIAL_FIT_BATCH}")
[[ -n "${ASSIGN_TILE}"             ]] && override_args+=(--assign_tile            "${ASSIGN_TILE}")
[[ -n "${EMBEDDING_NORMALIZATION}" ]] && override_args+=(--embedding_normalization "${EMBEDDING_NORMALIZATION}")
[[ "${NO_CONFIDENCE}" == "1"       ]] && override_args+=(--no_confidence)

# --------------------------------------------------------------------------
# Print + run
# --------------------------------------------------------------------------

cat <<EOF
==========================================================================
  SCOTCH-v1.large_scale.${PRESET}  --  cluster + annotate manifests
--------------------------------------------------------------------------
  DATA_ROOT       : ${DATA_ROOT}
  MERGED_NPZ      : ${MERGED_NPZ}
  MANIFEST_DIR    : ${MANIFEST_DIR}
  OUTPUT_DIR      : ${OUTPUT_DIR}
  PRESET          : ${PRESET}
  CURATOR_REPO    : ${CURATOR_REPO}
  PYTHON          : ${PYTHON}
  Overrides       : ${override_args[*]:-<none>}
==========================================================================
EOF

"${PYTHON}" "${SCOTCH_PY}" \
    --merged_npz   "${MERGED_NPZ}" \
    --manifest_dir "${MANIFEST_DIR}" \
    --output_dir   "${OUTPUT_DIR}" \
    --preset       "${PRESET}" \
    "${override_args[@]}"

echo
echo "Done.  Outputs:"
echo "  annotated manifests : ${OUTPUT_DIR}/manifest_<sid>.json"
echo "  flat summary        : ${OUTPUT_DIR}/clusters_summary.jsonl"
echo "  SCOTCH sidecar      : ${OUTPUT_DIR}/cluster_config.json"
