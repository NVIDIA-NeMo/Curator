#!/bin/bash
# Cluster speaker embeddings (AHC) and write annotated manifests.
#
# Reads:
#   - Original NeMo JSONL manifests (same ones used for embedding extraction)
#   - Per-shard embeddings_*.npz files from run_speaker_id_titanet.sh
#
# Writes:
#   - New manifests in output_manifests/ with speaker_label + confidence_score
#     added to each JSON line.
#
# CPU-only — no GPU required.
#
# Usage:
#   bash run_speaker_clustering.sh
#   bash run_speaker_clustering.sh --threshold 0.38   # stricter: fewer false merges
#   bash run_speaker_clustering.sh --shard_level_clustering
#
# Defaults match Curator clustering stage + run_pipeline.py:
#   --embedding_normalization center_global  (subtract batch mean before cosine)
#   --threshold 0.292  (TitaNet + mean row: cosine @ EER from local Vox1-O subset;
#                     see tutorials/audio/speaker_id/TITANET_VS_WESPKResNet_benchmark.md)

set -euo pipefail

CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-0.292}"
EMBEDDING_NORM="${EMBEDDING_NORM:-center_global}"

# ── Paths ──────────────────────────────────────────────────────────
branch_name="sot_mt_asr_rt"
NEMO_ROOT="/home/taejinp/projects/${branch_name}/NeMo"
CURATOR_ROOT="/home/taejinp/projects/curator-speaker-id-annote/Curator"
export PYTHONPATH="${NEMO_ROOT}:${CURATOR_ROOT}:${PYTHONPATH:-}"

SCRIPT="${CURATOR_ROOT}/tutorials/audio/speaker_id/run_pipeline.py"

BASE_DIR="/disk_a_nvd/datasets/LS_PnC_Concatenated/full_no_pnc"
MANIFEST_PREFIX="${BASE_DIR}/sharded_manifests_Canary_style/manifest"
SHARD_START=0
SHARD_END=49
INPUT_MANIFEST="${MANIFEST_PREFIX}__OP_${SHARD_START}..${SHARD_END}_CL_.json"

EMBEDDING_DIR="/home/taejinp/projects/curator-speaker-id-annote/output"
OUTPUT_MANIFEST_DIR="/home/taejinp/projects/curator-speaker-id-annote/output_manifests"

# ── Parse arguments ───────────────────────────────────────────────
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --embedding_dir)        EMBEDDING_DIR="$2"; shift 2 ;;
        --output_manifest_dir)  OUTPUT_MANIFEST_DIR="$2"; shift 2 ;;
        --threshold)            CLUSTER_THRESHOLD="$2"; shift 2 ;;
        --embedding_normalization) EMBEDDING_NORM="$2"; shift 2 ;;
        *)                      PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

NUM_FILES=$(ls -1 "${EMBEDDING_DIR}"/embeddings_*.npz 2>/dev/null | wc -l)

echo "============================================"
echo "Speaker Clustering (AHC)"
echo "============================================"
echo "Input manifest : ${INPUT_MANIFEST}"
echo "Embedding dir  : ${EMBEDDING_DIR} (${NUM_FILES} npz files)"
echo "Output dir     : ${OUTPUT_MANIFEST_DIR}"
echo "Threshold      : ${CLUSTER_THRESHOLD}  (cosine; TitaNet+mean @ EER default)"
echo "Embedding norm : ${EMBEDDING_NORM}"
echo "Pass-thru      : ${PASSTHROUGH_ARGS[*]+"${PASSTHROUGH_ARGS[*]}"}"
echo "============================================"

python "${SCRIPT}" \
    --cluster \
    --input_manifest "${INPUT_MANIFEST}" \
    --embedding_dir "${EMBEDDING_DIR}" \
    --output_manifest_dir "${OUTPUT_MANIFEST_DIR}" \
    --threshold "${CLUSTER_THRESHOLD}" \
    --embedding_normalization "${EMBEDDING_NORM}" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}

echo ""
echo "============================================"
echo "Clustering complete."
echo "Annotated manifests:"
echo "  ${OUTPUT_MANIFEST_DIR}/"
ls -1 "${OUTPUT_MANIFEST_DIR}"/*.json 2>/dev/null | head -5
NUM_OUT=$(ls -1 "${OUTPUT_MANIFEST_DIR}"/*.json 2>/dev/null | wc -l)
if [ "${NUM_OUT}" -gt 5 ]; then
    echo "  ... (${NUM_OUT} files total)"
fi
echo ""
echo "Each manifest line now has: speaker_label, confidence_score"
echo "============================================"
