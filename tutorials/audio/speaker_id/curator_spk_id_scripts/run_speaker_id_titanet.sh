#!/bin/bash
# Extract speaker embeddings from NeMo-tarred data using multiple GPUs.
#
# Saves one .npz per manifest/tar shard (e.g. 50 shards → 50 npz files).
# Embeddings are stored **raw** (no cohort mean / whitening here).  Mean
# subtraction for AHC is done in clustering: run_speaker_clustering.sh uses
# ``center_global`` + threshold 0.292 by default (same TitaNet model).
#
# Next step (same ``embeddings_*.npz`` dir):
#   bash run_speaker_clustering.sh
#   # or:  --embedding_dir /path  if you override OUTPUT_DIR below
#
# Usage:
#   bash run_speaker_id_titanet.sh --devices 0,1
#   bash run_speaker_id_titanet.sh --devices 0,1 --output_dir /path/to/embeddings
#   bash run_speaker_id_titanet.sh --devices 0,1 --batch_size 128
#   bash run_speaker_id_titanet.sh --devices 0,1 --max_cuts 100
#   bash run_speaker_id_titanet.sh --devices 0              # single GPU
#
# Env: OUTPUT_DIR (default matches run_speaker_clustering.sh EMBEDDING_DIR)

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────
branch_name="sot_mt_asr_rt"
NEMO_ROOT="/home/taejinp/projects/${branch_name}/NeMo"
CURATOR_ROOT="/home/taejinp/projects/curator-speaker-id-annote/Curator"
export PYTHONPATH="${NEMO_ROOT}:${CURATOR_ROOT}:${PYTHONPATH:-}"

BASE_DIR="/disk_a_nvd/datasets/LS_PnC_Concatenated/full_no_pnc"
MANIFEST_PREFIX="${BASE_DIR}/sharded_manifests_Canary_style/manifest"
TAR_PREFIX="${BASE_DIR}/audio"
# Default aligned with curator_spk_id_scripts/run_speaker_clustering.sh (EMBEDDING_DIR)
OUTPUT_DIR="${OUTPUT_DIR:-/home/taejinp/projects/curator-speaker-id-annote/output}"
SCRIPT="${CURATOR_ROOT}/tutorials/audio/speaker_id/run_pipeline.py"
MODEL="nvidia/speakerverification_en_titanet_large"
SHARD_START=0
SHARD_END=49

# ── Parse arguments ───────────────────────────────────────────────
DEVICES="0,1"
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices)     DEVICES="$2"; shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        *)             PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

IFS=',' read -ra GPU_LIST <<< "${DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
TOTAL_SHARDS=$((SHARD_END - SHARD_START + 1))

echo "============================================"
echo "Speaker Embedding Extraction"
echo "============================================"
echo "Devices    : ${DEVICES} (${NUM_GPUS} GPUs)"
echo "Shards     : ${SHARD_START}..${SHARD_END} (${TOTAL_SHARDS} total)"
echo "Output dir : ${OUTPUT_DIR}"
echo "Pass-thru  : ${PASSTHROUGH_ARGS[*]+"${PASSTHROUGH_ARGS[*]}"}"
echo "============================================"

# ── Split shards across GPUs & launch ─────────────────────────────
SHARDS_PER_GPU=$((TOTAL_SHARDS / NUM_GPUS))
REMAINDER=$((TOTAL_SHARDS % NUM_GPUS))
offset=${SHARD_START}

pids=()

for i in "${!GPU_LIST[@]}"; do
    GPU="${GPU_LIST[$i]}"
    COUNT=${SHARDS_PER_GPU}
    if [ "$i" -lt "${REMAINDER}" ]; then
        COUNT=$((COUNT + 1))
    fi
    GPU_SHARD_START=${offset}
    GPU_SHARD_END=$((offset + COUNT - 1))
    offset=$((offset + COUNT))

    SHARD_MANIFEST="${MANIFEST_PREFIX}__OP_${GPU_SHARD_START}..${GPU_SHARD_END}_CL_.json"
    SHARD_TAR="${TAR_PREFIX}__OP_${GPU_SHARD_START}..${GPU_SHARD_END}_CL_.tar"

    echo "[GPU ${GPU}]  shards ${GPU_SHARD_START}..${GPU_SHARD_END}  (${COUNT} shards)"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    python "${SCRIPT}" \
        --direct \
        --input_manifest "${SHARD_MANIFEST}" \
        --input_tar "${SHARD_TAR}" \
        --lhotse_mode nemo_tarred \
        --output_dir "${OUTPUT_DIR}" \
        --output_format npz \
        --model_name "${MODEL}" \
        ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"} &
    pids+=($!)
done

echo ""
echo "Launched ${NUM_GPUS} workers.  PIDs: ${pids[*]}"
echo "============================================"

# ── Wait for all workers ──────────────────────────────────────────
FAIL=0
for pid in "${pids[@]}"; do
    wait "${pid}" || FAIL=$((FAIL + 1))
done

if [ "${FAIL}" -gt 0 ]; then
    echo "ERROR: ${FAIL} GPU worker(s) failed."
    exit 1
fi

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "All workers finished."
echo "Per-shard embedding files:"
echo "  ${OUTPUT_DIR}/embeddings_*.npz"
echo ""
ls -1 "${OUTPUT_DIR}"/embeddings_*.npz 2>/dev/null | head -5
NUM_FILES=$(ls -1 "${OUTPUT_DIR}"/embeddings_*.npz 2>/dev/null | wc -l)
if [ "${NUM_FILES}" -gt 5 ]; then
    echo "  ... (${NUM_FILES} files total)"
fi
echo "============================================"
