#!/bin/bash
# Submit CPU-only speaker clustering jobs (one per corpus).
# Run AFTER all embedding extraction jobs complete.
#
# Usage:
#   bash submit_clustering.sh [--dry-run]
#   bash submit_clustering.sh [--dry-run] [corpus_name]  # e.g. ytc_ru

set -euo pipefail

DRY_RUN=false
FILTER=""
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
    [[ "$arg" != "--dry-run" ]] && FILTER="$arg"
done

MANIFESTS_DIR="/lustre/fs11/portfolios/convai/projects/convai_convaird_nemo-speech/users/ameister/TTS_Granary/granary_filtered"
WORK_DIR="/lustre/fs11/portfolios/convai/projects/convai_convaird_nemo-speech/users/gzelenfroind/speaker_id"
CURATOR_DIR="/lustre/fs11/portfolios/convai/projects/convai_convaird_nemo-speech/users/gzelenfroind/Curator"
CONTAINER="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ytc2/nemo_dev_20240717_aistore.sqsh"
ACCOUNT="convai_convaird_nemo-speech"
PARTITION="cpu_short"

mkdir -p "${WORK_DIR}/logs"

CORPORA=(
    "ytc_ru|ytc/ru|64"
    "yodas_0fc_ru|yodas/0_from_captions/ru|256"
    "yodas_0bw_ru|yodas/0_by_whisper/ru|1024"
    "yodas_1bw_ru|yodas/1_by_whisper/ru|8192"
)

for corpus_def in "${CORPORA[@]}"; do
    IFS='|' read -r NAME MANIFEST_SUB NUM_SHARDS <<< "$corpus_def"

    # Filter if specified
    [[ -n "$FILTER" && "$NAME" != "$FILTER" ]] && continue

    MANIFEST_DIR="${MANIFESTS_DIR}/${MANIFEST_SUB}"
    EMB_DIR="${WORK_DIR}/embeddings/${MANIFEST_SUB}"
    OUT_DIR="${WORK_DIR}/output_manifests/${MANIFEST_SUB}"
    LOG_DIR="${WORK_DIR}/logs/${NAME}"
    MAX_IDX=$((NUM_SHARDS - 1))
    mkdir -p "${LOG_DIR}"

    # Check completeness
    DONE=$(ls "${EMB_DIR}"/embeddings_*.npz 2>/dev/null | wc -l)
    if [ "$DONE" -lt "$NUM_SHARDS" ]; then
        echo "=== ${NAME}: SKIP (${DONE}/${NUM_SHARDS} embeddings done) ==="
        echo
        continue
    fi

    CLUSTER_PY="${WORK_DIR}/scripts/run_clustering.py"
    JOB_SCRIPT="${WORK_DIR}/logs/${NAME}_cluster.sbatch"
    cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=spkclust_${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/cluster.out
#SBATCH --error=${LOG_DIR}/cluster.err

srun --container-image=${CONTAINER} \
     --container-mounts="/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw" \
     --container-writable \
     python ${CLUSTER_PY} \
    --manifest_dir "${MANIFEST_DIR}" \
    --embedding_dir "${EMB_DIR}" \
    --output_dir "${OUT_DIR}" \
    --threshold 0.292
EOF

    echo "=== ${NAME}: clustering ${NUM_SHARDS} shards ==="
    echo "    Embeddings: ${EMB_DIR} (${DONE}/${NUM_SHARDS})"
    echo "    Output:     ${OUT_DIR}"

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] Would submit: sbatch ${JOB_SCRIPT}"
    else
        JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
        echo "    Submitted: job ${JOB_ID}"
    fi
    echo
done
