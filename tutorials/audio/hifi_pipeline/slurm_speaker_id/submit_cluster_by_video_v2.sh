#!/bin/bash
# Submit per-video speaker clustering v2 jobs.
#
# Flow per corpus:
#   1. Read source manifests + embeddings
#   2. GroupByVideoStage: resolve video_id from manifest 'id' field
#   3. Cluster per video_id (AHC, cosine, center_global)
#   4. Write to output_manifests_by_video_v2/
#
# Usage:
#   bash submit_cluster_by_video_v2.sh [--dry-run] [corpus_name]

set -euo pipefail

DRY_RUN=false
FILTER=""
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
    [[ "$arg" != "--dry-run" ]] && FILTER="$arg"
done

GRANARY="${GRANARY:?Set GRANARY to your granary_filtered directory}"
WORK_DIR="${WORK_DIR:?Set WORK_DIR to your working directory}"
CURATOR_DIR="${CURATOR_DIR:?Set CURATOR_DIR to your NeMo Curator checkout}"
CONTAINER="${CONTAINER:?Set CONTAINER to your nemo container .sqsh path}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Slurm account}"
SCRIPT="${WORK_DIR}/scripts/cluster_by_video_v2.py"

# NAME|MANIFEST_SUB|NUM_SHARDS|PARTITION|MEM|TIME
CORPORA=(
    "ytc_ru|ytc/ru|64|cpu_short|64G|01:00:00"
    "yodas_0fc_ru|yodas/0_from_captions/ru|256|cpu_short|128G|02:00:00"
    "yodas_0bw_ru|yodas/0_by_whisper/ru|1024|cpu_long|128G|04:00:00"
    "yodas_1bw_ru|yodas/1_by_whisper/ru|8192|cpu_long|128G|08:00:00"
)

for corpus_def in "${CORPORA[@]}"; do
    IFS='|' read -r NAME MANIFEST_SUB NUM_SHARDS PARTITION MEM TIME <<< "$corpus_def"

    [[ -n "$FILTER" && "$NAME" != "$FILTER" ]] && continue

    MANIFEST_DIR="${GRANARY}/${MANIFEST_SUB}"
    EMB_DIR="${WORK_DIR}/embeddings/${MANIFEST_SUB}"
    OUT_DIR="${WORK_DIR}/output_manifests_by_video_v2/${MANIFEST_SUB}"
    LOG_DIR="${WORK_DIR}/logs/${NAME}"
    mkdir -p "${LOG_DIR}"

    # Check embeddings complete
    DONE=$(ls "${EMB_DIR}"/embeddings_*.npz 2>/dev/null | wc -l)
    if [ "$DONE" -lt "$NUM_SHARDS" ]; then
        echo "=== ${NAME}: SKIP (${DONE}/${NUM_SHARDS} embeddings) ==="
        echo
        continue
    fi

    JOB_SCRIPT="${WORK_DIR}/logs/${NAME}_cluster_video_v2.sbatch"
    cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=vclust2_${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/cluster_video_v2.out
#SBATCH --error=${LOG_DIR}/cluster_video_v2.err
#SBATCH --export=ALL,PYTHONPATH=${CURATOR_DIR}

srun --container-image=${CONTAINER} \
     --container-mounts="${CONTAINER_MOUNTS:-/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw}" \
     --container-writable \
     python ${SCRIPT} \
        --manifest_dir "${MANIFEST_DIR}" \
        --embedding_dir "${EMB_DIR}" \
        --output_dir "${OUT_DIR}" \
        --threshold 0.292
EOF

    echo "=== ${NAME}: per-video clustering v2 (${NUM_SHARDS} shards) ==="
    echo "    Manifests:   ${MANIFEST_DIR}"
    echo "    Embeddings:  ${EMB_DIR} (${DONE}/${NUM_SHARDS})"
    echo "    Output:      ${OUT_DIR}"

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] Would submit: sbatch ${JOB_SCRIPT}"
    else
        JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
        echo "    Submitted: job ${JOB_ID}"
    fi
    echo
done
