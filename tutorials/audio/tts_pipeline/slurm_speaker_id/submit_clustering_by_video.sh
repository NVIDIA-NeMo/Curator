#!/bin/bash
# Submit per-video speaker clustering jobs.
#
# Required environment variables:
#   MANIFESTS_DIR  - base path to granary-filtered manifests
#   WORK_DIR       - working directory for embeddings, logs, output
#   CONTAINER      - path to squashfs container image
#   ACCOUNT        - Slurm account name

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

MANIFESTS_DIR="${MANIFESTS_DIR:?Set MANIFESTS_DIR}"
WORK_DIR="${WORK_DIR:?Set WORK_DIR}"
CONTAINER="${CONTAINER:?Set CONTAINER}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT}"
CLUSTER_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cluster_by_video.py"

CORPORA=(
    "ytc_ru|ytc/ru|64|cpu_short|64G|01:00:00"
    "yodas_0fc_ru|yodas/0_from_captions/ru|256|cpu_short|128G|02:00:00"
    "yodas_0bw_ru|yodas/0_by_whisper/ru|1024|cpu_long|128G|04:00:00"
    "yodas_1bw_ru|yodas/1_by_whisper/ru|8192|cpu_long|128G|08:00:00"
)

for corpus_def in "${CORPORA[@]}"; do
    IFS='|' read -r NAME MANIFEST_SUB NUM_SHARDS PARTITION MEM TIME <<< "$corpus_def"

    MANIFEST_DIR="${MANIFESTS_DIR}/${MANIFEST_SUB}"
    EMB_DIR="${WORK_DIR}/embeddings/${MANIFEST_SUB}"
    OUT_DIR="${WORK_DIR}/output_manifests_by_video/${MANIFEST_SUB}"
    LOG_DIR="${WORK_DIR}/logs/${NAME}"
    mkdir -p "${LOG_DIR}"

    DONE=$(ls "${EMB_DIR}"/embeddings_*.npz 2>/dev/null | wc -l)
    if [ "$DONE" -lt "$NUM_SHARDS" ]; then
        echo "=== ${NAME}: SKIP (${DONE}/${NUM_SHARDS} embeddings) ==="
        continue
    fi

    JOB_SCRIPT="${WORK_DIR}/logs/${NAME}_cluster_video.sbatch"
    cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=vclust_${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/cluster_video.out
#SBATCH --error=${LOG_DIR}/cluster_video.err

srun --container-image=${CONTAINER} \
     --container-mounts="/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw" \
     --container-writable \
     python ${CLUSTER_PY} \
    --manifest_dir "${MANIFEST_DIR}" \
    --embedding_dir "${EMB_DIR}" \
    --output_dir "${OUT_DIR}" \
    --threshold 0.292
EOF

    echo "=== ${NAME}: per-video clustering ${NUM_SHARDS} shards ==="
    echo "    Output: ${OUT_DIR}"

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] Would submit: sbatch ${JOB_SCRIPT}"
    else
        sbatch "${JOB_SCRIPT}" 2>&1
    fi
    echo
done
