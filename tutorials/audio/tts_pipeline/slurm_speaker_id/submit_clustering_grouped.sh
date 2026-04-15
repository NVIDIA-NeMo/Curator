#!/bin/bash
# Submit global clustering in groups of GROUP_SIZE shards.
# Each group runs independently with global clustering within the group.
#
# Required environment variables:
#   MANIFESTS_DIR  - base path to granary-filtered manifests
#   WORK_DIR       - working directory for embeddings, logs, output
#   CURATOR_DIR    - path to Curator repo checkout
#   CONTAINER      - path to squashfs container image
#   ACCOUNT        - Slurm account name

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

MANIFESTS_DIR="${MANIFESTS_DIR:?Set MANIFESTS_DIR}"
WORK_DIR="${WORK_DIR:?Set WORK_DIR}"
CURATOR_DIR="${CURATOR_DIR:?Set CURATOR_DIR to Curator repo path}"
CONTAINER="${CONTAINER:?Set CONTAINER}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT}"
GROUP_SIZE="${GROUP_SIZE:-64}"  # ~100K utterances per group

# NAME|MANIFEST_SUB|NUM_SHARDS
CORPORA=(
    "yodas_0fc_ru|yodas/0_from_captions/ru|256"
    "yodas_0bw_ru|yodas/0_by_whisper/ru|1024"
    "yodas_1bw_ru|yodas/1_by_whisper/ru|8192"
)

for corpus_def in "${CORPORA[@]}"; do
    IFS='|' read -r NAME MANIFEST_SUB NUM_SHARDS <<< "$corpus_def"

    MANIFEST_DIR="${MANIFESTS_DIR}/${MANIFEST_SUB}"
    EMB_DIR="${WORK_DIR}/embeddings_full/${MANIFEST_SUB}"
    OUT_DIR="${WORK_DIR}/output_manifests_global/${MANIFEST_SUB}"
    LOG_DIR="${WORK_DIR}/logs/${NAME}"
    mkdir -p "${LOG_DIR}" "${OUT_DIR}"

    # Check embeddings complete
    DONE=$(ls "${EMB_DIR}"/embeddings_*.npz 2>/dev/null | wc -l)
    if [ "$DONE" -lt "$NUM_SHARDS" ]; then
        echo "=== ${NAME}: SKIP (${DONE}/${NUM_SHARDS} embeddings) ==="
        echo
        continue
    fi

    # Split into groups
    GROUP_IDX=0
    START=0
    while [ $START -lt $NUM_SHARDS ]; do
        END=$((START + GROUP_SIZE - 1))
        [ $END -ge $NUM_SHARDS ] && END=$((NUM_SHARDS - 1))

        JOB_SCRIPT="${WORK_DIR}/logs/${NAME}_cluster_g${GROUP_IDX}.sbatch"
        cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=gclust_${NAME}_g${GROUP_IDX}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch_singlenode
#SBATCH --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=30 --mem=228G
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/cluster_g${GROUP_IDX}.out
#SBATCH --error=${LOG_DIR}/cluster_g${GROUP_IDX}.err

srun --container-image=${CONTAINER} \\
     --container-mounts=/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw \\
     --container-writable \\
     bash -c "export PYTHONPATH=${CURATOR_DIR}:\\\$PYTHONPATH && python ${CURATOR_DIR}/tutorials/audio/speaker_id/run_pipeline.py \\
    --cluster \\
    --input_manifest '${MANIFEST_DIR}/shard_{${START}..${END}}.jsonl' \\
    --embedding_dir ${EMB_DIR} \\
    --output_manifest_dir ${OUT_DIR} \\
    --threshold 0.292 \\
    --embedding_normalization center_global \\
    --linkage_method average"
EOF

        echo "  ${NAME} group ${GROUP_IDX}: shards ${START}-${END}"
        if [ "$DRY_RUN" = true ]; then
            echo "    [DRY RUN]"
        else
            sbatch "${JOB_SCRIPT}" 2>&1 | sed 's/^/    /'
        fi

        START=$((END + 1))
        GROUP_IDX=$((GROUP_IDX + 1))
    done
    echo "=== ${NAME}: ${GROUP_IDX} groups of ${GROUP_SIZE} shards ==="
    echo
done
