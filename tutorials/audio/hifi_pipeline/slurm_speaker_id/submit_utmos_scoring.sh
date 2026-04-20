#!/bin/bash
# Submit UTMOSv2 scoring jobs — 1 GPU per shard, skip already-scored.
#
# Usage:
#   bash submit_utmos_scoring.sh [--dry-run] [corpus_name]

set -euo pipefail

DRY_RUN=false
FILTER=""
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
    [[ "$arg" != "--dry-run" ]] && FILTER="$arg"
done

WORK_DIR="${WORK_DIR:?Set WORK_DIR to your working directory}"
SCORE_PY="${SCORE_PY:-${WORK_DIR}/scripts/score_utmos_shard.py}"
CONTAINER="${CONTAINER:?Set CONTAINER to your curator-utmos.sqsh path}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Slurm account}"
PARTITION="${PARTITION:-batch_singlenode}"

mkdir -p "${WORK_DIR}/logs"

# NAME|MANIFEST_SUB|TAR_PATTERN|NUM_SHARDS
CORPORA=(
    "ytc_ru|ytc/ru|s3://YTC/ru/webds_ru/audio_{SHARD}.tar|64"
    "yodas_0fc_ru|yodas/0_from_captions/ru|s3://yodas2/ru/0_from_captions/audio_{SHARD}.tar|256"
    "yodas_0bw_ru|yodas/0_by_whisper/ru|s3://yodas2/ru/0_by_whisper/audio_{SHARD}.tar|1024"
    "yodas_1bw_ru|yodas/1_by_whisper/ru|s3://yodas2/ru/1_by_whisper/audio_{SHARD}.tar|8192"
)

MAX_ARRAY=1000

for corpus_def in "${CORPORA[@]}"; do
    IFS='|' read -r NAME MANIFEST_SUB TAR_PATTERN NUM_SHARDS <<< "$corpus_def"

    [[ -n "$FILTER" && "$NAME" != "$FILTER" ]] && continue

    MANIFEST_DIR="${WORK_DIR}/output_manifests_by_video_v2/${MANIFEST_SUB}"
    OUT_DIR="${WORK_DIR}/output_utmos_by_video_v2/${MANIFEST_SUB}"
    LOG_DIR="${WORK_DIR}/logs/${NAME}"
    mkdir -p "${OUT_DIR}" "${LOG_DIR}"

    DONE=$(ls "${MANIFEST_DIR}"/shard_*.jsonl 2>/dev/null | wc -l)
    if [ "$DONE" -lt "$NUM_SHARDS" ]; then
        echo "=== ${NAME}: SKIP (${DONE}/${NUM_SHARDS} clustered shards) ==="
        echo
        continue
    fi

    # Build list of unscored shard IDs → file
    SHARD_LIST="${WORK_DIR}/logs/${NAME}_missing_shards.txt"
    > "${SHARD_LIST}"
    MISSING_COUNT=0
    for sid in $(seq 0 $((NUM_SHARDS - 1))); do
        [ -f "${OUT_DIR}/shard_${sid}.jsonl" ] && continue
        [ ! -f "${MANIFEST_DIR}/shard_${sid}.jsonl" ] && continue
        echo "$sid" >> "${SHARD_LIST}"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    done

    if [ $MISSING_COUNT -eq 0 ]; then
        echo "=== ${NAME}: ALL DONE (${NUM_SHARDS}/${NUM_SHARDS}) ==="
        echo
        continue
    fi

    SCORED=$((NUM_SHARDS - MISSING_COUNT))
    echo "=== ${NAME}: ${MISSING_COUNT} remaining (${SCORED}/${NUM_SHARDS} done) ==="

    # Submit in chunks of MAX_ARRAY
    CHUNK_IDX=0
    OFFSET=0
    while [ $OFFSET -lt $MISSING_COUNT ]; do
        CHUNK_SIZE=$((MISSING_COUNT - OFFSET))
        [ $CHUNK_SIZE -gt $MAX_ARRAY ] && CHUNK_SIZE=$MAX_ARRAY
        ARRAY_MAX=$((CHUNK_SIZE - 1))

        JOB_SCRIPT="${WORK_DIR}/logs/${NAME}_utmos_${CHUNK_IDX}.sbatch"
        cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=utmos_${NAME}_${CHUNK_IDX}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-${ARRAY_MAX}%128
#SBATCH --output=${LOG_DIR}/utmos_%a.out
#SBATCH --error=${LOG_DIR}/utmos_%a.err

# Map array index to actual shard ID via file
LINE=\$(( ${OFFSET} + \${SLURM_ARRAY_TASK_ID} + 1 ))
SHARD_ID=\$(sed -n "\${LINE}p" ${SHARD_LIST})

if [ -z "\${SHARD_ID}" ]; then
    echo "No shard ID at line \${LINE}"
    exit 0
fi

export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_cache
export MPLCONFIGDIR=/tmp/mpl_cache
export UTMOSV2_CHACHE=\${WORK_DIR}/utmosv2_cache
export AIS_ENDPOINT=\${AIS_ENDPOINT:-http://localhost:51080}

srun --export=ALL \\
     --container-image=${CONTAINER} \\
     --container-mounts="\${CONTAINER_MOUNTS:-/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw}" \\
     --container-writable \\
     /opt/conda/bin/python ${SCORE_PY} \\
        --manifest_dir "${MANIFEST_DIR}" \\
        --tar_pattern "${TAR_PATTERN}" \\
        --output_dir "${OUT_DIR}" \\
        --shard_ids "\${SHARD_ID}" \\
        --batch_size 64
EOF

        echo "  chunk ${CHUNK_IDX}: ${CHUNK_SIZE} shards (array 0-${ARRAY_MAX})"
        if $DRY_RUN; then
            echo "    [DRY RUN] sbatch ${JOB_SCRIPT}"
        else
            JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
            echo "    Submitted: job ${JOB_ID}"
        fi

        OFFSET=$((OFFSET + CHUNK_SIZE))
        CHUNK_IDX=$((CHUNK_IDX + 1))
    done
    echo
done
