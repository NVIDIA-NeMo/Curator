#!/bin/bash
# Submit speaker embedding extraction jobs for all corpora.
#
# Usage:
#   bash submit_embeddings.sh [--dry-run]
#
# Submits 4 Slurm array jobs (one per corpus). Each array task processes
# one shard: downloads tar from S3, runs TitaNet, saves embeddings_N.npz.
#
# Required environment variables:
#   MANIFESTS_DIR  - base path to granary-filtered manifests
#   WORK_DIR       - working directory for embeddings, logs, scripts
#   CONTAINER      - path to squashfs container image
#   ACCOUNT        - Slurm account name
#   AIS_ENDPOINT   - AIStore endpoint URL
#   AIS_AUTHN_TOKEN - AIStore auth JWT

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ---- Paths (override via env) ----
MANIFESTS_DIR="${MANIFESTS_DIR:?Set MANIFESTS_DIR to base path of granary-filtered manifests}"
WORK_DIR="${WORK_DIR:?Set WORK_DIR to working directory for embeddings/logs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_PY="${SCRIPT_DIR}/extract_shard_embeddings.py"
CONTAINER="${CONTAINER:?Set CONTAINER to path of squashfs container image}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to Slurm account name}"
PARTITION="${PARTITION:-batch_singlenode}"
AIS_ENDPOINT="${AIS_ENDPOINT:?Set AIS_ENDPOINT}"
AIS_AUTHN_TOKEN="${AIS_AUTHN_TOKEN:?Set AIS_AUTHN_TOKEN}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${WORK_DIR}/.cache/hf}"

mkdir -p "${WORK_DIR}/logs"

# ---- Corpus definitions ----
# Format: NAME|MANIFEST_SUBDIR|TAR_PATTERN|NUM_SHARDS
CORPORA=(
    "ytc_ru|ytc/ru|s3://YTC/ru/webds_ru/audio_{SHARD}.tar|64"
    "yodas_0fc_ru|yodas/0_from_captions/ru|s3://yodas2/ru/0_from_captions/audio_{SHARD}.tar|256"
    "yodas_0bw_ru|yodas/0_by_whisper/ru|s3://yodas2/ru/0_by_whisper/audio_{SHARD}.tar|1024"
    "yodas_1bw_ru|yodas/1_by_whisper/ru|s3://yodas2/ru/1_by_whisper/audio_{SHARD}.tar|8192"
)

for corpus_def in "${CORPORA[@]}"; do
    IFS='|' read -r NAME MANIFEST_SUB TAR_PATTERN NUM_SHARDS <<< "$corpus_def"

    MANIFEST_DIR="${MANIFESTS_DIR}/${MANIFEST_SUB}"
    EMB_DIR="${WORK_DIR}/embeddings/${MANIFEST_SUB}"
    LOG_DIR="${WORK_DIR}/logs/${NAME}"
    mkdir -p "${EMB_DIR}" "${LOG_DIR}"

    MAX_IDX=$((NUM_SHARDS - 1))
    MAX_ARRAY=1000  # Slurm MaxArraySize - 1

    # Split tar pattern into prefix/suffix around {SHARD}
    TAR_PREFIX="${TAR_PATTERN%%\{SHARD\}*}"
    TAR_SUFFIX="${TAR_PATTERN##*\{SHARD\}}"

    # Split into chunks of MAX_ARRAY if needed
    CHUNK_START=0
    CHUNK_IDX=0
    while [ $CHUNK_START -le $MAX_IDX ]; do
        CHUNK_END=$((CHUNK_START + MAX_ARRAY))
        [ $CHUNK_END -gt $MAX_IDX ] && CHUNK_END=$MAX_IDX

    CHUNK_SIZE=$((CHUNK_END - CHUNK_START + 1))
    ARRAY_MAX=$((CHUNK_SIZE - 1))

    JOB_SCRIPT="${WORK_DIR}/logs/${NAME}_embed_${CHUNK_IDX}.sbatch"
    cat > "${JOB_SCRIPT}" <<EOF_HEADER
#!/bin/bash
#SBATCH --job-name=spkemb_${NAME}_${CHUNK_IDX}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-${ARRAY_MAX}%64
#SBATCH --output=${LOG_DIR}/${CHUNK_START}_%a.out
#SBATCH --error=${LOG_DIR}/${CHUNK_START}_%a.err
EOF_HEADER
    cat >> "${JOB_SCRIPT}" <<EOF_BODY

# Compute actual shard ID from chunk offset
SHARD_ID=\$((${CHUNK_START} + \${SLURM_ARRAY_TASK_ID}))
EOF_BODY
    cat >> "${JOB_SCRIPT}" <<EOF_VARS
MANIFEST_PATH="${MANIFEST_DIR}/shard_\${SHARD_ID}.jsonl"
TAR_URL="${TAR_PREFIX}\${SHARD_ID}${TAR_SUFFIX}"
EMB_DIR="${EMB_DIR}"
OUT_FILE="\${EMB_DIR}/embeddings_\${SHARD_ID}.npz"
EXTRACT_PY="${EXTRACT_PY}"
CONTAINER="${CONTAINER}"
EOF_VARS
    cat >> "${JOB_SCRIPT}" <<EOF_LOGIC

# Skip if already done
if [ -f "\${OUT_FILE}" ]; then
    echo "Shard \${SHARD_ID}: embeddings already exist, skipping"
    exit 0
fi

# Skip if manifest doesn't exist
if [ ! -f "\${MANIFEST_PATH}" ]; then
    echo "Shard \${SHARD_ID}: manifest not found at \${MANIFEST_PATH}, skipping"
    exit 0
fi

export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export HF_HOME="\${TRANSFORMERS_CACHE}"
mkdir -p "\${TRANSFORMERS_CACHE}"

export AIS_ENDPOINT="${AIS_ENDPOINT}"
export AIS_AUTHN_TOKEN="${AIS_AUTHN_TOKEN}"

srun --container-image="\${CONTAINER}" \
     --container-mounts="/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw" \
     --container-writable \
     --export=ALL \
     python "\${EXTRACT_PY}" \
    --manifest_path "\${MANIFEST_PATH}" \
    --tar_url "\${TAR_URL}" \
    --output_dir "\${EMB_DIR}" \
    --shard_id \${SHARD_ID} \
    --batch_size 64 \
    --skip_filtered \
    --ais_token "\${AIS_AUTHN_TOKEN}"
EOF_LOGIC

    echo "=== ${NAME}: chunk ${CHUNK_IDX} (shards ${CHUNK_START}-${CHUNK_END}) ==="
    echo "    Script:   ${JOB_SCRIPT}"

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] Would submit: sbatch ${JOB_SCRIPT}"
    else
        JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
        echo "    Submitted: job ${JOB_ID}"
    fi

        CHUNK_START=$((CHUNK_END + 1))
        CHUNK_IDX=$((CHUNK_IDX + 1))
    done  # end while chunk loop

    echo "  ${NAME}: ${NUM_SHARDS} shards total, output -> ${EMB_DIR}"
    echo
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
