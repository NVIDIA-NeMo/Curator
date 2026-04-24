#!/bin/bash
# End-to-end HIFI pipeline submission for a single corpus.
#
# Submits Slurm array jobs for each stage, chaining via --dependency=afterok.
# Each stage uses the appropriate Docker container (sqsh on lustre).
#
# Usage:
#   bash submit_e2e.sh --corpus ytc_ru --dry-run
#   bash submit_e2e.sh --corpus yodas_0fc_ru
#   bash submit_e2e.sh --corpus yodas_1bw_ru --start-from embed
#
# Stages (in order):
#   sed -> sed_post -> segment -> diarize -> transcribe ->
#   embed -> group_video -> cluster -> utmos

set -euo pipefail

# ---- Parse args ----
CORPUS=""
DRY_RUN=false
START_FROM="sed"
CLUSTERING="per_video"
SCOTCH_PRESET="librispeech-2026-04"

while [[ $# -gt 0 ]]; do
    case $1 in
        --corpus) CORPUS="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        --start-from) START_FROM="$2"; shift 2;;
        --clustering) CLUSTERING="$2"; shift 2;;
        --scotch-preset) SCOTCH_PRESET="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done
[[ -z "$CORPUS" ]] && { echo "Usage: $0 --corpus <name> [--start-from <stage>] [--clustering per_video|scotch] [--scotch-preset <name>] [--dry-run]"; exit 1; }
case "$CLUSTERING" in
    per_video|scotch) ;;
    *) echo "--clustering must be 'per_video' or 'scotch', got: $CLUSTERING"; exit 1;;
esac

# ---- Paths ----
WORK="${WORK:?Set WORK to your working directory}"
GRANARY="${GRANARY:?Set GRANARY to your granary_filtered directory}"
CURATOR="${CURATOR:?Set CURATOR to your NeMo Curator checkout}"
SQSH_DIR="${SQSH_DIR:?Set SQSH_DIR to your container directory}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Slurm account}"

STAGE_PY="${WORK}/scripts/run_stage.py"

# ---- Container mapping ----
CONT_NEMO="${SQSH_DIR}/curator-hifi-nemo-stages.sqsh"
CONT_VLLM="${SQSH_DIR}/curator-hifi-pipeline.sqsh"
CONT_UTMOS="${SQSH_DIR}/curator-utmos.sqsh"
CONT_SCOTCH="${SQSH_DIR}/curator-scotch.sqsh"
SCOTCH_PY="${WORK}/scripts/run_cluster_scotch.py"

# ---- Corpus definitions ----
declare -A CORPUS_SUB CORPUS_SHARDS CORPUS_TAR
CORPUS_SUB[ytc_ru]="ytc/ru"
CORPUS_SHARDS[ytc_ru]=64
CORPUS_TAR[ytc_ru]="s3://YTC/ru/webds_ru/audio_{SHARD}.tar"

CORPUS_SUB[yodas_0fc_ru]="yodas/0_from_captions/ru"
CORPUS_SHARDS[yodas_0fc_ru]=256
CORPUS_TAR[yodas_0fc_ru]="s3://yodas2/ru/0_from_captions/audio_{SHARD}.tar"

CORPUS_SUB[yodas_0bw_ru]="yodas/0_by_whisper/ru"
CORPUS_SHARDS[yodas_0bw_ru]=1024
CORPUS_TAR[yodas_0bw_ru]="s3://yodas2/ru/0_by_whisper/audio_{SHARD}.tar"

CORPUS_SUB[yodas_1bw_ru]="yodas/1_by_whisper/ru"
CORPUS_SHARDS[yodas_1bw_ru]=8192
CORPUS_TAR[yodas_1bw_ru]="s3://yodas2/ru/1_by_whisper/audio_{SHARD}.tar"

SUB="${CORPUS_SUB[$CORPUS]}"
NUM_SHARDS="${CORPUS_SHARDS[$CORPUS]}"
TAR_PATTERN="${CORPUS_TAR[$CORPUS]}"
TAR_PREFIX="${TAR_PATTERN%%\{SHARD\}*}"
TAR_SUFFIX="${TAR_PATTERN##*\{SHARD\}}"

echo "=== E2E Pipeline: ${CORPUS} (${NUM_SHARDS} shards) ==="
echo "    Source: ${GRANARY}/${SUB}"
echo "    Output: ${WORK}/e2e_output/${SUB}"
echo "    Start from: ${START_FROM}"
echo "    Clustering: ${CLUSTERING}"
echo

# ---- Stage definitions ----
# STAGE_NAME|CONTAINER|PARTITION|GPUS|MEM|TIME|INPUT_DIR|OUTPUT_DIR|EXTRA_ARGS
#
# The clustering stages differ by --clustering:
#   per_video: group_video -> cluster (per-video AHC, hack around N^2 memory)
#   scotch:    cluster_scotch (corpus-wide BIRCH+AHC, single non-array job)
# In scotch mode utmos reads from clustered_scotch instead of clustered.
if [[ "$CLUSTERING" == "scotch" ]]; then
    CLUSTER_OUTPUT="${WORK}/e2e_output/${SUB}/clustered_scotch"
    CLUSTERING_STAGES=(
        "cluster_scotch|${CONT_SCOTCH}|cpu_short|0|128G|02:00:00|${WORK}/e2e_output/${SUB}/transcribe|${CLUSTER_OUTPUT}|"
    )
else
    CLUSTER_OUTPUT="${WORK}/e2e_output/${SUB}/clustered"
    CLUSTERING_STAGES=(
        "group_video|${CONT_NEMO}|cpu_short|0|32G|00:10:00|${WORK}/e2e_output/${SUB}/transcribe|${WORK}/e2e_output/${SUB}/grouped|"
        "cluster|${CONT_NEMO}|cpu_short|0|64G|00:30:00|${WORK}/e2e_output/${SUB}/grouped|${CLUSTER_OUTPUT}|--embedding_dir ${WORK}/e2e_output/${SUB}/embeddings"
    )
fi

STAGES=(
    "sed|${CONT_NEMO}|batch_singlenode|1|64G|00:30:00|${GRANARY}/${SUB}|${WORK}/e2e_output/${SUB}/sed|--sed_checkpoint /opt/checkpoints/Cnn14_DecisionLevelMax.pth"
    "sed_post|${CONT_NEMO}|cpu_short|0|32G|00:15:00|${WORK}/e2e_output/${SUB}/sed|${WORK}/e2e_output/${SUB}/sed_post|"
    "segment|${CONT_NEMO}|cpu_short|0|32G|00:15:00|${WORK}/e2e_output/${SUB}/sed_post|${WORK}/e2e_output/${SUB}/segment|"
    "diarize|${CONT_NEMO}|batch_singlenode|1|64G|00:30:00|${WORK}/e2e_output/${SUB}/segment|${WORK}/e2e_output/${SUB}/diarize|"
    "transcribe|${CONT_VLLM}|batch_singlenode|2|128G|00:30:00|${WORK}/e2e_output/${SUB}/diarize|${WORK}/e2e_output/${SUB}/transcribe|--language Ru --tensor_parallel_size 2"
    "embed|${CONT_NEMO}|batch_singlenode|1|64G|00:30:00|${WORK}/e2e_output/${SUB}/transcribe|${WORK}/e2e_output/${SUB}/embeddings|"
    "${CLUSTERING_STAGES[@]}"
    "utmos|${CONT_UTMOS}|batch_singlenode|1|64G|00:30:00|${CLUSTER_OUTPUT}|${WORK}/e2e_output/${SUB}/utmos|"
)

MAX_ARRAY=1000
MOUNTS="${CONTAINER_MOUNTS:-/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw}"
PREV_JOB=""
STARTED=false

for stage_def in "${STAGES[@]}"; do
    IFS='|' read -r STAGE CONTAINER PARTITION GPUS MEM TIME INPUT_DIR OUTPUT_DIR EXTRA <<< "$stage_def"

    # Skip stages before --start-from
    if ! $STARTED; then
        [[ "$STAGE" == "$START_FROM" ]] && STARTED=true || continue
    fi

    LOG_DIR="${WORK}/e2e_logs/${CORPUS}/${STAGE}"
    mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

    # --- Special case: corpus-wide SCOTCH clustering.
    # One non-array job over all shards; uses run_cluster_scotch.py, not run_stage.py.
    if [[ "$STAGE" == "cluster_scotch" ]]; then
        JOB_SCRIPT="${LOG_DIR}/cluster_scotch.sbatch"
        DEP_FLAG=""
        [[ -n "$PREV_JOB" ]] && DEP_FLAG="#SBATCH --dependency=afterok:${PREV_JOB}"

        cat > "${JOB_SCRIPT}" <<EOFSBATCH
#!/bin/bash
#SBATCH --job-name=cluster_scotch_${CORPUS}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/cluster_scotch.out
#SBATCH --error=${LOG_DIR}/cluster_scotch.err
${DEP_FLAG}

srun --export=ALL \\
     --container-image=${CONTAINER} \\
     --container-mounts="${MOUNTS}" \\
     --container-writable \\
     python /opt/scotch/run_cluster_scotch.py \\
        --manifest_dir "${INPUT_DIR}" \\
        --embedding_dir "${WORK}/e2e_output/${SUB}/embeddings" \\
        --output_dir "${OUTPUT_DIR}" \\
        --num_shards ${NUM_SHARDS} \\
        --preset ${SCOTCH_PRESET}
EOFSBATCH

        echo "  cluster_scotch: one non-array job over ${NUM_SHARDS} shards"
        if $DRY_RUN; then
            echo "    [DRY RUN] sbatch ${JOB_SCRIPT}"
        else
            JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
            echo "    Submitted: job ${JOB_ID}"
            PREV_JOB="${JOB_ID}"
        fi
        echo
        continue
    fi

    MAX_IDX=$((NUM_SHARDS - 1))
    GPU_FLAG=""
    [[ "$GPUS" -gt 0 ]] && GPU_FLAG="--gpus-per-task=${GPUS}"

    # Determine if stage needs audio tar
    NEEDS_TAR=""
    case "$STAGE" in
        sed|diarize|embed|utmos) NEEDS_TAR="yes";;
    esac

    CHUNK_START=0
    CHUNK_IDX=0
    while [ $CHUNK_START -le $MAX_IDX ]; do
        CHUNK_END=$((CHUNK_START + MAX_ARRAY))
        [ $CHUNK_END -gt $MAX_IDX ] && CHUNK_END=$MAX_IDX
        CHUNK_SIZE=$((CHUNK_END - CHUNK_START + 1))
        ARRAY_MAX=$((CHUNK_SIZE - 1))

        JOB_SCRIPT="${LOG_DIR}/${STAGE}_${CHUNK_IDX}.sbatch"

        # Pick python path: pipeline container uses venv, others use conda
        if [[ "$CONTAINER" == *curator-hifi-pipeline* ]]; then
            PY_PATH="/usr/bin/python3"
            EXTRA_PYTHONPATH=":/opt/curator_venv/lib/python3.10/site-packages"
        else
            PY_PATH="/opt/conda/bin/python"
            EXTRA_PYTHONPATH=""
        fi

        # Build tar URL line
        TAR_LINE=""
        if [[ -n "$NEEDS_TAR" ]]; then
            TAR_LINE="--tar_url \"${TAR_PREFIX}\${SHARD_ID}${TAR_SUFFIX}\""
        fi

        # Dependency
        DEP_FLAG=""
        [[ -n "$PREV_JOB" ]] && DEP_FLAG="#SBATCH --dependency=afterok:${PREV_JOB}"

        cat > "${JOB_SCRIPT}" <<EOFSBATCH
#!/bin/bash
#SBATCH --job-name=${STAGE}_${CORPUS}_${CHUNK_IDX}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1 --ntasks=1
${GPU_FLAG:+#SBATCH ${GPU_FLAG}}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --array=0-${ARRAY_MAX}%128
#SBATCH --output=${LOG_DIR}/${STAGE}_%a.out
#SBATCH --error=${LOG_DIR}/${STAGE}_%a.err
${DEP_FLAG}

export PYTHONUNBUFFERED=1
export PYTHONPATH=/opt/Curator${EXTRA_PYTHONPATH}
export HF_HOME=${WORK}/hf_cache
export UTMOSV2_CHACHE=${WORK}/utmosv2_cache
export AIS_ENDPOINT=${AIS_ENDPOINT:-http://localhost:51080}

SHARD_ID=\$((${CHUNK_START} + \${SLURM_ARRAY_TASK_ID}))
MANIFEST="${INPUT_DIR}/shard_\${SHARD_ID}.jsonl"

if [ ! -f "\${MANIFEST}" ]; then
    echo "Manifest not found: \${MANIFEST}"
    exit 0
fi

srun --export=ALL \\
     --container-image=${CONTAINER} \\
     --container-mounts="${MOUNTS}" \\
     --container-writable \\
     ${PY_PATH} ${STAGE_PY} \\
        --stage ${STAGE} \\
        --manifest_path "\${MANIFEST}" \\
        ${TAR_LINE} \\
        --output_dir "${OUTPUT_DIR}" \\
        --shard_id \${SHARD_ID} \\
        ${EXTRA}
EOFSBATCH

        echo "  ${STAGE} chunk ${CHUNK_IDX}: shards ${CHUNK_START}-${CHUNK_END}"
        if $DRY_RUN; then
            echo "    [DRY RUN] sbatch ${JOB_SCRIPT}"
        else
            JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
            echo "    Submitted: job ${JOB_ID}"
            PREV_JOB="${JOB_ID}"
        fi

        CHUNK_START=$((CHUNK_END + 1))
        CHUNK_IDX=$((CHUNK_IDX + 1))
    done
    echo
done

echo "=== Pipeline submitted for ${CORPUS}. Stages chained via --dependency. ==="
