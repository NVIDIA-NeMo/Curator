#!/bin/bash
# Submit the full HIFI data curation pipeline as a Slurm job.
#
# Usage:
#   bash submit_pipeline.sh [--dry-run]
#
# The pipeline runs all stages sequentially on a single GPU node:
#   SED -> SED postprocess -> segment extract -> diarize ->
#   transcribe (3-pass cascade) -> embed -> cluster -> utmos
#
# Required environment variables:
#   CONTAINER       - path to squashfs container image
#   ACCOUNT         - Slurm account name
#   CURATOR_DIR     - path to Curator repo checkout
#
# Optional environment variables (have defaults):
#   PARTITION       - Slurm partition (default: batch_singlenode)
#   NUM_GPUS        - GPUs per job (default: 1)
#   CPUS_PER_TASK   - CPUs per job (default: 32)
#   MEM             - Memory per job (default: 128G)
#   TIME            - Wall time (default: 24:00:00)
#
# Pipeline parameters are configured in the CORPORA array below.
# Edit as needed for your datasets.

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ---- Required vars ----
CONTAINER="${CONTAINER:?Set CONTAINER to path of squashfs container image}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to Slurm account name}"
CURATOR_DIR="${CURATOR_DIR:?Set CURATOR_DIR to Curator repo path}"

# ---- Optional vars with defaults ----
PARTITION="${PARTITION:-batch_singlenode}"
NUM_GPUS="${NUM_GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-128G}"
TIME="${TIME:-24:00:00}"
WORK_DIR="${WORK_DIR:-${CURATOR_DIR}/output/hifi_pipeline}"
LOG_DIR="${WORK_DIR}/logs"

# ---- Pipeline defaults ----
STAGES="${STAGES:-sed,sed_post,segment,diarize,transcribe,embed,cluster,utmos}"
LANGUAGE="${LANGUAGE:-Ru}"
SED_CHECKPOINT="${SED_CHECKPOINT:-}"
SED_MODEL_TYPE="${SED_MODEL_TYPE:-Cnn14_DecisionLevelMax}"
SED_THRESHOLD="${SED_THRESHOLD:-0.5}"
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORT="${VLLM_PORT:-8200}"
OMNI_MODEL="${OMNI_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct}"
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-0.292}"
CLUSTER_BATCH_SIZE="${CLUSTER_BATCH_SIZE:-64}"
UTMOS_BATCH_SIZE="${UTMOS_BATCH_SIZE:-16}"
SPEAKER_MODEL="${SPEAKER_MODEL:-nvidia/speakerverification_en_titanet_large}"

mkdir -p "${LOG_DIR}"

# ---- Corpus definitions ----
# Format: NAME|INPUT_MANIFEST|OUTPUT_SUBDIR
# Edit this array for your datasets.
CORPORA=(
    "ytc_ru|/path/to/ytc/ru/manifest.jsonl|ytc_ru"
    "yodas_0fc_ru|/path/to/yodas/0_from_captions/ru/manifest.jsonl|yodas_0fc_ru"
    "yodas_0bw_ru|/path/to/yodas/0_by_whisper/ru/manifest.jsonl|yodas_0bw_ru"
    "yodas_1bw_ru|/path/to/yodas/1_by_whisper/ru/manifest.jsonl|yodas_1bw_ru"
)

PIPELINE_PY="${CURATOR_DIR}/tutorials/audio/hifi_pipeline/run_pipeline.py"

for corpus_def in "${CORPORA[@]}"; do
    IFS='|' read -r NAME INPUT_MANIFEST OUTPUT_SUBDIR <<< "$corpus_def"

    OUTPUT_DIR="${WORK_DIR}/${OUTPUT_SUBDIR}"
    mkdir -p "${OUTPUT_DIR}"

    JOB_SCRIPT="${LOG_DIR}/${NAME}_pipeline.sbatch"
    cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=hifi_${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/${NAME}_pipeline.out
#SBATCH --error=${LOG_DIR}/${NAME}_pipeline.err

srun --container-image=${CONTAINER} \\
     --container-mounts=/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw \\
     --container-writable \\
     bash -c "export PYTHONPATH=${CURATOR_DIR}:\\\$PYTHONPATH && \\
     python ${PIPELINE_PY} \\
        --input_manifest '${INPUT_MANIFEST}' \\
        --output_dir '${OUTPUT_DIR}' \\
        --stages '${STAGES}' \\
        --language '${LANGUAGE}' \\
        --sed_checkpoint '${SED_CHECKPOINT}' \\
        --sed_model_type '${SED_MODEL_TYPE}' \\
        --sed_threshold ${SED_THRESHOLD} \\
        --vllm_host '${VLLM_HOST}' \\
        --vllm_port ${VLLM_PORT} \\
        --omni_model '${OMNI_MODEL}' \\
        --llm_model '${LLM_MODEL}' \\
        --speaker_model '${SPEAKER_MODEL}' \\
        --cluster_threshold ${CLUSTER_THRESHOLD} \\
        --cluster_batch_size ${CLUSTER_BATCH_SIZE} \\
        --utmos_batch_size ${UTMOS_BATCH_SIZE}"
EOF

    echo "=== ${NAME} ==="
    echo "    Input:  ${INPUT_MANIFEST}"
    echo "    Output: ${OUTPUT_DIR}"
    echo "    Stages: ${STAGES}"
    echo "    Script: ${JOB_SCRIPT}"

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] Would submit: sbatch ${JOB_SCRIPT}"
    else
        JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
        echo "    Submitted: job ${JOB_ID}"
    fi
    echo
done

echo "Monitor with: squeue -u \$USER -n 'hifi_*'"
