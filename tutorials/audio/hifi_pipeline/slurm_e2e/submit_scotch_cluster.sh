#!/bin/bash
# Corpus-wide SCOTCH (BIRCH+AHC) clustering as an alternative to the
# per-video `cluster` stage in submit_e2e.sh.
#
# Reuses already-processed transcribe + embed outputs. Single non-array
# job — gather -> cluster -> scatter — writes per-shard clustered jsonls
# plus a cluster_config.json sidecar.
#
# Usage:
#   bash submit_scotch_cluster.sh --corpus ytc_ru
#   bash submit_scotch_cluster.sh --corpus ytc_ru --dry-run
#   bash submit_scotch_cluster.sh --corpus yodas_0fc_ru --preset librispeech-2026-04
#
# Required env vars:
#   WORK       working dir on lustre
#   CURATOR    NeMo Curator checkout (for PYTHONPATH)
#   SQSH_DIR   directory containing *.sqsh container images
#   ACCOUNT    slurm account

set -euo pipefail

CORPUS=""
DRY_RUN=false
PRESET="librispeech-2026-04"

while [[ $# -gt 0 ]]; do
    case $1 in
        --corpus) CORPUS="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        --preset) PRESET="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done
[[ -z "$CORPUS" ]] && { echo "Usage: $0 --corpus <name> [--preset <name>] [--dry-run]"; exit 1; }

WORK="${WORK:?Set WORK to your working directory}"
CURATOR="${CURATOR:?Set CURATOR to your NeMo Curator checkout}"
SQSH_DIR="${SQSH_DIR:?Set SQSH_DIR to your container directory}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Slurm account}"

CONT_SCOTCH="${SQSH_DIR}/curator-scotch.sqsh"
MOUNTS="${CONTAINER_MOUNTS:-/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw}"

declare -A CORPUS_SUB CORPUS_SHARDS
CORPUS_SUB[ytc_ru]="ytc/ru";           CORPUS_SHARDS[ytc_ru]=64
CORPUS_SUB[yodas_0fc_ru]="yodas/0_from_captions/ru"; CORPUS_SHARDS[yodas_0fc_ru]=256
CORPUS_SUB[yodas_0bw_ru]="yodas/0_by_whisper/ru";    CORPUS_SHARDS[yodas_0bw_ru]=1024
CORPUS_SUB[yodas_1bw_ru]="yodas/1_by_whisper/ru";    CORPUS_SHARDS[yodas_1bw_ru]=8192

[[ -z "${CORPUS_SUB[$CORPUS]+x}" ]] && { echo "Unknown corpus: $CORPUS"; exit 1; }

SUB="${CORPUS_SUB[$CORPUS]}"
NUM_SHARDS="${CORPUS_SHARDS[$CORPUS]}"

MANIFEST_DIR="${WORK}/e2e_output/${SUB}/transcribe"
EMBEDDING_DIR="${WORK}/e2e_output/${SUB}/embeddings"
OUTPUT_DIR="${WORK}/e2e_output/${SUB}/clustered_scotch"
LOG_DIR="${WORK}/e2e_logs/${CORPUS}/cluster_scotch"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# Memory scales with N and with the leaf pdist matrix. The preset caps
# leaves well under 200k so 128G is plenty for ytc_ru / yodas scales.
MEM="${SCOTCH_MEM:-128G}"
TIME="${SCOTCH_TIME:-02:00:00}"
PARTITION="${SCOTCH_PARTITION:-cpu_short}"

JOB_SCRIPT="${LOG_DIR}/cluster_scotch.sbatch"
cat > "${JOB_SCRIPT}" <<EOFSBATCH
#!/bin/bash
#SBATCH --job-name=scotch_${CORPUS}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/cluster_scotch.out
#SBATCH --error=${LOG_DIR}/cluster_scotch.err

srun --export=ALL \\
     --container-image=${CONT_SCOTCH} \\
     --container-mounts="${MOUNTS}" \\
     --container-writable \\
     python /opt/scotch/run_cluster_scotch.py \\
        --manifest_dir "${MANIFEST_DIR}" \\
        --embedding_dir "${EMBEDDING_DIR}" \\
        --output_dir "${OUTPUT_DIR}" \\
        --num_shards ${NUM_SHARDS} \\
        --preset ${PRESET}
EOFSBATCH

echo "=== SCOTCH cluster: ${CORPUS} (${NUM_SHARDS} shards, preset=${PRESET}) ==="
echo "  manifest : ${MANIFEST_DIR}"
echo "  embedding: ${EMBEDDING_DIR}"
echo "  output   : ${OUTPUT_DIR}"
echo "  sbatch   : ${JOB_SCRIPT}"
if $DRY_RUN; then
    echo "[DRY RUN] sbatch ${JOB_SCRIPT}"
else
    JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
    echo "Submitted: job ${JOB_ID}"
fi
