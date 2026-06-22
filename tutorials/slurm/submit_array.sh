#!/bin/bash
# =============================================================================
# NeMo Curator — Slurm array submit script
#
# Splits a large set of JSONL or Parquet files across multiple Slurm array
# tasks so that each job independently processes its assigned slice of the input.
#
# Example: 2000 input files, --array=0-19 -> 20 jobs x ~100 files each.
#
# How it works:
#   - FilePartitioningStage groups all files into source tasks.
#   - Each array task reads SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT and
#     selects only the source tasks assigned to it via deterministic SHA-256 hashing.
#   - Jobs run in parallel with no coordination between them.
#
# Prerequisites:
#   - NeMo Curator source checked out on a shared filesystem (Lustre, NFS, etc.)
#   - A virtualenv built at ${CURATOR_DIR}/.venv with NeMo Curator installed
#   - INPUT_DIR set to a directory of JSONL or Parquet files visible from all compute nodes
#   - OUTPUT_DIR set to a writable shared directory
#   - CHECKPOINT_PATH set to a writable shared path for retry manifests
#
# Usage:
#   # 20 jobs (task IDs 0-19), ~100 files per job with a 2000-file dataset
#   sbatch --array=0-19 tutorials/slurm/submit_array.sh
#
#   # Override array size or resources at submission time:
#   sbatch --array=0-9  --cpus-per-task=32 tutorials/slurm/submit_array.sh
#   sbatch --array=0-39 --time=02:00:00    tutorials/slurm/submit_array.sh
#
# Array indexing:
#   shard_index  = SLURM_ARRAY_TASK_ID   (set automatically by Slurm)
#   total_shards = SLURM_ARRAY_TASK_COUNT (set automatically by Slurm)
#   minimum_shard_index defaults to 0 — no env var fallback.
#   shard_index_offset defaults to 0 and is added to SLURM_ARRAY_TASK_ID
#     only when SHARD_INDEX is not explicitly set.
#
# If your array does not start at 0 (e.g. --array=1-20), set:
#   MINIMUM_SHARD_INDEX=1 sbatch --array=1-20 tutorials/slurm/submit_array.sh
# =============================================================================

#SBATCH --job-name=curator-array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=0
#SBATCH --time=01:00:00
#SBATCH --output=array_%A_%a.log
#SBATCH --error=array_%A_%a.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
CURATOR_DIR="${CURATOR_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Input and output directories
INPUT_DIR="${INPUT_DIR:-/path/to/your/input/directory}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/your/output/directory}"

# Retry manifests are written under:
#   ${CHECKPOINT_PATH}/.nemo_curator_metadata/.slurm_array_retry/
# FailedTask marker files are written under:
#   ${CHECKPOINT_PATH}/.nemo_curator_metadata/.failed_tasks/<slurm-job>/<array-task>/<shard>/
# Defaults to OUTPUT_DIR. If you override CONTAINER_MOUNTS, make sure it still
# includes CHECKPOINT_PATH.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${OUTPUT_DIR}}"

# Official NeMo Curator container from NGC.
# Browse available tags: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo-curator:26.02}"

# Mount the shared filesystem paths that contain your code and data.
# Format: <host_path>:<container_path>[,<host_path2>:<container_path2>]
# Override this if your cluster expects filesystem roots, e.g. /lustre:/lustre,/home:/home.
DEFAULT_CONTAINER_MOUNTS="${CURATOR_DIR}:${CURATOR_DIR},${INPUT_DIR}:${INPUT_DIR},${OUTPUT_DIR}:${OUTPUT_DIR}"
if [[ "${CHECKPOINT_PATH}" != "${OUTPUT_DIR}" ]]; then
    DEFAULT_CONTAINER_MOUNTS="${DEFAULT_CONTAINER_MOUNTS},${CHECKPOINT_PATH}:${CHECKPOINT_PATH}"
fi
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${DEFAULT_CONTAINER_MOUNTS}}"

# Input and output file types
INPUT_FILE_TYPE="${INPUT_FILE_TYPE:-jsonl}"
OUTPUT_FILE_TYPE="${OUTPUT_FILE_TYPE:-jsonl}"

# Number of files to read into a single DocumentBatch
FILES_PER_PARTITION="${FILES_PER_PARTITION:-1}"

# Shard index and total shards.
#
# SHARD_INDEX_OFFSET is useful on clusters that limit the maximum Slurm array
# index. For example, submit --array=0-999 with SHARD_INDEX_OFFSET=1000 to
# process logical shards 1000-1999.
SHARD_INDEX_OFFSET="${SHARD_INDEX_OFFSET:-0}"
SHARD_INDEX="${SHARD_INDEX:-$((SLURM_ARRAY_TASK_ID + SHARD_INDEX_OFFSET))}"
TOTAL_SHARDS="${TOTAL_SHARDS:-${SLURM_ARRAY_TASK_COUNT}}"

# Offset between 0-indexed hash assignments and SLURM_ARRAY_TASK_ID.
# Leave at 0 for --array=0-N. Set to the array start value for --array=K-N.
MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX:-0}"

# Backend adapter source-task filtering is controlled by environment rather
# than reader-stage constructor arguments.
NEMO_CURATOR_SLURM_ARRAY_ENABLED=1
NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="${SHARD_INDEX}"
NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="${TOTAL_SHARDS}"
NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX}"

# BaseStageAdapter writes one marker JSON per FailedTask when this env var is
# set. Keep one directory per Slurm array job/task so retries can inspect just
# the FailedTasks from that job, while multi-node workers for that same job
# write into the same directory.
FAILED_TASKS_DIR="${FAILED_TASKS_DIR:-${CHECKPOINT_PATH}/.nemo_curator_metadata/.failed_tasks/slurm_job_${SLURM_JOB_ID:-local}/array_task_${SLURM_ARRAY_TASK_ID:-local}/shard_${SHARD_INDEX}}"
NEMO_CURATOR_FAILED_TASKS_DIR="${NEMO_CURATOR_FAILED_TASKS_DIR:-${FAILED_TASKS_DIR}}"

# Use SlurmRayClient only when this array task spans multiple nodes. Single-node
# array tasks can use the regular RayClient.
NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
USE_SLURM_RAY=0
if (( NUM_NODES > 1 )); then
    USE_SLURM_RAY=1
fi

mkdir -p "${CURATOR_DIR}/logs" "${OUTPUT_DIR}" "${CHECKPOINT_PATH}" "${NEMO_CURATOR_FAILED_TASKS_DIR}"

export CURATOR_DIR
export INPUT_DIR
export OUTPUT_DIR
export CHECKPOINT_PATH
export FAILED_TASKS_DIR
export NEMO_CURATOR_FAILED_TASKS_DIR
export INPUT_FILE_TYPE
export OUTPUT_FILE_TYPE
export FILES_PER_PARTITION
export SHARD_INDEX_OFFSET
export SHARD_INDEX
export TOTAL_SHARDS
export MINIMUM_SHARD_INDEX
export NEMO_CURATOR_SLURM_ARRAY_ENABLED
export NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX
export NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS
export NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX
export USE_SLURM_RAY

echo "=================================================="
echo "  NeMo Curator — Slurm Array Demo"
echo "=================================================="
echo "  Job array ID   : ${SLURM_ARRAY_JOB_ID}"
echo "  Array task ID  : ${SLURM_ARRAY_TASK_ID}"
echo "  Array task cnt : ${SLURM_ARRAY_TASK_COUNT}"
echo "  Shard index    : ${SHARD_INDEX}"
echo "  Shard offset   : ${SHARD_INDEX_OFFSET}"
echo "  Total shards   : ${TOTAL_SHARDS}"
echo "  Nodes          : ${NUM_NODES}"
echo "  Ray client     : $([[ "${USE_SLURM_RAY}" == "1" ]] && echo SlurmRayClient || echo RayClient)"
echo "  Node           : $(hostname)"
echo "  Container : ${CONTAINER_IMAGE}"
echo "  Mounts    : ${CONTAINER_MOUNTS}"
echo "  Dir       : ${CURATOR_DIR}"
echo "  Checkpoint path: ${CHECKPOINT_PATH}"
echo "  FailedTask dir : ${NEMO_CURATOR_FAILED_TASKS_DIR}"
echo "=================================================="

# Each array task processes only the deterministic source tasks hashed to its
# SLURM_ARRAY_TASK_ID. With --nodes=1, the task uses a local RayClient. With
# --nodes>1, the same Python entrypoint runs on every node and --slurm enables
# SlurmRayClient so workers join the head-node Ray cluster.
srun \
    --ntasks-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir="${CURATOR_DIR}" \
    bash -c '
set -euo pipefail

export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
export RAY_PORT_BROADCAST_DIR="${CURATOR_DIR}/logs"

# Activate the local virtualenv so the latest Curator code (from this
# checkout) is used instead of the version bundled in the container image.
source "${CURATOR_DIR}/.venv/bin/activate"

echo "[$(hostname)] SLURM_NODEID=${SLURM_NODEID:-0} python=$(python --version 2>&1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    | sed "s/^/  [$(hostname)] GPU /" || echo "  [$(hostname)] no GPUs"

pipeline_args=(
    --input-dir "${INPUT_DIR}"
    --input-file-type "${INPUT_FILE_TYPE}"
    --output-dir "${OUTPUT_DIR}"
    --output-file-type "${OUTPUT_FILE_TYPE}"
    --files-per-partition "${FILES_PER_PARTITION}"
    --shard-index "${SHARD_INDEX}"
    --total-shards "${TOTAL_SHARDS}"
    --minimum-shard-index "${MINIMUM_SHARD_INDEX}"
    --checkpoint-path "${CHECKPOINT_PATH}"
)

if [[ "${USE_SLURM_RAY}" == "1" ]]; then
    pipeline_args+=(--slurm)
fi

python "${CURATOR_DIR}/tutorials/slurm/array_pipeline.py" "${pipeline_args[@]}"
'

echo "=================================================="
echo "  Array task ${SLURM_ARRAY_TASK_ID} DONE"
echo "=================================================="
