#!/bin/bash
# Slurm array submit script for tutorials/slurm/array_pipeline.py.
# Refer to the README for more details.
#
# Usage:
#   sbatch --array=0-19 tutorials/slurm/submit_array.sh
#   MINIMUM_SHARD_INDEX=1 sbatch --array=1-20 tutorials/slurm/submit_array.sh

#SBATCH --job-name=curator-array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=0
#SBATCH --time=01:00:00
#SBATCH --output=array_%A_%a.log
#SBATCH --error=array_%A_%a.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if (($#)); then
    echo "ERROR: unexpected argument(s): $*" >&2
    echo "Pass sbatch options before tutorials/slurm/submit_array.sh." >&2
    exit 2
fi

CURATOR_DIR="${CURATOR_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

INPUT_DIR="${INPUT_DIR:-/path/to/your/input/directory}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/your/output/directory}"

# Retry attempts must reuse this path. Completion manifests live under
# ${CHECKPOINT_PATH}/.nemo_curator_metadata/.slurm_array_completion/.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${OUTPUT_DIR}}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo-curator:26.02}"
# CONTAINER_IMAGE and CONTAINER_MOUNTS are interpolated directly into the srun
# flags below; they are not forwarded as environment variables into the container.

DEFAULT_CONTAINER_MOUNTS="${CURATOR_DIR}:${CURATOR_DIR},${INPUT_DIR}:${INPUT_DIR},${OUTPUT_DIR}:${OUTPUT_DIR}"
if [[ "${CHECKPOINT_PATH}" != "${OUTPUT_DIR}" ]]; then
    DEFAULT_CONTAINER_MOUNTS="${DEFAULT_CONTAINER_MOUNTS},${CHECKPOINT_PATH}:${CHECKPOINT_PATH}"
fi
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${DEFAULT_CONTAINER_MOUNTS}}"

INPUT_FILE_TYPE="${INPUT_FILE_TYPE:-jsonl}"
OUTPUT_FILE_TYPE="${OUTPUT_FILE_TYPE:-jsonl}"
FILES_PER_PARTITION="${FILES_PER_PARTITION:-1}"

if [[ -z "${SHARD_INDEX:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit_array.sh requires sbatch --array or SHARD_INDEX to be set." >&2
    exit 2
fi
if [[ -z "${TOTAL_SHARDS:-}" && -z "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
    echo "ERROR: submit_array.sh requires sbatch --array or TOTAL_SHARDS to be set." >&2
    exit 2
fi

SHARD_INDEX_OFFSET="${SHARD_INDEX_OFFSET:-0}"
SHARD_INDEX="${SHARD_INDEX:-$((SLURM_ARRAY_TASK_ID + SHARD_INDEX_OFFSET))}"
TOTAL_SHARDS="${TOTAL_SHARDS:-${SLURM_ARRAY_TASK_COUNT}}"
MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX:-0}"

export NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="${SHARD_INDEX}"
export NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="${TOTAL_SHARDS}"
export NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX}"

NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
USE_SLURM_RAY=0
if (( NUM_NODES > 1 )); then
    USE_SLURM_RAY=1
fi

mkdir -p "${CURATOR_DIR}/logs" "${OUTPUT_DIR}" "${CHECKPOINT_PATH}"

export CURATOR_DIR
export INPUT_DIR
export OUTPUT_DIR
export CHECKPOINT_PATH
export INPUT_FILE_TYPE
export OUTPUT_FILE_TYPE
export FILES_PER_PARTITION
export SHARD_INDEX_OFFSET
export SHARD_INDEX
export TOTAL_SHARDS
export MINIMUM_SHARD_INDEX
export USE_SLURM_RAY

echo "=================================================="
echo "  NeMo Curator — Slurm Array Demo"
echo "=================================================="
echo "  Job array ID   : ${SLURM_ARRAY_JOB_ID:-manual}"
echo "  Array task ID  : ${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX}}"
echo "  Array task cnt : ${SLURM_ARRAY_TASK_COUNT:-${TOTAL_SHARDS}}"
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
echo "=================================================="

srun \
    --ntasks-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir="${CURATOR_DIR}" \
    bash -c '
set -euo pipefail

export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-local}"
export RAY_PORT_BROADCAST_DIR="${CURATOR_DIR}/logs"

# Prefer this checkout over the container package.
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
    --checkpoint-path "${CHECKPOINT_PATH}"
)

if [[ "${USE_SLURM_RAY}" == "1" ]]; then
    pipeline_args+=(--slurm)
fi

python "${CURATOR_DIR}/tutorials/slurm/array_pipeline.py" "${pipeline_args[@]}"
'

echo "=================================================="
echo "  Array task ${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX}} DONE"
echo "=================================================="
