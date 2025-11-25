#!/bin/bash
#
# Sample SLURM + Singularity/Apptainer Ray cluster launcher
#
# USAGE:
#   RUN_COMMAND="python your_script.py" sbatch --nodes=2 --gres=gpu:4 ray-cluster.sbatch
#
# NOTES:
#   - Customise the SBATCH directives below for your cluster (account, partition, etc.).
#   - This script:
#       * Starts a Ray head node on the first SLURM node
#       * Starts Ray workers on the remaining nodes
#       * Runs a user-provided command on the head node inside the same container
#   - Most behaviour can be overridden with environment variables (see "User knobs").

########################################################
# SLURM Directives
########################################################
#SBATCH --job-name=ray-cluster-example
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=ray-job-%j.out
#SBATCH --error=ray-job-%j.err

# Common cluster-specific options you may want to add:
# #SBATCH --account=your_account
# #SBATCH --partition=your_partition
# #SBATCH --mem=0
# #SBATCH --qos=your_qos
# #SBATCH --constraint=gpu
# #SBATCH --exclude=some_node[01-10]

########################################################
# Safety & debug options
########################################################
set -euo pipefail

: "${DEBUG:=0}"
if [[ "${DEBUG}" -eq 1 ]]; then
  set -x
fi

########################################################
# User knobs (override via env vars)
########################################################

# Command to run *after* the Ray cluster is up.
: "${RUN_COMMAND:=python -c 'import ray; ray.init(); print(ray.cluster_resources())'}"

# Ray ports
: "${GCS_PORT:=6379}"       # Ray GCS port
: "${CLIENT_PORT:=10001}"   # Ray Client (ray://) port
: "${DASH_PORT:=8265}"      # Ray Dashboard port

# Container runtime & image
: "${CONTAINER_CMD:=singularity}"           # could be "apptainer" on certain sites
: "${IMAGE:=/path/to/your-container-image.sif}"   # override to your .sif image

# Base bind-mounts (comma-separated, Singularity/Apptainer style)
: "${CONTAINER_MOUNTS:=$(pwd)}"

# Scratch / temporary directories (default under $PWD, override to $SCRATCH, /tmp, etc.)
: "${SCRATCH_ROOT:=${SCRATCH:-$(pwd)}}"

JOB_ID="${SLURM_JOB_ID:?This script must be run via sbatch under SLURM}"

RAY_TMP="${RAY_TMP:-${SCRATCH_ROOT}/ray_tmp/${JOB_ID}}"
WORKERS_TMP="${WORKERS_TMP:-${SCRATCH_ROOT}/ray_workers_tmp/${JOB_ID}}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${SCRATCH_ROOT}/ray_spill/${JOB_ID}}"

mkdir -p "${RAY_TMP}" "${WORKERS_TMP}" "${RAY_SPILL_DIR}"

# Hugging Face cache / offline settings
: "${HF_HOME:=${SCRATCH_ROOT}/.cache/huggingface}"
: "${HF_HUB_OFFLINE:=1}"   # default to offline mode; set to 0 to allow network access
mkdir -p "${HF_HOME}"

# Startup waits (seconds)
: "${HEAD_STARTUP_WAIT:=10}"
: "${WORKER_STARTUP_WAIT:=60}"

########################################################
# Resource detection
########################################################
NUM_CPUS_PER_NODE="${NUM_CPUS_PER_NODE:-${SLURM_CPUS_ON_NODE:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc --all)}}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"

########################################################
# Node list / topology
########################################################
# You can override NODES from the environment if you want a custom subset.
if [[ -z "${NODES:-}" ]]; then
  mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
else
  # Split space-separated list from env into array
  read -r -a NODES <<< "${NODES}"
fi

if [[ "${#NODES[@]}" -lt 1 ]]; then
  echo "ERROR: No nodes found for this job." >&2
  exit 1
fi

HEAD_NODE_NAME="${NODES[0]}"
NUM_WORKERS=$(( ${#NODES[@]} - 1 ))

# IP address of the head node (first IP if multiple)
HEAD_NODE_IP=$(srun \
  --nodes=1 \
  --ntasks=1 \
  --overlap \
  -w "${HEAD_NODE_NAME}" \
  bash -c "hostname -I | awk '{print \$1}'"
)

RAY_GCS_ADDRESS="${HEAD_NODE_IP}:${GCS_PORT}"
RAY_CLIENT_ADDRESS="${HEAD_NODE_IP}:${CLIENT_PORT}"  # For ray:// clients if needed

########################################################
# Container bind mounts
########################################################
# Mount HF cache
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${HF_HOME}"

# Head node: also mount temporary dirs and spill dir
HEAD_CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${RAY_TMP}:/tmp,${RAY_SPILL_DIR}"
WORKER_CONTAINER_MOUNTS="${CONTAINER_MOUNTS}"

########################################################
# Environment variables for container
########################################################
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export HF_HOME
export HF_HUB_OFFLINE
export OMP_NUM_THREADS="${NUM_CPUS_PER_NODE}"

# Propagate key vars into the container via Singularity/Apptainer
for v in PYTHONNOUSERSITE PYTORCH_CUDA_ALLOC_CONF PYTHONPATH HF_HOME HF_HUB_OFFLINE OMP_NUM_THREADS; do
  eval "export SINGULARITYENV_${v}=\"\${${v}}\""
done

# Example Ray tuning env vars (uncomment to adjust object store usage)
# export SINGULARITYENV_RAY_memory_usage_threshold=0.98
# export SINGULARITYENV_RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.6

########################################################
# Cleanup handling
########################################################
cleanup() {
  echo "Cleaning up Ray temporary directories..."
  rm -rf "${RAY_TMP}" "${WORKERS_TMP}" "${RAY_SPILL_DIR}" || true
}
trap cleanup EXIT INT TERM

########################################################
# Start Ray head node
########################################################
echo "Starting Ray head node on ${HEAD_NODE_NAME} (IP: ${HEAD_NODE_IP})"

srun \
  --nodes=1 \
  --ntasks=1 \
  --overlap \
  --export=ALL \
  -w "${HEAD_NODE_NAME}" \
    ${CONTAINER_CMD} exec \
      --bind="${HEAD_CONTAINER_MOUNTS}" \
      --containall \
      --nv \
      ${IMAGE} \
      bash -c "
        ray start \
          --head \
          --num-cpus ${NUM_CPUS_PER_NODE} \
          --num-gpus ${NUM_GPUS_PER_NODE} \
          --temp-dir /tmp \
          --node-ip-address ${HEAD_NODE_IP} \
          --port ${GCS_PORT} \
          --disable-usage-stats \
          --dashboard-host 0.0.0.0 \
          --dashboard-port ${DASH_PORT} \
          --ray-client-server-port ${CLIENT_PORT} \
          --object-spilling-directory ${RAY_SPILL_DIR} \
          --block
      " &

sleep "${HEAD_STARTUP_WAIT}"

########################################################
# Start Ray workers
########################################################
if [[ "${NUM_WORKERS}" -gt 0 ]]; then
  echo "Starting ${NUM_WORKERS} Ray worker node(s)..."
fi

for ((i = 1; i < ${#NODES[@]}; i++)); do
  NODE_I="${NODES[$i]}"
  WORKER_TMP_I="${WORKERS_TMP}/${i}"
  mkdir -p "${WORKER_TMP_I}"

  echo "Initializing worker ${i} on ${NODE_I}"

  srun \
    -N1 -n1 \
    --exclusive \
    --cpu-bind=none \
    --mpi=none \
    --export=ALL \
    --gres=gpu:"${NUM_GPUS_PER_NODE}" \
    -w "${NODE_I}" \
    ${CONTAINER_CMD} exec \
      --bind="${WORKER_CONTAINER_MOUNTS},${WORKER_TMP_I}:/tmp" \
      --containall \
      --nv \
      ${IMAGE} \
      bash -c "
        ray start \
          --address ${RAY_GCS_ADDRESS} \
          --num-cpus ${NUM_CPUS_PER_NODE} \
          --num-gpus ${NUM_GPUS_PER_NODE} \
          --block
    " &
done

if [[ "${NUM_WORKERS}" -gt 0 ]]; then
  sleep "${WORKER_STARTUP_WAIT}"
fi

########################################################
# Run the user command on the head node
########################################################
echo "RUNNING COMMAND on head node ${HEAD_NODE_NAME}:"
echo "  ${RUN_COMMAND}"
echo

srun \
  -N1 -n1 -c1 \
  --overlap \
  --cpu-bind=none \
  --mpi=none \
  --export=ALL \
  -w "${HEAD_NODE_NAME}" \
  "${CONTAINER_CMD}" exec \
    --bind="${HEAD_CONTAINER_MOUNTS}" \
    --containall \
    --nv \
    "${IMAGE}" \
    bash -c "${RUN_COMMAND}"

echo "COMMAND FINISHED. Ray cluster will be cleaned up when the job ends."