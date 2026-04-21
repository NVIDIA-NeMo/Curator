#!/bin/bash
# =============================================================================
# NeMo Curator — SLURM submit script (NGC container via Pyxis/enroot)
#
# Runs the slurm demo pipeline inside the official NeMo Curator container
# using the Pyxis SLURM plugin, with the local Curator virtualenv activated
# so that the latest (unreleased) code is used.
#
# This mirrors the pattern used for the Nemotron-Parse PDF pipeline.
#
# Prerequisites:
#   - Pyxis plugin installed on the cluster (check: srun --help | grep container)
#   - NeMo Curator source checked out on a shared filesystem (Lustre / NFS)
#   - A virtualenv built from that source: python -m venv .venv && pip install -e .
#   - The shared filesystem mounted at the same path inside the container
#
# Usage:
#   sbatch tutorials/slurm/submit_container.sh
#
# Override resources without editing this file:
#   sbatch --nodes=1 --gpus-per-node=2 tutorials/slurm/submit_container.sh
#   sbatch --nodes=1 --gpus-per-node=8 tutorials/slurm/submit_container.sh
#   sbatch --nodes=2 --gpus-per-node=2 tutorials/slurm/submit_container.sh
#   sbatch --nodes=2 --gpus-per-node=8 tutorials/slurm/submit_container.sh
# =============================================================================

#SBATCH --job-name=curator-slurm-demo-container
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_demo_container_%j.log
#SBATCH --error=logs/slurm_demo_container_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
CURATOR_DIR="${CURATOR_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Official NeMo Curator container from NGC.
# Browse available tags: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo-curator:26.02}"

# Mount the shared filesystem that contains your code and data.
# Format: <host_path>:<container_path>[,<host_path2>:<container_path2>]
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre}"

# Shared directory for Ray port broadcast — must be visible to ALL nodes.
# On most clusters /tmp is node-local, so we use a Lustre path here.
# Adjust to any shared filesystem path accessible from every compute node.
export RAY_PORT_BROADCAST_DIR="${CURATOR_DIR}/logs"

echo "=================================================="
echo "  NeMo Curator — SLURM Demo (container)"
echo "=================================================="
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Nodes     : ${SLURM_JOB_NODELIST} (${SLURM_JOB_NUM_NODES} nodes)"
echo "  GPUs/node : ${SLURM_GPUS_ON_NODE:-none}"
echo "  Container : ${CONTAINER_IMAGE}"
echo "  Mounts    : ${CONTAINER_MOUNTS}"
echo "  Dir       : ${CURATOR_DIR}"
echo "=================================================="

mkdir -p logs

srun \
    --ntasks-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir="${CURATOR_DIR}" \
    bash -c "
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
export RAY_PORT_BROADCAST_DIR='${CURATOR_DIR}/logs'

# Activate the local virtualenv so the latest Curator code (from this
# checkout) is used instead of the version bundled in the container image.
source '${CURATOR_DIR}/.venv/bin/activate'

echo \"[\$(hostname)] SLURM_NODEID=\${SLURM_NODEID} python=\$(python --version 2>&1)\"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    | sed \"s/^/  [\$(hostname)] GPU /\" || echo \"  [\$(hostname)] no GPUs\"

python '${CURATOR_DIR}/tutorials/slurm/pipeline.py' \
    --slurm \
    --num-tasks 80
"

echo "=================================================="
echo "  DONE"
echo "=================================================="
