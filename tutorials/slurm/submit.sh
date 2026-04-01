#!/bin/bash
# =============================================================================
# NeMo Curator — SLURM submit script (bare-metal, shared virtualenv)
#
# Runs the slurm demo pipeline across multiple nodes using SlurmRayClient.
# Uses a virtualenv installed on a shared filesystem (Lustre/NFS) so every
# node sees the same Python environment without a container runtime.
#
# Prerequisites:
#   - NeMo Curator source checked out on a shared filesystem
#   - A virtualenv built from that source:
#       python -m venv .venv && pip install -e .
#   - Shared filesystem accessible from all nodes (e.g. Lustre, NFS)
#
# If your cluster has Pyxis/enroot, prefer submit_container.sh instead —
# it avoids managing a shared Python installation.
#
# Usage:
#   sbatch tutorials/slurm/submit.sh
#
# Override resources without editing this file:
#   sbatch --nodes=1 --gpus-per-node=2 tutorials/slurm/submit.sh
#   sbatch --nodes=1 --gpus-per-node=8 tutorials/slurm/submit.sh
#   sbatch --nodes=2 --gpus-per-node=2 tutorials/slurm/submit.sh
#   sbatch --nodes=2 --gpus-per-node=8 tutorials/slurm/submit.sh
# =============================================================================

#SBATCH --job-name=curator-slurm-demo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_demo_%j.log
#SBATCH --error=logs/slurm_demo_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
CURATOR_DIR="${CURATOR_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Use per-job /tmp to avoid cross-job Ray state collisions.
# On clusters where /tmp is node-local, set RAY_PORT_BROADCAST_DIR to a
# shared filesystem path so all nodes can discover the head's GCS port:
#   export RAY_PORT_BROADCAST_DIR="/shared/ray_ports"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

echo "=================================================="
echo "  NeMo Curator — SLURM Demo"
echo "=================================================="
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Nodes     : ${SLURM_JOB_NODELIST} (${SLURM_JOB_NUM_NODES} nodes)"
echo "  GPUs/node : ${SLURM_GPUS_ON_NODE:-none}"
echo "  CPUs/node : ${SLURM_CPUS_ON_NODE:-N/A}"
echo "  Dir       : ${CURATOR_DIR}"
echo "=================================================="

mkdir -p logs

srun \
    --ntasks-per-node=1 \
    bash -c "
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}

# Activate the shared virtualenv — must be on Lustre/NFS visible to all nodes.
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
