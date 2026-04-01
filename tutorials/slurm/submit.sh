#!/bin/bash
# =============================================================================
# NeMo Curator — SLURM submit script (bare-metal / venv)
#
# Runs the slurm demo pipeline across multiple nodes using SlurmRayClient.
# Each node runs one srun task; node 0 acts as the Ray head, nodes 1-N are
# Ray workers. Only the head runs the pipeline — workers exit when the head
# calls ray_client.stop().
#
# Prerequisites:
#   - NeMo Curator installed in a shared virtualenv (see README.md)
#   - Shared filesystem accessible from all nodes (e.g. Lustre, NFS)
#
# Usage:
#   sbatch tutorials/slurm/submit.sh
#
# Override resources without editing this file:
#   sbatch --nodes=4 --cpus-per-task=32 tutorials/slurm/submit.sh
# =============================================================================

#SBATCH --job-name=curator-slurm-demo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_demo_%j.log
#SBATCH --error=logs/slurm_demo_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
CURATOR_DIR="${CURATOR_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VENV="${CURATOR_DIR}/.venv"

# Use /tmp per-job to avoid cross-job Ray state collisions
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

echo "=================================================="
echo "  NeMo Curator — SLURM Demo"
echo "=================================================="
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Nodes    : ${SLURM_JOB_NODELIST} (${SLURM_JOB_NUM_NODES} nodes)"
echo "  Dir      : ${CURATOR_DIR}"
echo "=================================================="

mkdir -p logs

srun \
    --ntasks-per-node=1 \
    bash -c "
source '${VENV}/bin/activate'
echo \"[\$(hostname)] SLURM_NODEID=\${SLURM_NODEID} python=\$(which python)\"
python '${CURATOR_DIR}/tutorials/slurm/pipeline.py' \
    --slurm \
    --num-tasks 40
"

echo "=================================================="
echo "  DONE"
echo "=================================================="
