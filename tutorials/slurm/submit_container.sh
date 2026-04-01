#!/bin/bash
# =============================================================================
# NeMo Curator — SLURM submit script (NGC container via Pyxis/enroot)
#
# Same pipeline as submit.sh but runs inside the official NeMo Curator
# container image using the Pyxis SLURM plugin (--container-image flag).
#
# Pyxis/enroot is available on most NVIDIA DGX SuperPOD and OCI clusters.
# If your cluster uses a different container runtime (Singularity, Apptainer),
# see the README for the equivalent flags.
#
# Prerequisites:
#   - Pyxis plugin installed on the cluster (check: srun --help | grep container)
#   - Shared filesystem mounted at the same path inside the container
#
# Usage:
#   sbatch tutorials/slurm/submit_container.sh
# =============================================================================

#SBATCH --job-name=curator-slurm-demo-container
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_demo_container_%j.log
#SBATCH --error=logs/slurm_demo_container_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
CURATOR_DIR="${CURATOR_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Official NeMo Curator container from NGC.
# Browse available tags: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo-curator:latest}"

# Mount the shared filesystem that contains your code/data.
# Format: <host_path>:<container_path>[,<host_path2>:<container_path2>]
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre}"

export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

echo "=================================================="
echo "  NeMo Curator — SLURM Demo (container)"
echo "=================================================="
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Nodes     : ${SLURM_JOB_NODELIST} (${SLURM_JOB_NUM_NODES} nodes)"
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
echo \"[\$(hostname)] SLURM_NODEID=\${SLURM_NODEID} python=\$(which python)\"
python '${CURATOR_DIR}/tutorials/slurm/pipeline.py' \
    --slurm \
    --num-tasks 40
"

echo "=================================================="
echo "  DONE"
echo "=================================================="
