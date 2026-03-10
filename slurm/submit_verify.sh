#!/bin/bash
# =========================================================================
# Verify SlurmRayClient + XennaExecutor.
#
# Override nodes/GPUs via sbatch flags:
#   sbatch --nodes=1 --gpus-per-node=2 slurm/submit_verify.sh
#   sbatch --nodes=1 --gpus-per-node=8 slurm/submit_verify.sh
#   sbatch --nodes=2 --gpus-per-node=2 slurm/submit_verify.sh
#   sbatch --nodes=2 --gpus-per-node=8 slurm/submit_verify.sh
# =========================================================================

#SBATCH --job-name=verify-slurm-ray
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --output=slurm/outputs/verify_%j.log
#SBATCH --error=slurm/outputs/verify_%j.log

set -euo pipefail

CURATOR_DIR="${CURATOR_DIR:-/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/abhgarg/code/Curator}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/abhgarg/cache/uv}"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

cd "${CURATOR_DIR}"
source .venv/bin/activate

echo "=========================================="
echo " SlurmRayClient Verification"
echo "=========================================="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Nodes     : ${SLURM_JOB_NODELIST} (${SLURM_JOB_NUM_NODES})"
echo "GPUs/node : ${SLURM_GPUS_ON_NODE:-none}"
echo "CPUs/node : ${SLURM_CPUS_ON_NODE:-N/A}"
echo "Python    : $(which python)"
echo "Ray       : $(which ray)"
echo "=========================================="

python slurm/verify_slurm.py

echo "=========================================="
echo " DONE"
echo "=========================================="
