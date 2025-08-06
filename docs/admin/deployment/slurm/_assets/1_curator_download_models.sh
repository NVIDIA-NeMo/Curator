#!/bin/bash

#SBATCH --job-name=download_models
#SBATCH -p defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

# Update Me!
#SBATCH --output=/home/<username>/logs/%x_%j.log
USER_DIR="/home/${USER}"
CONTAINER_IMAGE="${USER_DIR}/path-to/curator.sqsh"
#

LOCAL_WORKSPACE="${USER_DIR}/nemo_curator_local_workspace"
LOCAL_WORKSPACE_MOUNT="${LOCAL_WORKSPACE}:/config"
NEMO_CONFIG_MOUNT="${NEMO_CONFIG}:/nemo_curator/config/nemo_curator.yaml"
CONTAINER_MOUNTS="${LOCAL_WORKSPACE_MOUNT},${NEMO_CONFIG_MOUNT}"

export NEMO_CURATOR_RAY_SLURM_JOB=1
export NEMO_CURATOR_LOCAL_DOCKER_JOB=1

# Download Models
srun \
  --mpi=none \
  --container-writable \
  --no-container-remap-root \
  --export=NEMO_CURATOR_RAY_SLURM_JOB,NEMO_CURATOR_LOCAL_DOCKER_JOB \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    --  python3 -m nemo_curator.video.models.model_cli download \
        --config-file /config/model_download.yaml