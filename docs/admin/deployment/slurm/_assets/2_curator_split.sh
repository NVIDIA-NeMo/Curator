#!/bin/bash

#SBATCH --job-name=split
#SBATCH -p defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --comment=fact_off
#SBATCH --gres=gpu:8
#SBATCH --dependency=singleton

## Update Me!
#SBATCH --output=/home/<username>/logs/%x_%j.log
USER_DIR="/home/${USER}"
CONTAINER_IMAGE="${USER_DIR}/path-to/curator.sqsh"
S3_DATA_DIR=s3://BUCKET/INPUT_PREFIX
S3_CLIPS_DIR=s3://BUCKET/INPUT_PREFIX
##

AWS_CREDS="/home/${USER}/.aws/credentials"
LOCAL_WORKSPACE="${USER_DIR}/nemo_curator_local_workspace"
NEMO_CONFIG="/home/${USER}/.config/nemo_curator/config.yaml"
AWS_CREDS_MOUNT="${AWS_CREDS}:/creds/s3_creds"
LOCAL_WORKSPACE_MOUNT="${LOCAL_WORKSPACE}:/config"
NEMO_CONFIG_MOUNT="${NEMO_CONFIG}:/nemo_curator/config/nemo_curator.yaml"
CONTAINER_MOUNTS="${LOCAL_WORKSPACE_MOUNT},${NEMO_CONFIG_MOUNT},${AWS_CREDS_MOUNT}"

export NEMO_S3_PROFILE_PATH="/creds/s3_creds"
export NEMO_CURATOR_RAY_SLURM_JOB=1
export NEMO_CURATOR_LOCAL_DOCKER_JOB=1

# Split
srun \
  --mpi=none \
  --no-container-remap-root \
  --no-container-mount-home \
  --gres=gpu:8 \
  --export=NEMO_CURATOR_RAY_SLURM_JOB,NEMO_CURATOR_LOCAL_DOCKER_JOB,NEMO_S3_PROFILE_PATH \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    --  python3 -m nemo_curator.video.pipelines.video.run_pipeline split \
        --input-video-path ${S3_DATA_DIR} \
        --output-clip-path ${S3_CLIPS_DIR} \
        --captioning-prompt-text "Elaborate on the visual and narrative elements of the video" \
        --splitting-algorithm "fixed-stride" \
        --fixed-stride-split-duration 30 \
        --limit 10 \
        --verbose