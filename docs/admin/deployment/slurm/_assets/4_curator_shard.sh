#!/bin/bash

#SBATCH --job-name=shard
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
CONTAINER_IMAGE="${USER_DIR}/path/to/curator.sqsh"
S3_CLIPS_DIR=s3://BUCKET/CLIPS_PATH
S3_SEMDEDUPE_DIR=s3://BUCKET/CLIPS_PATH/semantic
S3_SHARDS_DIR=s3://BUCKET/SHARDS_PATH
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

# Shard
srun \
  --mpi=none \
  --container-writable \
  --no-container-remap-root \
  --no-container-mount-home \
  --export=NEMO_CURATOR_RAY_SLURM_JOB,NEMO_CURATOR_LOCAL_DOCKER_JOB,NEMO_S3_PROFILE_PATH \
  --gres=gpu:8 \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    -- python3 -m nemo_curator.video.pipelines.video.run_pipeline shard\
      --input-clip-path ${S3_CLIPS_DIR}\
      --output-dataset-path ${S3_SHARDS_DIR}\
      --annotation-version v0\
      --input-semantic-dedup-path ${S3_SEMDEDUPE_DIR}\
      --semantic-dedup-epsilon 0.01\
      --verbose