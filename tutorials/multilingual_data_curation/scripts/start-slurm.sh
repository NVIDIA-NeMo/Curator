#!/bin/bash

# TODO: Edit these parameters to match your job requirements
#SBATCH --job-name={ACCOUNT_NAME}.{JOB_NAME}
#SBATCH --nodes={NUM_NODES}
#SBATCH --time={TIME_LIMIT}
#SBATCH --account={ACCOUNT_NAME}
#SBATCH --partition={PARTITION_NAME}
#SBATCH --output={JOB_NAME}-%j.out
#SBATCH --exclusive

# =================================================================
# Begin easy customization
# =================================================================

# Base directory for all Slurm job logs and files
# Does not affect directories referenced in your script
export BASE_DIR=`pwd`
export BASE_JOB_DIR=$BASE_DIR/nemo-curator-jobs
export RUN_ID="`date +"%Y_%m_%d"`-$SLURM_JOB_ID"
export JOB_DIR=$BASE_JOB_DIR/$RUN_ID

# TODO: Edit this path
DATA_DIR=/path/to/data

# Directory for Dask cluster communication and logging
# Must be paths inside the container that are accessible across nodes
export LOGDIR=$JOB_DIR/logs
export PROFILESDIR=$JOB_DIR/profiles
export SCHEDULER_FILE=$LOGDIR/scheduler.json
export SCHEDULER_LOG=$LOGDIR/scheduler.log
export DONE_MARKER=$LOGDIR/done.txt

# Main script to run
# In the script, Dask must connect to a cluster through the Dask scheduler
# We recommend passing the path to a Dask scheduler's file in a
# nemo_curator.utils.distributed_utils.get_client call like the examples
# TODO: Use GPU for deduplication modules, CPU for everything else
# export NVIDIA_VISIBLE_DEVICES=void  # For CPU only
export DEVICE="gpu"
# TODO: Edit this path to match your script
export SCRIPT_PATH=/path/to/script.py
# TODO: Edit this command as needed to match your script's arguments
export SCRIPT_COMMAND="python -u $SCRIPT_PATH \
    --scheduler-file $SCHEDULER_FILE \
    --device $DEVICE"

# Container parameters
# TODO: Edit this path to match your container image
export CONTAINER_IMAGE=/path/to/container.sqsh
# Make sure to mount the directories your script references
# TODO: Edit this path to match your container mounts
export MOUNTS="$HOME:$HOME,\
$BASE_DIR:$BASE_DIR,\
$DATA_DIR:$DATA_DIR"
# Below must be path to entrypoint script on your system
export CONTAINER_ENTRYPOINT=$BASE_DIR/scripts/container-entrypoint.sh

# TODO: Network interface specific to the cluster being used
export INTERFACE=eth0
export PROTOCOL=tcp

# CPU related variables
# 0 means no memory limit
export CPU_WORKER_MEMORY_LIMIT=0
# TODO: Edit this number to help avoid OOM errors on the CPU workers
export CPU_WORKERS=84

# GPU related variables
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"
export RMM_SCHEDULER_POOL_SIZE="1GB"
export RMM_WORKER_POOL_SIZE="72GiB"
export LIBCUDF_CUFILE_POLICY=OFF

# =================================================================
# End easy customization
# =================================================================

# Start the container
srun \
    --container-mounts=${MOUNTS} \
    --container-image=${CONTAINER_IMAGE} \
    ${CONTAINER_ENTRYPOINT}
