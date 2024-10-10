#! /bin/bash

#SBATCH --job-name=nemo-curator:pretraining-curation
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=04:00:00

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =================================================================
# Begin easy customization
# =================================================================

# Base directory for all SLURM job logs and files
# Does not affect directories referenced in your script
export BASE_JOB_DIR=`pwd`/nemo-curator-jobs
export JOB_DIR=$BASE_JOB_DIR/$SLURM_JOB_ID

# Directory for Dask cluster communication and logging
# Must be paths inside the container that are accessible across nodes
export LOGDIR=$JOB_DIR/logs
export PROFILESDIR=$JOB_DIR/profiles
export SCHEDULER_FILE=$LOGDIR/scheduler.json
export SCHEDULER_LOG=$LOGDIR/scheduler.log
export DONE_MARKER=$LOGDIR/done.txt

# Device type
# This will change depending on the module to run
export DEVICE="gpu"

# Container parameters
export CONTAINER_IMAGE=/path/to/container
# Make sure to mount the directories your script references
export BASE_DIR=`pwd`
export MOUNTS="${BASE_DIR}:${BASE_DIR}"
# Below must be path to entrypoint script on your system
export CONTAINER_ENTRYPOINT=`pwd`/container-entrypoint.sh

# Network interface specific to the cluster being used
export INTERFACE=eth0
export PROTOCOL=tcp

# CPU related variables
export CPU_WORKER_MEMORY_LIMIT=0  # 0 means no memory limit
export CPU_WORKER_PER_NODE=128  # number of cpu workers per node

# GPU related variables
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"
export RMM_SCHEDULER_POOL_SIZE="1GB"
export RMM_WORKER_POOL_SIZE="72GiB"
export LIBCUDF_CUFILE_POLICY=OFF
export DASK_DATAFRAME__QUERY_PLANNING=False


# =================================================================
# End easy customization
# =================================================================

# Start the container
srun \
    --container-mounts=${MOUNTS} \
    --container-image=${CONTAINER_IMAGE} \
    ${CONTAINER_ENTRYPOINT}
