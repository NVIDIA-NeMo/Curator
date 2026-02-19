#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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



# USAGE: sbatch --{optional-flags} ray-sbatch-job.sh
# EXAMPLE: RUN_COMMAND="python curation_script.py" sbatch --time=00:30:00 --nodes=2 ray-sbatch-job.sh

########################################################
# SLURM Directives
########################################################
#SBATCH --exclusive
#SBATCH --output=curator-%j.out
#SBATCH --error=curator-%j.err

# ---------------------------------------------------
# These are some commonly used slurm directives that can be uncommented and modified as needed
# ---------------------------------------------------
# #SBATCH --job-name=curator-pipeline
# #SBATCH --nodes=1
# #SBATCH --time=02:00:00
# #SBATCH --account=my_account
# #SBATCH --partition=my_partition
# #SBATCH --gpus-per-node=8
# #SBATCH --dependency=singleton


set -ux


########################################################
########################################################
#                  User Configuration                  #
########################################################
########################################################

# ---------------------------------------------------
# These variables are usually set by the user
# ---------------------------------------------------
: "${RUN_COMMAND:=python -c 'import ray; ray.init(); print(ray.cluster_resources())'}"
: "${IMAGE:=nvcr.io/nvidia/nemo-curator:latest}"
: "${CONTAINER_MOUNTS:=}"

# ---------------------------------------------------
# These variables have sensible defaults but can be customized
# ---------------------------------------------------

# Ray cluster port configuration
# Ray workers prefer ports in the range [10002, 19999] so we try to avoid them.
: "${GCS_PORT:=6379}"
: "${CLIENT_PORT:=10001}"
: "${DASH_PORT:=8265}"
: "${NODE_MANAGER_PORT:=6800}"
: "${OBJECT_MANAGER_PORT:=6801}"
: "${RUNTIME_ENV_AGENT_PORT:=6802}"
: "${DASHBOARD_AGENT_GRPC_PORT:=6803}"
: "${METRICS_EXPORT_PORT:=6804}"

# Ray resource configuration
: "${NUM_CPUS_PER_NODE:=}"          # If empty, auto-detected via SLURM_CPUS_ON_NODE
: "${NUM_GPUS_PER_NODE:=8}"
: "${OBJECT_STORE_MEMORY:=}"        # If empty, uses Ray's default (auto-calculated)

# Logging configuration
: "${LOGURU_LEVEL:=INFO}"

# Optional command to run on each node before starting Ray
# Useful for patching, cleanup, or environment setup
# Example: NODE_INIT_COMMAND="bash pre_ray_startup.sh"
: "${NODE_INIT_COMMAND:=}"


########################################################
########################################################
##                                                    ##
##      SCRIPT LOGIC - No modifications needed        ##
##                                                    ##
########################################################
########################################################

JOB_ID=${SLURM_JOB_ID}

# If NUM_CPUS_PER_NODE is not set, auto-detect from SLURM
if [[ -z "${NUM_CPUS_PER_NODE}" ]]; then
    NUM_CPUS_PER_NODE=$(srun --jobid ${JOB_ID} --nodes=1 bash -c "echo \${SLURM_CPUS_ON_NODE}")
fi

# Build optional object store memory argument (only if OBJECT_STORE_MEMORY is set)
if [[ -n "${OBJECT_STORE_MEMORY:-}" ]]; then
    OBJECT_STORE_MEMORY_ARG="--object-store-memory ${OBJECT_STORE_MEMORY}"
else
    OBJECT_STORE_MEMORY_ARG=""
fi

# Getting the node names
NODES=${NODES:-$(scontrol show hostnames $(sacct -j ${JOB_ID} -X --json | jq -r .jobs[0].nodes))}
NODES=(${NODES})

HEAD_NODE_NAME=${NODES[0]}
HEAD_NODE_IP=$(srun --jobid ${JOB_ID} --nodes=1 --ntasks=1 -w "$HEAD_NODE_NAME" bash -c "hostname  --ip-address")


# Ray address exports
RAY_GCS_ADDRESS=$HEAD_NODE_IP:$GCS_PORT
RAY_CLIENT_ADDRESS=$HEAD_NODE_IP:$CLIENT_PORT
export RAY_GCS_ADDRESS
export RAY_CLIENT_ADDRESS
export RAY_DASHBOARD_ADDRESS="http://$HEAD_NODE_IP:$DASH_PORT"


# Xenna and logging environment variables
export RAY_MAX_LIMIT_FROM_API_SERVER=50000
export RAY_MAX_LIMIT_FROM_DATA_SOURCE=50000
export XENNA_RESPECT_CUDA_VISIBLE_DEVICES="1"

export LOGURU_LEVEL

# Hide GPUs from containers when running CPU-only
if [[ "${NUM_GPUS_PER_NODE}" == "0" ]]; then
    export NVIDIA_VISIBLE_DEVICES=void
fi

echo "RAY_DASHBOARD_ADDRESS: $RAY_DASHBOARD_ADDRESS"



# number of nodes other than the head node
NUM_WORKERS=$((${#NODES[@]} - 1))

########################################################
# Start ray on the head node
########################################################
srun \
  --nodes=1 \
  -w ${HEAD_NODE_NAME} \
  --container-image=$IMAGE \
  --container-mounts=$CONTAINER_MOUNTS \
  --overlap \
    bash -c "${NODE_INIT_COMMAND:+$NODE_INIT_COMMAND && }ray start \
                --head \
               --num-cpus ${NUM_CPUS_PER_NODE} \
               --num-gpus ${NUM_GPUS_PER_NODE} \
               --temp-dir /tmp/ray_${JOB_ID} \
               --node-ip-address ${HEAD_NODE_IP} \
               --port ${GCS_PORT} \
               --disable-usage-stats \
               --dashboard-host 0.0.0.0 \
               --dashboard-port ${DASH_PORT} \
               --ray-client-server-port ${CLIENT_PORT} \
               ${OBJECT_STORE_MEMORY_ARG} \
               --node-manager-port ${NODE_MANAGER_PORT} \
               --object-manager-port ${OBJECT_MANAGER_PORT} \
               --runtime-env-agent-port ${RUNTIME_ENV_AGENT_PORT} \
               --dashboard-agent-grpc-port ${DASHBOARD_AGENT_GRPC_PORT} \
               --metrics-export-port ${METRICS_EXPORT_PORT} \
               --block" &
sleep 40
echo "Started ray on the head node"

########################################################
# Start ray on the worker nodes
########################################################
for ((i = 1; i <= NUM_WORKERS; i++)); do
    NODE_I=${NODES[$i]}
    echo "Initializing WORKER $i at $NODE_I"
    srun \
      --nodes=1 \
      -w ${NODE_I} \
      --container-image=$IMAGE \
      --container-mounts=$CONTAINER_MOUNTS \
      --overlap \
        bash -c "${NODE_INIT_COMMAND:+$NODE_INIT_COMMAND && }ray start \
                    --address ${RAY_GCS_ADDRESS} \
                    --num-cpus ${NUM_CPUS_PER_NODE} \
                    --num-gpus ${NUM_GPUS_PER_NODE} \
                    ${OBJECT_STORE_MEMORY_ARG} \
                    --node-manager-port ${NODE_MANAGER_PORT} \
                    --object-manager-port ${OBJECT_MANAGER_PORT} \
                    --runtime-env-agent-port ${RUNTIME_ENV_AGENT_PORT} \
                    --dashboard-agent-grpc-port ${DASHBOARD_AGENT_GRPC_PORT} \
                    --metrics-export-port ${METRICS_EXPORT_PORT} \
                    --block;" &
    sleep 1
done
sleep 60
echo "Started ray on the worker nodes"

########################################################
# Run the command
########################################################
echo "RUNNING COMMAND $RUN_COMMAND"

srun \
  --nodes=1 \
  --overlap \
  -w ${HEAD_NODE_NAME} \
  --container-image=$IMAGE \
  --container-mounts=$CONTAINER_MOUNTS \
    bash -c "ray job submit --address $RAY_DASHBOARD_ADDRESS --submission-id=$JOB_ID -- $RUN_COMMAND"
