#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# EXAMPLE: RUN_SCRIPT="python curation_script.py" sbatch --time=00:30:00 --nodes=2 ray-sbatch-job.sh

########################################################
# SLURM Directives
########################################################
#SBATCH --exclusive
#SBATCH --output=curator-%j.out

# NOTE: Some commonly used options are commented out below, uncomment them if you need to set them or add your own
# #SBATCH --job-name=curator-pipeline
# #SBATCH --nodes=1
# #SBATCH --time=02:00:00
# #SBATCH --account=my_account
# #SBATCH --partition=my_partition
# #SBATCH --dependency=singleton


set -ux


########################################################
# User knobs (override via env vars) or in the script
########################################################
: "${RUN_COMMAND:=python -c 'import ray; ray.init(); print(ray.cluster_resources())'}" # Change as needed

# Ports
: "${GCS_PORT:=6379}"          # Ray GCS (native) port
: "${CLIENT_PORT:=10001}"      # Ray Client port (ray://)
: "${DASH_PORT:=8265}"         # Dashboard port


########################################################
# Container specific variables
########################################################
: "${IMAGE:=nvcr.io/nvidia/nemo-curator:25.09}"
: "${CONTAINER_MOUNTS:=}" # Set as needed


########################################################
# Ray setup variables
########################################################
NUM_CPUS_PER_NODE="${NUM_CPUS_PER_NODE:-$(srun --jobid ${JOB_ID} --nodes=1 bash -c "echo \${SLURM_CPUS_ON_NODE}")}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"


JOB_ID=${SLURM_JOB_ID}
# Getting the node names
NODES=${NODES:-$(scontrol show hostnames $(sacct -j ${JOB_ID} -X --json | jq -r .jobs[0].nodes))}
NODES=(${NODES})

HEAD_NODE_NAME=${NODES[0]}
HEAD_NODE_IP=$(srun --jobid ${JOB_ID} --nodes=1 --ntasks=1 -w "$HEAD_NODE_NAME" bash -c "hostname  --ip-address")

RAY_GCS_ADDRESS=$HEAD_NODE_IP:$GCS_PORT
RAY_CLIENT_ADDRESS=$HEAD_NODE_IP:$CLIENT_PORT
export RAY_GCS_ADDRESS
export RAY_CLIENT_ADDRESS
export RAY_ADDRESS="ray://$RAY_CLIENT_ADDRESS"

echo "RAY_ADDRESS: $RAY_ADDRESS"



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
    bash -c "ray start \
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
                --block" &
sleep 10

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
        bash -c "ray start \
                     --address ${RAY_GCS_ADDRESS} \
                     --num-cpus ${NUM_CPUS_PER_NODE} \
                     --num-gpus ${NUM_GPUS_PER_NODE} \
                     --block;" &
    sleep 1
done
sleep 10

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
    bash -c "RAY_ADDRESS=$RAY_ADDRESS $RUN_COMMAND"
