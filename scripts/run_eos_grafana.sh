#!/bin/bash
# Qwen Omni pipeline with Prometheus/Grafana monitoring.
# Supports single-node and multi-node (set --nodes).
# After job starts, check the log for the SSH tunnel command to access dashboards.
#SBATCH --job-name=qwen-grafana
#SBATCH --account=llmservice_nemo_speechlm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00

set -ux

# ---- USER CONFIG: update these paths for your environment ----
CONTAINER="/lustre/fsw/llmservice_nemo_speechlm/users/nkoluguri/containers/curator-nightly-lhotse.sqsh"
CURATOR_DIR="/lustre/fsw/llmservice_nemo_speechlm/users/mmkrtchyan/projects/Curator"
OUTPUT_DIR="/lustre/fsw/llmservice_nemo_speechlm/users/mmkrtchyan/outputs"
OUTPUT="${OUTPUT_DIR}/qwen_omni_yodas_grafana.jsonl"
DATA_CONFIG="/lustre/fsw/llmservice_nemo_speechlm/users/nkoluguri/projects/transformer_revamp/scripts/fc/am-fl-gc-ll-mc-mm-yt-yo_En-d0.5_v1.1.yaml"
HF_CACHE="/lustre/fsw/llmservice_nemo_speechlm/users/mmkrtchyan/.cache/huggingface"
LOGIN_HOST="login-eos01.eos.clusters.nvidia.com"
# ---- END USER CONFIG ----

mkdir -p "${OUTPUT_DIR}"

: "${GCS_PORT:=6379}"
: "${CLIENT_PORT:=10001}"
: "${DASH_PORT:=8265}"
: "${NODE_MANAGER_PORT:=6800}"
: "${OBJECT_MANAGER_PORT:=6801}"
: "${RUNTIME_ENV_AGENT_PORT:=6802}"
: "${DASHBOARD_AGENT_GRPC_PORT:=6803}"
: "${METRICS_EXPORT_PORT:=6804}"
: "${NUM_GPUS_PER_NODE:=8}"

JOB_ID=${SLURM_JOB_ID}

RAY_TEMP_DIR="/tmp/ray_${JOB_ID}"
METRICS_DIR="/tmp/nemo_curator_metrics_${JOB_ID}"
export RAY_TEMP_DIR METRICS_DIR

NUM_CPUS_PER_NODE=$(srun --jobid ${JOB_ID} --nodes=1 bash -c "echo \${SLURM_CPUS_ON_NODE}")

NODES=($(scontrol show hostnames ${SLURM_NODELIST}))
HEAD_NODE_NAME=${NODES[0]}
HEAD_NODE_IP=$(srun --jobid ${JOB_ID} --nodes=1 --ntasks=1 -w "$HEAD_NODE_NAME" bash -c "hostname --ip-address")

mkdir -p "${METRICS_DIR}" "${RAY_TEMP_DIR}"
CONTAINER_MOUNTS="/lustre:/lustre,${METRICS_DIR}:${METRICS_DIR},${RAY_TEMP_DIR}:${RAY_TEMP_DIR}"

RAY_GCS_ADDRESS=$HEAD_NODE_IP:$GCS_PORT
export RAY_GCS_ADDRESS
export RAY_ADDRESS="$HEAD_NODE_IP:$GCS_PORT"
export RAY_DASHBOARD_ADDRESS="http://$HEAD_NODE_IP:$DASH_PORT"
export RAY_MAX_LIMIT_FROM_API_SERVER=50000
export RAY_MAX_LIMIT_FROM_DATA_SOURCE=50000
export XENNA_RESPECT_CUDA_VISIBLE_DEVICES="1"
export HF_HOME="${HF_CACHE}"

echo "============================================"
echo "RAY_DASHBOARD_ADDRESS: $RAY_DASHBOARD_ADDRESS"
echo "RAY_TEMP_DIR: $RAY_TEMP_DIR"
echo "METRICS_DIR: $METRICS_DIR"
echo "HEAD_NODE: $HEAD_NODE_NAME ($HEAD_NODE_IP)"
echo ""
echo "To access dashboards, run on your local machine:"
echo "  ssh -L 3000:${HEAD_NODE_IP}:3000 -L 9090:${HEAD_NODE_IP}:9090 -L ${DASH_PORT}:${HEAD_NODE_IP}:${DASH_PORT} $(whoami)@${LOGIN_HOST}"
echo "  Grafana:    http://localhost:3000 (admin / admin)"
echo "  Prometheus: http://localhost:9090"
echo "  Ray:        http://localhost:${DASH_PORT}"
echo "============================================"

NODE_INIT_COMMAND="SITE=\$(python3 -c \"import nemo_curator; import os; print(os.path.dirname(nemo_curator.__file__))\") && mkdir -p \${SITE}/models \${SITE}/stages/audio/inference \${SITE}/stages/audio/io && cp ${CURATOR_DIR}/nemo_curator/models/qwen_omni.py \${SITE}/models/ && cp ${CURATOR_DIR}/nemo_curator/stages/audio/inference/qwen_omni.py \${SITE}/stages/audio/inference/ && touch \${SITE}/stages/audio/inference/__init__.py && cp ${CURATOR_DIR}/nemo_curator/stages/audio/io/nemo_tarred_reader.py \${SITE}/stages/audio/io/"

srun \
  --nodes=1 \
  -w ${HEAD_NODE_NAME} \
  --container-image=${CONTAINER} \
  --container-mounts=${CONTAINER_MOUNTS} \
  --overlap \
    bash -c "${NODE_INIT_COMMAND} && python -m nemo_curator.metrics.start_prometheus_grafana \
               --metrics_dir ${METRICS_DIR} \
               --yes && ray start \
               --head \
               --num-cpus ${NUM_CPUS_PER_NODE} \
               --num-gpus ${NUM_GPUS_PER_NODE} \
               --temp-dir ${RAY_TEMP_DIR} \
               --node-ip-address ${HEAD_NODE_IP} \
               --port ${GCS_PORT} \
               --disable-usage-stats \
               --dashboard-host 0.0.0.0 \
               --dashboard-port ${DASH_PORT} \
               --ray-client-server-port ${CLIENT_PORT} \
               --node-manager-port ${NODE_MANAGER_PORT} \
               --object-manager-port ${OBJECT_MANAGER_PORT} \
               --runtime-env-agent-port ${RUNTIME_ENV_AGENT_PORT} \
               --dashboard-agent-grpc-port ${DASHBOARD_AGENT_GRPC_PORT} \
               --metrics-export-port ${METRICS_EXPORT_PORT} \
               --block" &
sleep 60
echo "Started Prometheus, Grafana, and Ray head node"

NUM_WORKERS=$((${#NODES[@]} - 1))
for ((i = 1; i <= NUM_WORKERS; i++)); do
    NODE_I=${NODES[$i]}
    echo "Starting worker $i at $NODE_I"
    srun --nodes=1 -w ${NODE_I} --overlap \
        bash -c "mkdir -p ${METRICS_DIR} ${RAY_TEMP_DIR}"
    srun \
      --nodes=1 \
      -w ${NODE_I} \
      --container-image=${CONTAINER} \
      --container-mounts=${CONTAINER_MOUNTS} \
      --overlap \
        bash -c "${NODE_INIT_COMMAND} && ray start \
                    --address ${RAY_GCS_ADDRESS} \
                    --num-cpus ${NUM_CPUS_PER_NODE} \
                    --num-gpus ${NUM_GPUS_PER_NODE} \
                    --node-manager-port ${NODE_MANAGER_PORT} \
                    --object-manager-port ${OBJECT_MANAGER_PORT} \
                    --runtime-env-agent-port ${RUNTIME_ENV_AGENT_PORT} \
                    --dashboard-agent-grpc-port ${DASHBOARD_AGENT_GRPC_PORT} \
                    --metrics-export-port ${METRICS_EXPORT_PORT} \
                    --block;" &
    sleep 1
done
sleep 60
echo "All workers started"

RUN_COMMAND="python3 ${CURATOR_DIR}/examples/audio/qwen_omni_inprocess/run_pipeline.py --data_config ${DATA_CONFIG} --output ${OUTPUT} --corpus yodas --tensor_parallel_size 2 --batch_size 32 --max_num_seqs 16 --max_model_len 32768 --max_output_tokens 256 --gpu_memory_utilization 0.90 --prep_workers 16 --execution_mode streaming"

RUNTIME_ENV="{\"env_vars\": {\"HF_HOME\": \"${HF_CACHE}\", \"HF_HUB_OFFLINE\": \"1\", \"VLLM_WORKER_MULTIPROC_METHOD\": \"spawn\", \"VLLM_LOGGING_LEVEL\": \"ERROR\", \"TMPDIR\": \"/tmp\"}}"

echo "RUNNING: $RUN_COMMAND"
echo "RUNTIME_ENV: $RUNTIME_ENV"

srun \
  --nodes=1 \
  --overlap \
  -w ${HEAD_NODE_NAME} \
  --container-image=${CONTAINER} \
  --container-mounts=${CONTAINER_MOUNTS} \
    bash -c "${NODE_INIT_COMMAND} && ray job submit --address $RAY_DASHBOARD_ADDRESS --submission-id=$JOB_ID --runtime-env-json='${RUNTIME_ENV}' -- ${RUN_COMMAND}"
