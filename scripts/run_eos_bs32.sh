#!/bin/bash
#SBATCH --job-name=qwen-bs32
#SBATCH --account=llmservice_nemo_speechlm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:30:00

set -x

CONTAINER="/lustre/fsw/llmservice_nemo_speechlm/users/nkoluguri/containers/curator-nightly-lhotse.sqsh"
CURATOR_DIR="/lustre/fsw/llmservice_nemo_speechlm/users/mmkrtchyan/projects/Curator"
OUTPUT_DIR="/lustre/fsw/llmservice_nemo_speechlm/users/mmkrtchyan/outputs"
OUTPUT="${OUTPUT_DIR}/qwen_omni_yodas_bs32.jsonl"
DATA_CONFIG="/lustre/fsw/llmservice_nemo_speechlm/users/nkoluguri/projects/transformer_revamp/scripts/fc/am-fl-gc-ll-mc-mm-yt-yo_En-d0.5_v1.1.yaml"
HF_CACHE="/lustre/fsw/llmservice_nemo_speechlm/users/mmkrtchyan/.cache/huggingface"

mkdir -p "${OUTPUT_DIR}"

read -r -d '' CMD <<'EOFCMD' || true
export HF_HOME=HF_CACHE_PH
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=ERROR
export RAY_TMPDIR=/tmp
export TMPDIR=/tmp

SITE=$(python3 -c "import nemo_curator; import os; print(os.path.dirname(nemo_curator.__file__))")
mkdir -p ${SITE}/models ${SITE}/stages/audio/inference ${SITE}/stages/audio/io
cp CURATOR_DIR_PH/nemo_curator/models/qwen_omni.py ${SITE}/models/
cp CURATOR_DIR_PH/nemo_curator/stages/audio/inference/qwen_omni.py ${SITE}/stages/audio/inference/
touch ${SITE}/stages/audio/inference/__init__.py
cp CURATOR_DIR_PH/nemo_curator/stages/audio/io/nemo_tarred_reader.py ${SITE}/stages/audio/io/

python3 CURATOR_DIR_PH/examples/audio/qwen_omni_inprocess/run_pipeline.py \
    --data_config DATA_CONFIG_PH \
    --output OUTPUT_PH \
    --tensor_parallel_size 2 \
    --batch_size 32 \
    --max_num_seqs 16 \
    --max_model_len 32768 \
    --max_output_tokens 256 \
    --gpu_memory_utilization 0.90 \
    --prep_workers 16 \
    --corpus yodas \
    --execution_mode streaming
EOFCMD

CMD="${CMD//CURATOR_DIR_PH/${CURATOR_DIR}}"
CMD="${CMD//DATA_CONFIG_PH/${DATA_CONFIG}}"
CMD="${CMD//OUTPUT_PH/${OUTPUT}}"
CMD="${CMD//HF_CACHE_PH/${HF_CACHE}}"

srun --export=ALL \
     --container-image="${CONTAINER}" \
     --container-mounts="/lustre:/lustre" \
     bash -c "${CMD}"
