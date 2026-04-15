#!/bin/bash
#SBATCH --job-name=tts-cascade-ytc-ru
#SBATCH --account=nemotron_speech_asr
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=400GB
#SBATCH --time=00:30:00

set -x

CONTAINER="/lustre/fsw/portfolios/convai/users/mmkrtchyan/containers/curator-nightly-lhotse.sqsh"
CURATOR_DIR="/lustre/fsw/portfolios/convai/users/mmkrtchyan/projects/ASR/Curator"
OUTPUT_DIR="/lustre/fsw/portfolios/convai/users/mmkrtchyan/outputs/tts_cascade_ytc_ru"
DATA_CONFIG="${CURATOR_DIR}/scripts/ytc_ru.yaml"
HF_CACHE="/lustre/fsw/portfolios/convai/users/mmkrtchyan/.cache/huggingface"

mkdir -p "${OUTPUT_DIR}"

read -r -d '' CMD <<'EOFCMD' || true
export HF_HOME=HF_CACHE_PH
export HF_TOKEN=HF_TOKEN_PH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=ERROR
export RAY_TMPDIR=/tmp
export TMPDIR=/tmp
export XENNA_RESPECT_CUDA_VISIBLE_DEVICES=1
export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080
export AIS_AUTHN_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVzdGVycyI6bnVsbCwiYWRtaW4iOnRydWUsImlzcyI6Imh0dHBzOi8vbG9jYWxob3N0OjUyMDAxIiwic3ViIjoiYWRtaW4iLCJleHAiOjI0MDY1NzY3ODgsImlhdCI6MTc3NTg1Njc4OH0.NuwKfhdXBaOXYxx4eTataX7XWP1wEOwtopXhFGzppkw

/opt/venv/bin/pip install -q cosmos-xenna loguru pynvml qwen-omni-utils 2>/dev/null
SITE=$(/opt/venv/bin/python3 -c "import nemo_curator; import os; print(os.path.dirname(nemo_curator.__file__))")
echo "SITE: ${SITE}"
mkdir -p ${SITE}/models ${SITE}/stages/audio/inference ${SITE}/stages/audio/io ${SITE}/stages/audio/alm
cp CURATOR_DIR_PH/nemo_curator/models/qwen_omni.py ${SITE}/models/
cp CURATOR_DIR_PH/nemo_curator/stages/audio/inference/qwen_omni.py ${SITE}/stages/audio/inference/
cp CURATOR_DIR_PH/nemo_curator/stages/audio/inference/transcription_cascade_inprocess.py ${SITE}/stages/audio/inference/
touch ${SITE}/stages/audio/inference/__init__.py
cp CURATOR_DIR_PH/nemo_curator/stages/audio/io/nemo_tarred_reader.py ${SITE}/stages/audio/io/
cp CURATOR_DIR_PH/nemo_curator/stages/audio/alm/alm_manifest_writer.py ${SITE}/stages/audio/alm/
cp -r CURATOR_DIR_PH/nemo_curator/stages/audio/request ${SITE}/stages/audio/

/opt/venv/bin/python3 CURATOR_DIR_PH/tutorials/audio/tts_pipeline/run_pipeline.py \
    --input_manifest dummy \
    --data_config DATA_CONFIG_PH \
    --output_dir OUTPUT_DIR_PH \
    --stages transcribe \
    --language Ru \
    --omni_model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --tensor_parallel_size 2 \
    --max_tokens 512 \
    --temperature 0.0 \
    --batch_size 8 \
    --no_ray_local
EOFCMD

CMD="${CMD//CURATOR_DIR_PH/${CURATOR_DIR}}"
CMD="${CMD//DATA_CONFIG_PH/${DATA_CONFIG}}"
CMD="${CMD//OUTPUT_DIR_PH/${OUTPUT_DIR}}"
CMD="${CMD//HF_CACHE_PH/${HF_CACHE}}"
CMD="${CMD//HF_TOKEN_PH/${HF_TOKEN}}"

srun --export=ALL \
     --container-image="${CONTAINER}" \
     --container-mounts="/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw" \
     --container-writable \
     bash -c "${CMD}"
