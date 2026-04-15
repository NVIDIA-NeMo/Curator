#!/bin/bash
#SBATCH --job-name=tts-cascade-ytc-ru
#SBATCH --account=nemotron_speech_asr
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=128GB
#SBATCH --time=00:30:00

set -x

CONTAINER="/lustre/fs11/portfolios/convai/projects/convai_convaird_nemo-speech/users/mmkrtchyan/containers/curator-qwen3-omni.sqsh"
CURATOR_DIR="/lustre/fsw/portfolios/convai/users/mmkrtchyan/projects/ASR/Curator"
OUTPUT_DIR="/lustre/fsw/portfolios/convai/users/mmkrtchyan/outputs/tts_cascade_ytc_ru"
DATA_CONFIG="${CURATOR_DIR}/scripts/ytc_ru.yaml"
HF_CACHE="/lustre/fsw/portfolios/convai/users/mmkrtchyan/.cache/huggingface"

mkdir -p "${OUTPUT_DIR}"

srun --export=ALL \
     --container-image="${CONTAINER}" \
     --container-mounts="/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw" \
     --container-writable \
     bash -c "
export HF_HOME=${HF_CACHE}
export HF_TOKEN=${HF_TOKEN}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=ERROR
export TMPDIR=/tmp
export XENNA_RESPECT_CUDA_VISIBLE_DEVICES=1
export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080
export SKIP_VLLM=1

pip install -q soundfile lhotse librosa pynvml 2>/dev/null
cd ${CURATOR_DIR} && pip install --no-deps . 2>/dev/null

python3 ${CURATOR_DIR}/tutorials/audio/tts_pipeline/run_pipeline.py \
    --input_manifest dummy \
    --data_config ${DATA_CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --stages transcribe \
    --language Ru \
    --omni_model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --max_tokens 512 \
    --temperature 0.0 \
    --batch_size 32 \
    --no_ray_local
"
