# Qwen3-Omni-30B-A3B-Instruct with NeMo Curator


## get Docker image and run server
```bash
docker pull qwenllm/qwen3-omni
export NUM_GPU=2
docker run -itd --restart unless-stopped --gpus='all' --ipc=host --shm-size=8g --ulimit memlock=-1 --name qwen3_1 -p 8201:8201 --env PYTHONPATH=/home/nkarpov/workspace/NeMo:/home/nkarpov/workspace/Curator -v /home:/home -v /mnt:/mnt qwenllm/qwen3-omni bash -lc "vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8201 --host 0.0.0.0 --dtype bfloat16 --max-model-len 65536 --allowed-local-media-path / -tp $NUM_GPU"

```

## Conda environment `curator3`
From this directory (or with paths adjusted):

```bash
# Create environment (minimal deps so conda solves quickly)
conda activate curator3

git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
pip install -e .
cd tutorials/audio/qwen_omni/
pip install -r requirements.txt
python tutorials/audio/qwen_omni/run_qwen3.py

```
