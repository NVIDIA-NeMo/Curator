# Qwen3-Omni Audio Processing with NeMo Curator

Process audio/image/text data through Qwen3-Omni using NeMo Curator's pipeline.

## Input format

The pipeline expects a JSONL file where each row has some or all of these fields:

```json
{"audio_url": "/path/to/audio.wav", "text": "Describe what you hear.", "system_text": "You are an audio assistant."}
{"audio_url": "/path/to/speech.mp3", "text": "Transcribe this audio exactly."}
{"image_url": "/path/to/photo.jpg", "text": "What do you see?"}
```

- `audio_filepath` — local path, HTTP URL, or `s3://` / `ais://` object storage path
- `image_url` — local path, HTTP URL, or `s3://` / `ais://` object storage path
- `text` — the prompt / question
- `system_text` — optional system prompt

Output adds a `predicted_text` column with the model's response.


## Remote storage (AIStore / S3)

Audio and image files can be read from AIStore (`ais://`) or S3-compatible
(`s3://`) object storage. The pipeline uses the
[AIStore Python SDK](https://github.com/NVIDIA/aistore) to access remote
objects through an AIS gateway.

### Setup

1. Set the AIS gateway endpoint and AuthN token:
   ```bash
   export AIS_ENDPOINT=http://ais-gateway:51080
   export AIS_AUTHN_TOKEN="<your-token>"
   ```

2. Create an `.s3cfg` config (used for fallback endpoint/token resolution):
   ```ini
   [default]
   use_https = True
   host_base = ais-gateway:51080
   ```

3. Reference S3/AIS paths in your JSONL manifest:
   ```json
   {"audio_filepath": "s3://my-bucket/audio/clip.wav", "text": "Transcribe."}
   {"audio_filepath": "ais://my-bucket/audio/clip.wav", "text": "Transcribe."}
   ```

4. Pass `--s3cfg` when running the pipeline:
   ```bash
   docker run --gpus all --ipc=host --shm-size=8g \
     -v /path/to/data:/data \
     -v ~/.s3cfg:/root/.s3cfg:ro \
     -e AIS_ENDPOINT=http://ais-gateway:51080 \
     -e AIS_AUTHN_TOKEN="$AIS_AUTHN_TOKEN" \
     -e NUM_GPU=2 \
     curator-qwen3-omni \
     --input_manifest /data/input.jsonl \
     --output_path /data/output/ \
     --s3cfg '/root/.s3cfg[default]'
   ```

`s3://` paths are accessed via the AIS gateway as remote S3 backend
(`provider=aws`). `ais://` paths access native AIS buckets directly.


## Option 1: Pre-built image from nvcr.io (recommended)

A ready-to-run image with vLLM + Curator pre-installed is available at:

```
nvcr.io/nvidian/curator-qwen3-omni:latest
```

### Run (2 GPUs, full bf16 model)
```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -v /path/to/data:/data \
  -e NUM_GPU=2 \
  nvcr.io/nvidian/curator-qwen3-omni:latest \
  --input_manifest /data/input.jsonl --output_path /data/output/
```

### Run (1 GPU, AWQ 4-bit quantized)
If you only have a single GPU (48GB+), use the community AWQ 4-bit variant:
```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -v /path/to/data:/data \
  -e VLLM_MODEL=cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit \
  -e VLLM_MAX_MODEL_LEN=4096 \
  nvcr.io/nvidian/curator-qwen3-omni:latest \
  --input_manifest /data/input.jsonl --output_path /data/output/
```

The entrypoint automatically starts vLLM, waits for it to be healthy, runs the
pipeline, and shuts down.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_MODEL` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Model ID (HuggingFace) |
| `VLLM_PORT` | `8200` | Server port |
| `VLLM_DTYPE` | `bfloat16` | Data type |
| `VLLM_MAX_MODEL_LEN` | `65536` | Max model length |
| `NUM_GPU` | `1` | Tensor parallel size |
| `HEALTH_TIMEOUT` | `1200` | Health check timeout (seconds) |

### GPU requirements

| Model variant | VRAM needed | GPUs |
|---|---|---|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` (bf16) | ~47 GB per GPU | 2x 24GB+ or 1x 80GB |
| `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` | ~16 GB | 1x 48GB |


## Option 2: Build the image yourself

```bash
# From the Curator repo root
docker build -t curator-qwen3-omni -f tutorials/audio/qwen_omni/Dockerfile .

docker run --gpus all --ipc=host --shm-size=8g \
  -v /path/to/data:/data \
  -e NUM_GPU=2 \
  curator-qwen3-omni \
  --input_manifest /data/input.jsonl --output_path /data/output/
```


## Option 3: Separate environments

Run the vLLM server and Curator pipeline in separate environments. Useful for
debugging or when you already have a running vLLM server.

### Start the vLLM server
```bash
docker pull qwenllm/qwen3-omni
export NUM_GPU=2
docker run -itd --restart unless-stopped --gpus='all' --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --name qwen3_1 -p 8201:8201 \
  -v /home:/home -v /mnt:/mnt \
  qwenllm/qwen3-omni bash -lc \
  "vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8201 --host 0.0.0.0 \
   --dtype bfloat16 --max-model-len 65536 --allowed-local-media-path / -tp $NUM_GPU"
```

### Install Curator and run the pipeline
```bash
conda activate curator3

git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
pip install -e .
pip install -r tutorials/audio/qwen_omni/requirements.txt
python tutorials/audio/qwen_omni/run_pipeline.py \
  --input_manifest /path/to/input.jsonl --host localhost --port 8201
```


## Option 4: Programmatic server launch (no Docker)

If you have both vLLM and Curator installed (possibly in different Python
environments), use the `--start-server` flag to launch vLLM as a subprocess:

```bash
python tutorials/audio/qwen_omni/run_pipeline.py \
  --start-server \
  --vllm-python /usr/bin/python3 \
  --tensor-parallel-size 2 \
  --input_manifest /path/to/input.jsonl
```

This starts vLLM using the specified Python executable, runs the pipeline,
and shuts down the server automatically.


## Option 5: Hosted API (no GPU needed)

Use the NVIDIA inference API with a hosted model (e.g. Gemini) instead of
running vLLM locally. Audio is sent as base64-encoded `input_audio` format.

```bash
export API_KEY="$NVIDIA_API_KEY"
python tutorials/audio/qwen_omni/run_hosted.py \
  --input_manifest /path/to/input.jsonl \
  --model-name gcp/google/gemini-2.5-pro
```

Note: Audio files must be local paths (they are base64-encoded before sending).
Supported hosted models with audio: Gemini 2.5 Pro, Gemini 3 Pro.
