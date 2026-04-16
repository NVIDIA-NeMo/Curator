# Full E2E HIFI pipeline: SED + diarization + transcription cascade + speaker ID.
#
# Uses qwen3-omni image (vLLM) as base, Curator + NeMo in isolated venv.
#
# Build:
#   docker build -t curator-hifi-pipeline -f tutorials/audio/hifi_pipeline/Dockerfile .
#
# Run:
#   docker run --gpus all --ipc=host --shm-size=8g \
#     -v /path/to/data:/data -v /path/to/models:/models \
#     curator-hifi-pipeline \
#     --input_manifest /data/manifest.jsonl \
#     --stages sed,sed_post,segment,diarize,transcribe,embed,cluster \
#     --sed_checkpoint /models/Cnn14_DecisionLevelMax.pth \
#     --language Ru

FROM qwenllm/qwen3-omni

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv git curl libsndfile1 ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/curator_venv && \
    /opt/curator_venv/bin/pip install --upgrade pip

WORKDIR /opt/Curator
COPY pyproject.toml /opt/Curator/
COPY nemo_curator/__init__.py nemo_curator/package_info.py /opt/Curator/nemo_curator/

RUN /opt/curator_venv/bin/pip install --no-cache-dir .

COPY . /opt/Curator
RUN /opt/curator_venv/bin/pip install --no-cache-dir . && \
    /opt/curator_venv/bin/pip install --no-cache-dir \
    -r tutorials/audio/hifi_pipeline/requirements.txt && \
    /opt/curator_venv/bin/pip install --no-cache-dir \
    nemo_toolkit[asr] huggingface_hub openai && \
    /opt/curator_venv/bin/pip install --no-cache-dir \
    torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

COPY tutorials/audio/hifi_pipeline/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
