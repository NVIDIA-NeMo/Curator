# Single-container Dockerfile for Qwen3-Omni + NeMo Curator.
#
# Uses the Qwen3-Omni image (which has vLLM + model deps) as base,
# and installs Curator in an isolated virtualenv to avoid dependency
# conflicts (e.g. transformers version mismatch).
#
# Build (from repo root):
#   docker build -t curator-qwen3-omni -f tutorials/audio/qwen_omni/Dockerfile .
#
# Run:
#   docker run --gpus all --ipc=host --shm-size=8g \
#     -v /path/to/data:/data \
#     curator-qwen3-omni \
#     --input_path /data/input.jsonl --output_path /data/output/

FROM qwenllm/qwen3-omni

# Install system packages needed for the virtualenv and Curator install
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    git \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create an isolated virtualenv for Curator.
# IMPORTANT: Do NOT use --system-site-packages — Curator needs its own
# transformers<=4.55.2 while the system Python has transformers>=4.55
# for vLLM.
RUN python3 -m venv /opt/curator_venv

# Copy dependency metadata first for better layer caching
WORKDIR /opt/Curator
COPY pyproject.toml /opt/Curator/
COPY nemo_curator/__init__.py nemo_curator/package_info.py /opt/Curator/nemo_curator/

# Pre-install dependencies (cached layer — only re-runs if pyproject.toml changes)
RUN /opt/curator_venv/bin/pip install --no-cache-dir .

# Copy the full repo and reinstall (picks up actual code)
COPY . /opt/Curator
RUN /opt/curator_venv/bin/pip install --no-cache-dir . && \
    /opt/curator_venv/bin/pip install --no-cache-dir \
    -r tutorials/audio/qwen_omni/requirements.txt

# Copy and set up entrypoint
COPY tutorials/audio/qwen_omni/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
