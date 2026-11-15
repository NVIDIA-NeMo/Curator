# syntax=docker/dockerfile:1
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# One image for every tutorial under tutorials/audio/granary_v2/: metadata, Indic ASR,
# Qwen Omni ASR, the text pipeline and the full pipelines in full_pipeline/.
# Build docker/audio/Dockerfile.trt_llm --target granary from this checkout first: the base
# image already holds this checkout at /opt/Curator, so nothing is copied here.
ARG GRANARY_IMAGE=curator-granary-metadata
FROM ${GRANARY_IMAGE}

# Retain the TensorRT/NeMo stack while adding Qwen's in-process vLLM runtime and the
# Dynamo driver. Freezing every installed package as an override keeps the metadata and
# Indic environments unchanged; only the packages named below (and their new
# dependencies) are added.
RUN uv pip freeze --python /opt/venv/bin/python --exclude-editable > /tmp/granary-overrides.txt \
    && uv pip install --no-config --no-sources --python /opt/venv/bin/python \
        --overrides /tmp/granary-overrides.txt --only-binary :all: \
        'vllm==0.15.1' 'qwen-omni-utils==0.0.9' \
        'ai-dynamo==1.3.1' 'nixl-cu12==1.0.0' \
    && rm /tmp/granary-overrides.txt

# Dynamo manages its newer text-model stack in isolated Ray worker environments.
RUN bash /opt/Curator/docker/common/install_etcd_nats.sh

# Serving workers (Torch 2.11 + CUDA 12.9) JIT-compile FlashInfer sampling kernels at startup
# and need nvcc plus the cuRAND headers. The audio runtime keeps its CUDA 12.8 alias;
# serving workers select this toolkit through CUDA_HOME in the YAML's runtime_env.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-nvcc-12-9=12.9.86-1 cuda-cudart-dev-12-9=12.9.79-1 \
        cuda-cccl-12-9=12.9.27-1 libcurand-dev-12-9 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --set cuda /usr/local/cuda-12.8

# vLLM 0.15.1 lists Qwen3ASRConfig in its lazy config registry but omits the module from its
# wheel; qwen-asr ships the real class.
RUN printf '%s\n' \
        '# vLLM 0.15.1 advertises this module but does not ship it.' \
        'from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig' \
        > /opt/venv/lib/python3.12/site-packages/vllm/transformers_utils/configs/qwen3_asr.py

# qwen-asr 0.0.6 targets vLLM 0.14's attention API. Drop the two keyword arguments vLLM 0.15.1
# removed (it now reads the same override from its active model configuration). The assertions
# fail the build if the installed package differs from the one this was written against.
RUN /opt/venv/bin/python - <<'PY'
from pathlib import Path

path = Path("/opt/venv/lib/python3.12/site-packages/qwen_asr/core/vllm_backend/qwen3_asr.py")
text = path.read_text()
old = (
    "        self.attn = MMEncoderAttention(\n"
    "            num_heads=self.num_local_heads,\n"
    "            head_size=self.head_dim,\n"
    "            scale=self.scaling,\n"
    "            multimodal_config=multimodal_config,\n"
    "        )"
)
assert text.count(old) == 1
text = text.replace(old, old.replace("            multimodal_config=multimodal_config,\n", ""))
old_override = "            attn_backend_override=attn_backend_override,\n"
assert text.count(old_override) == 1
path.write_text(text.replace(old_override, ""))
PY

# The base image's expandable-segments allocator (required by its TensorRT stages) is left
# unchanged. vLLM tensor-parallel CUDA graphs need CUDA-IPC-compatible allocations, so the
# vLLM-based entrypoints opt out for their own Ray cluster; see full_pipeline/README.md.
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
