# syntax=docker/dockerfile:1
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build docker/audio/Dockerfile.trt_llm --target granary from this checkout first.
ARG GRANARY_IMAGE=curator-granary-metadata
FROM ${GRANARY_IMAGE}

# Retain the TensorRT/NeMo stack while adding Qwen's in-process vLLM runtime.
RUN uv pip freeze --python /opt/venv/bin/python --exclude-editable > /tmp/granary-overrides.txt \
    && uv pip install --no-config --no-sources --python /opt/venv/bin/python \
        --overrides /tmp/granary-overrides.txt --only-binary :all: \
        'vllm==0.15.1' 'qwen-omni-utils==0.0.9' \
    && rm /tmp/granary-overrides.txt

# Keep the launcher and pipeline at the same checkout as this image build.
COPY . /opt/Curator

# vLLM tensor-parallel CUDA graphs need CUDA-IPC-compatible allocations.
# Override the metadata base image's expandable-segment allocator.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    VLLM_WORKER_MULTIPROC_METHOD=spawn
