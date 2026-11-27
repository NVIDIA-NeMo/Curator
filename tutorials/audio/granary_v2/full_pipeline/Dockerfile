# syntax=docker/dockerfile:1
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image for every pipeline except the two Indic ones. Build the base first (README.md section 1): docker/Dockerfile
# with the audio_tensorrt and inference_server extras; it already holds this checkout at /opt/Curator.
ARG BASE_IMAGE=curator-base
FROM ${BASE_IMAGE}

# The eight-speaker Sortformer checkpoint needs this NeMo encoder (see the metadata tutorial).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs \
    && git -C /tmp lfs install --system \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && git init /tmp/nemo-reference \
    && git -C /tmp/nemo-reference remote add origin https://github.com/ipmedenn/NeMo.git \
    && git -C /tmp/nemo-reference fetch --depth 1 origin 5d56c27c0e80f4cd8c93b28bacca5f98440a45f9 \
    && git -C /tmp/nemo-reference checkout --detach FETCH_HEAD \
    && uv pip uninstall --python /opt/venv/bin/python nemo-toolkit \
    && uv pip install --no-config --no-sources --python /opt/venv/bin/python \
        --no-deps --no-build-isolation /tmp/nemo-reference \
    && rm -rf /tmp/nemo-reference

# FFmpeg with libopus for torchcodec and the Opus clips; the base has none, and install_ffmpeg.sh lacks Opus/Ogg.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# vLLM must spawn, not fork, its CUDA workers.
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
