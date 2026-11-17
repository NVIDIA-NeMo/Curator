# syntax=docker/dockerfile:1
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image for en_granary_pipeline.yaml and the european_*_granary_pipeline.yaml files.
# Build the base first (see README.md section 1): the repository's docker/Dockerfile with the
# audio_tensorrt and inference_server extras (public CUDA 12.9 base, Torch 2.11, vLLM 0.22, Dynamo,
# etcd/NATS). It already holds this checkout at /opt/Curator, so nothing is copied here.
ARG BASE_IMAGE=curator-base
FROM ${BASE_IMAGE}

# The eight-speaker Sortformer checkpoint needs this NeMo encoder (see the metadata tutorial).
# TensorRT builds and runs the Silero and SED engines; it is the pin of the repository's
# audio_tensorrt extra, which uv sync already installed.
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

# FFmpeg with libopus: torchcodec loads the audio for ASR and language ID through its shared libraries, and the
# pipeline decodes and writes Opus clips. The repository's docker/Dockerfile installs no FFmpeg, and
# docker/common/install_ffmpeg.sh builds one without Opus or Ogg support.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# vLLM-based entrypoints opt out of expandable segments for their own Ray cluster; see README.
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
