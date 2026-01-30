# syntax=docker/dockerfile:1
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_VER=12.8.1
ARG LINUX_VER=ubuntu24.04

FROM nvidia/cuda:${CUDA_VER}-cudnn-devel-${LINUX_VER} AS cuda


########################################################################
# Base image
########################################################################

FROM cuda AS build

ARG CURATOR_ENV=ci
ENV CURATOR_ENVIRONMENT=${CURATOR_ENV}

ENV NVIDIA_PRODUCT_NAME="NeMo Curator"

# Install base dependency
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-dev \
    python3 \
    python3-dev \
    python3-venv \
    python-is-python3\
    curl \
    git \
    vim \
    build-essential \
    pkg-config \
    libpcre2-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Install uv
ENV UV_VERSION="0.8.22"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_CACHE_DIR=/opt/uv_cache
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
ENV UV_LINK_MODE=copy
RUN uv venv ${UV_PROJECT_ENVIRONMENT} --system-site-packages

# Install Rust for code_annotation
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
ENV CARGO_HOME="/root/.cargo"

FROM build AS nemo_curator

# Install FFmpeg (Video curation dependency)
COPY docker/common/install_ffmpeg.sh .
RUN bash install_ffmpeg.sh && rm install_ffmpeg.sh

# Clone InternVideo (Video curation dependency)
ARG INTERN_VIDEO_COMMIT=09d872e5093296c6f36b8b3a91fc511b76433bf7
COPY external/intern_video2_multimodal.patch .
RUN git clone https://github.com/OpenGVLab/InternVideo.git && \
    cd InternVideo && \
    git checkout ${INTERN_VIDEO_COMMIT} && \
    patch -p1 < /opt/intern_video2_multimodal.patch && \
    rm /opt/intern_video2_multimodal.patch

# Install Curator
COPY . /opt/Curator

# Ensure code_annotation submodules are present (clone if not copied)
ARG HYPERPOLYGLOT_COMMIT=515af8f43a7e865408daf5b8578e7e880a3894ac
ARG RUST_CODE_ANALYSIS_COMMIT=efe98948323d4965348559ca607838746e6d7e4c
RUN cd /opt/Curator && \
    if [ ! -f "nemo_curator/code_annotation/external/hyperpolyglot/Cargo.toml" ]; then \
        rm -rf nemo_curator/code_annotation/external/hyperpolyglot && \
        git clone https://github.com/VibhuJawa/hyperpolyglot.git nemo_curator/code_annotation/external/hyperpolyglot && \
        cd nemo_curator/code_annotation/external/hyperpolyglot && git checkout ${HYPERPOLYGLOT_COMMIT}; \
    fi && \
    cd /opt/Curator && \
    if [ ! -f "nemo_curator/code_annotation/external/software_metrics/rust-code-analysis/Cargo.toml" ]; then \
        rm -rf nemo_curator/code_annotation/external/software_metrics/rust-code-analysis && \
        git clone https://github.com/mozilla/rust-code-analysis.git nemo_curator/code_annotation/external/software_metrics/rust-code-analysis && \
        cd nemo_curator/code_annotation/external/software_metrics/rust-code-analysis && git checkout ${RUST_CODE_ANALYSIS_COMMIT}; \
    fi

# Install Python dependencies (code_annotation Rust extension is built automatically via --extra code)
RUN --mount=type=cache,target=/opt/uv_cache \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/opt/Curator/nemo_curator/code_annotation/rust/target \
    cd /opt/Curator && \
    uv sync --link-mode copy --locked --extra all --all-groups && \
    uv add /opt/InternVideo/InternVideo2/multi_modality

COPY <<EOF /opt/venv/env.sh
export UV_PROJECT_ENVIRONMENT=/opt/venv
export PATH="/opt/venv/bin:$PATH"
export UV_LINK_MODE=copy
export PATH="/root/.local/bin:$PATH"
EOF

RUN chmod +x /opt/venv/env.sh

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"
