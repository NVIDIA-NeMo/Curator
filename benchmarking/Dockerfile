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

########################################################################
# Base image
########################################################################

FROM nvidia/cuda:${CUDA_VER}-cudnn-devel-${LINUX_VER} AS curator_system_base

ARG CURATOR_ENV=ci
ENV CURATOR_ENVIRONMENT=${CURATOR_ENV}

ENV NVIDIA_PRODUCT_NAME="NeMo Curator"

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    curl \
    git \
    less \
    openssh-client \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3.12-dev \
    vim
WORKDIR /opt

# Install micromamba
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
RUN curl -Ls https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    micromamba shell init -s bash -r $MAMBA_ROOT_PREFIX
   
# Install uv
ENV UV_VERSION="0.8.22"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_CACHE_DIR=/opt/uv_cache
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
ENV UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=600
RUN uv venv ${UV_PROJECT_ENVIRONMENT} --system-site-packages

# heredocs only works with buildkit enabled (enabled by default),
# see https://github.com/moby/buildkit/issues/3474
COPY <<EOF /opt/venv/env.sh
export UV_PROJECT_ENVIRONMENT=/opt/venv
export PATH="/opt/venv/bin:$PATH"
export UV_LINK_MODE=copy
export PATH="/root/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=600
EOF

RUN chmod +x /opt/venv/env.sh

########################################################################
# Build and install specific Curator dependencies
########################################################################
FROM curator_system_base AS curator_setup_deps

# Install ffmpeg
COPY docker/common/install_ffmpeg.sh .
RUN bash install_ffmpeg.sh && \
    rm install_ffmpeg.sh

ARG INTERN_VIDEO_COMMIT=09d872e5093296c6f36b8b3a91fc511b76433bf7
COPY external/intern_video2_multimodal.patch .
# Clone InternVideo (Video curation dependency)
RUN git clone https://github.com/OpenGVLab/InternVideo.git && \
    cd InternVideo && \
    git checkout ${INTERN_VIDEO_COMMIT} && \
    patch -p1 < /opt/intern_video2_multimodal.patch && \
    rm /opt/intern_video2_multimodal.patch


########################################################################
# curator_benchmark image - 
#
# use cases:
# * Start a container standalone to run all Curator benchmarks. Datasets are downloaded automatically and reside only in the container.
# * Start a container to run Curator benchmarks using a YAML matrix on the host.
# * Start a container to run benchmarks on a version of Curator on the host.
# * Start a container to run benchmarks using datasets that were previously downloaded and reside on the host.
# * Start a container to download datasets to the host.
#
# docker volume mounts:
# Typical use cases (possibly even make these required mounts) will be:
# * benchmark results dir that need to exist outside of a container, mount as /results
# * datasets dir likely specific to a machine, mount as /datasets
# Advanced uses cases could include:
# * config file dirs for running new/specific runs, mount as /config
# * Curator source components, to allow for quick test/benhmarking of local changes without rebuilding the image:
#   mount as /opt/Curator/nemo_curator, /opt/Curator/tests, etc.
########################################################################

FROM curator_system_base AS curator_benchmarking

# Copy artifacts from setup_deps stage.
COPY --from=curator_setup_deps /opt /opt

# Install Curator, which includes benchmarking tools
COPY . /opt/Curator
RUN cd /opt/Curator \
 && uv sync --link-mode copy --locked --extra all --all-groups \
 && uv add /opt/InternVideo/InternVideo2/multi_modality \
 && uv cache prune \
 && uv cache clean distance

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

# Install deps for specific benchmark scripts.
# FIXME: look into a way that script authors can install their own deps so this does not need to be updated for each new script dep.
RUN apt-get install -y --no-install-recommends \
    wget

# One-time cleanup
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python", "/opt/Curator/benchmarking/run.py"]
