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
# gnupg upgrade Address CVE-2025-68973
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-dev \
    python3 \
    python3-dev \
    python3-venv \
    python-is-python3\
    curl \
    git \
    vim && \
    apt-get install -y --only-upgrade gnupg && \
    apt-get clean && \
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
RUN uv venv ${UV_PROJECT_ENVIRONMENT} --system-site-packages --seed

FROM build AS nemo_curator_dep

# Install FFmpeg
COPY docker/common/install_ffmpeg.sh .
RUN bash install_ffmpeg.sh && \
    rm install_ffmpeg.sh

FROM nemo_curator_dep AS nemo_curator

ARG CURATOR_EXTRA=all
ARG CURATOR_GROUPS="--all-groups"
WORKDIR /opt/Curator

# Selective file copying before dependency installation for better layer caching
COPY pyproject.toml uv.lock /opt/Curator/
COPY nemo_curator/__init__.py nemo_curator/package_info.py /opt/Curator/nemo_curator/

# Install Curator
RUN uv sync --link-mode copy --locked --extra ${CURATOR_EXTRA} ${CURATOR_GROUPS} --no-cache && \
    find /opt/venv -type d -path "*ray/_private/runtime_env/agent/thirdparty_files/aiohttp*" -exec rm -rf {} + && \
    # Address GHSA-72hv-8253-57qq: jackson-core bundled in ray_dist.jar (jackson-core 2.16.1 < 2.18.6)
    # Ray's Java support is not used; deleting the JAR removes the vulnerable jackson-core classes.
    # TODO: Remove once Ray upgrades bundled jackson-core to >=2.18.6 (https://github.com/ray-project/ray/issues/35949)
    find /opt/venv -name "ray_dist.jar" -delete && \
    if find /opt/venv -name "ray_dist.jar" | grep -q .; then \
        echo "ERROR: GHSA-72hv-8253-57qq not fixed — ray_dist.jar still present after deletion" && exit 1; \
    fi

# Patch wandb-core: bump google.golang.org/grpc from 1.79.2 to 1.79.3 (CVE fix)
ARG TARGETARCH
RUN GRPC_VERSION=1.79.3 && \
    GO_VERSION=1.26.1 && \
    WANDB_VERSION=$(python3 -c "import wandb; print(wandb.__version__)") && \
    WANDB_CORE_BIN=/opt/venv/lib/python3.12/site-packages/wandb/bin/wandb-core && \
    curl -fsSL "https://dl.google.com/go/go${GO_VERSION}.linux-${TARGETARCH}.tar.gz" | tar -C /tmp -xz && \
    export PATH="/tmp/go/bin:$PATH" && \
    export GOPATH=/tmp/gopath && \
    git clone --depth 1 --branch "v${WANDB_VERSION}" https://github.com/wandb/wandb.git /tmp/wandb-src && \
    cd /tmp/wandb-src/core && \
    go get google.golang.org/grpc@v${GRPC_VERSION} && \
    go mod tidy && \
    go mod vendor && \
    CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build -trimpath -ldflags="-s -w" \
        -o "${WANDB_CORE_BIN}" ./cmd/wandb-core/ && \
    rm -rf /tmp/wandb-src /tmp/go /tmp/gopath

COPY . /opt/Curator

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
