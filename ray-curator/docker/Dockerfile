# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

ARG CUDA_VER=12.8.1
ARG LINUX_VER=ubuntu24.04

FROM nvidia/cuda:${CUDA_VER}-cudnn-devel-${LINUX_VER} AS cuda


########################################################################
# Base image
########################################################################

FROM cuda AS build

ARG CURATOR_ENV=dev
ENV CURATOR_ENVIRONMENT=${CURATOR_ENV}

ENV NVIDIA_PRODUCT_NAME="NeMo Curator"

# Install pixi
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
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

WORKDIR /opt

# Install uv
ENV UV_VERSION="0.7.2"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_CACHE_DIR=/opt/uv_cache
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
ENV UV_LINK_MODE=copy
RUN uv venv ${UV_PROJECT_ENVIRONMENT} --system-site-packages

# Copy configuration files
COPY ray-curator /opt/ray-curator

# Install dependencies
RUN cd ray-curator && \
    uv sync --link-mode copy --locked --extra all --all-groups

COPY <<EOF /opt/venv/env.sh
export UV_PROJECT_ENVIRONMENT=/opt/venv
export PATH="/opt/venv/bin:$PATH"
export UV_LINK_MODE=copy
export PATH="/root/.local/bin:$PATH"
EOF

RUN chmod +x /opt/venv/env.sh
