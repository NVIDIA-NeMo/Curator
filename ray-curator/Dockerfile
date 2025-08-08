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

ENV NVIDIA_PRODUCT_NAME="NeMo Curator"

# Install pixi
RUN apt-get update && apt-get install -y curl ca-certificates git \
    && curl -fsSL https://pixi.sh/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.pixi/bin:$PATH"

WORKDIR /tmp

# Copy configuration files
COPY ray-curator/pyproject.toml ./

# Install dependencies
RUN pixi install

# Create activation script
RUN pixi shell-hook -s bash > /shell-hook && \
    echo "#!/bin/bash" > /app/entrypoint.sh && \
    cat /shell-hook >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

##############################################################################
# Final stage
##############################################################################

FROM cuda AS nemo_curator

WORKDIR /opt

RUN apt-get update && apt-get install -y ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Copy pixi environment
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /app/entrypoint.sh /app/entrypoint.sh

WORKDIR /workspace

FROM nemo_curator AS nemo_curator_dev

RUN cd /opt/ray-curator && \
    pxix install -e dev
