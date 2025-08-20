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
RUN apt-get update && apt-get install -y curl ca-certificates git \
    && curl -fsSL https://pixi.sh/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.pixi/bin:$PATH"

WORKDIR /opt

# Copy configuration files
COPY ray-curator/pyproject.toml ./
COPY ray-curator/ray_curator ./ray_curator

# Install dependencies
RUN pixi config set pinning-strategy latest-up --global && \
    pixi install -e $CURATOR_ENVIRONMENT

RUN pixi shell-hook -e $CURATOR_ENVIRONMENT -s bash > /shell-hook && \
    echo "#!/bin/bash" > /opt/entrypoint.sh && \
    cat /shell-hook >> /opt/entrypoint.sh && \
    echo 'exec "$@"' >> /opt/entrypoint.sh && \
    chmod +x /opt/entrypoint.sh

##############################################################################
# Final stage
##############################################################################

FROM cuda AS nemo_curator

WORKDIR /opt

# Install video dependency
RUN apt-get update && apt-get install -y ca-certificates git \
    libcrypt-dev \
    autoconf \
    automake \
    build-essential \
    cmake \
    libaom-dev \
    libass-dev \
    libdav1d-dev \
    libdrm-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libnuma-dev \
    libopenh264-dev \
    libtool \
    libva-dev \
    libvorbis-dev \
    libvpx-dev \
    libwebp-dev \
    pkg-config \
    texinfo \
    vainfo \
    wget \
    yasm \
    zlib1g-dev \    
    && rm -rf /var/lib/apt/lists/*

# Install ffmpeg
COPY ray-curator/docker/common/install_ffmpeg.sh .
RUN bash install_ffmpeg.sh --FFMPEG_VERSION=7.0.1 --NVCODEC_VERSION=12.1.14.0 && \
    rm install_ffmpeg.sh

# Copy pixi environment
COPY --from=build /opt/.pixi/envs /opt/.pixi/envs
COPY --from=build /opt/entrypoint.sh /opt/entrypoint.sh
COPY --from=build /opt/ray_curator /opt/ray_curator

WORKDIR /workspace

ENTRYPOINT ["/opt/entrypoint.sh"]

FROM nemo_curator as nemo_curator_dev

ENV PATH="/root/.pixi/bin:$PATH"
COPY --from=build /root/.pixi/bin/pixi /root/.pixi/bin/pixi
COPY --from=build /opt/pyproject.toml /opt/pyproject.toml
