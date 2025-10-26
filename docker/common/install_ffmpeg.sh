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

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

FFMPEG_VERSION=7.0.1
NVCODEC_VERSION=12.1.14.0

for i in "$@"; do
    case $i in
        --FFMPEG_VERSION=?*) FFMPEG_VERSION="${i#*=}";;
        --NVCODEC_VERSION=?*) NVCODEC_VERSION="${i#*=}";;
        *) ;;
    esac
    shift
done

# Install video dependency
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
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
    vainfo \
    wget \
    yasm \
    zlib1g-dev

# Install NVCODEC
wget -O /tmp/nv-codec-headers.tar.gz https://github.com/FFmpeg/nv-codec-headers/releases/download/n${NVCODEC_VERSION}/nv-codec-headers-${NVCODEC_VERSION}.tar.gz
tar xzvf /tmp/nv-codec-headers.tar.gz -C /tmp/
cd /tmp/nv-codec-headers-${NVCODEC_VERSION}
make
make install

# INSTALL FFMPEG
wget -O /tmp/ffmpeg-snapshot.tar.bz2 https://www.ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2
tar xjvf /tmp/ffmpeg-snapshot.tar.bz2 -C /tmp/
cd /tmp/ffmpeg-${FFMPEG_VERSION}
PATH="/usr/local/cuda/bin:$PATH" ./configure \
    --prefix=/usr/local \
    --enable-nonfree \
    --enable-cuda-nvcc \
    --enable-libnpp \
    --enable-libopenh264 \
    --enable-libaom \
    --enable-libdav1d \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libwebp \
    --enable-vaapi \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --extra-libs=-lpthread \
    --extra-libs=-lm \
    --disable-static \
    --enable-shared \
    --disable-doc \
    --disable-debug
make -j$(nproc)
make install
ldconfig

# Clean up
cd /
rm -rf /tmp/ffmpeg*
rm -rf /tmp/nv-codec-headers*
rm -rf /var/lib/apt/lists/
