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

FFMPEG_BRANCH=n8.0.1
NVCODEC_VERSION=12.1.14.0

for i in "$@"; do
    case $i in
        --FFMPEG_BRANCH=?*) FFMPEG_BRANCH="${i#*=}";;
        --NVCODEC_VERSION=?*) NVCODEC_VERSION="${i#*=}";;
        *) ;;
    esac
    shift
done

# Install video dependency
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libcrypt-dev \
    libgl1 \
    libglib2.0-0t64 \
    libnuma-dev \
    libtool \
    libvpx-dev \
    nasm \
    pkg-config \
    wget \
    yasm \
    zlib1g-dev

# Install nv-codec-headers (NVENC + NVDEC bridge to the NVIDIA driver)
wget -O /tmp/nv-codec-headers.tar.gz https://github.com/FFmpeg/nv-codec-headers/releases/download/n${NVCODEC_VERSION}/nv-codec-headers-${NVCODEC_VERSION}.tar.gz
tar xzvf /tmp/nv-codec-headers.tar.gz -C /tmp/
cd /tmp/nv-codec-headers-${NVCODEC_VERSION}
make
make install

# Build FFmpeg from upstream git, branch ${FFMPEG_BRANCH} (n8.0.1), with the
# legal-approved tightly-scoped allowlist:
#   - --disable-everything strips ALL components by default (encoders,
#     decoders, muxers, demuxers, parsers, bsfs, hwaccels, filters, protocols)
#     and we re-enable only what's needed.
#   - --enable-version3 selects LGPLv3+.
#   - Encoders: only NVENC (h264/hevc/av1) and libvpx-vp9 + rawvideo.
#   - Decoders: NVDEC variants for h264/hevc/av1/vp9 + software vp8/vp9 +
#     mpeg1/2/4 + libvpx_vp9 + rawvideo. NO software h264/hevc/av1.
#   - Shared-linked: installs libav*.so to /usr/local/lib so PyAV and other
#     source-built Python deps can pkg-config against this build.
cd /tmp
git clone --depth 1 --branch "${FFMPEG_BRANCH}" --recurse-submodules https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure \
    --prefix="/usr/local" \
    --enable-shared \
    --disable-static \
    --extra-cflags="-I/usr/local/cuda/include" \
    --extra-ldflags="-L/usr/local/cuda/lib64" \
    --extra-libs="-lpthread -lm" \
    --ld="g++" \
    --enable-version3 \
    --disable-everything \
    --disable-network \
    --disable-doc \
    --disable-ffplay \
    --disable-vaapi \
    --disable-vdpau \
    --disable-dxva2 \
    --disable-libdrm \
    --enable-encoder=rawvideo,libvpx_vp9,h264_nvenc,hevc_nvenc,av1_nvenc \
    --enable-decoder=rawvideo,libvpx_vp9,vp9,vp8,h264_cuvid,hevc_cuvid,av1_cuvid,mpeg1video,mpeg2video,mpeg4 \
    --enable-muxer=mp4,rawvideo,image2pipe \
    --enable-demuxer=mov,mp4,m4a,3gp,3g2,mj2,avi,matroska,webm,image2,image2pipe \
    --enable-parser=h264,hevc,av1,vp8,vp9 \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb \
    --enable-protocol=file,pipe \
    --enable-filter=scale,format,null,copy \
    --enable-libvpx \
    --enable-cuda \
    --enable-cuvid \
    --enable-nvdec \
    --enable-nvenc \
    --enable-ffnvcodec
make -j$(nproc)
make install
ldconfig

# Clean up
cd /
rm -rf /tmp/ffmpeg*
rm -rf /tmp/nv-codec-headers*
rm -rf /var/lib/apt/lists/
