#!/bin/bash
# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

set -euo pipefail

FFMPEG_VERSION=8.0.1
NVCODEC_VERSION=12.1.14.0
CHECK_ONLY=0

usage() {
    cat <<'EOF'
Usage: install_ffmpeg.sh [--check] [--FFMPEG_VERSION=<version>] [--NVCODEC_VERSION=<ver>]

Installs the FFmpeg capabilities needed by Curator video workflows and
benchmarks. The build includes ffprobe, NVENC/NVDEC, VP8/VP9, software
h264/hevc/av1 decoders, and the libopenh264 software h264 encoder.

Options:
  --check                    Verify the required FFmpeg capabilities without installing.
  --FFMPEG_VERSION=<version> FFmpeg upstream release version (default: 8.0.1).
  --NVCODEC_VERSION=<ver>    nv-codec-headers release version (default: 12.1.14.0).
  -h, --help                 Show this help.

License notice:
  This script links Cisco OpenH264 through the libopenh264 package. You are
  responsible for any license obligations imposed by the resulting binaries.
EOF
}

for arg in "$@"; do
    case $arg in
        --check)                  CHECK_ONLY=1 ;;
        --FFMPEG_VERSION=?*)      FFMPEG_VERSION="${arg#*=}" ;;
        --NVCODEC_VERSION=?*)     NVCODEC_VERSION="${arg#*=}" ;;
        -h|--help)                usage; exit 0 ;;
        *)                        echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

check_ffmpeg() {
    local status=0
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "ERROR: ffmpeg not found on PATH" >&2
        status=1
    fi
    if ! command -v ffprobe >/dev/null 2>&1; then
        echo "ERROR: ffprobe not found on PATH" >&2
        status=1
    fi
    if [ "$status" -ne 0 ]; then
        return "$status"
    fi

    local encoders
    local decoders
    local encoder
    local decoder
    encoders=$(ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}')
    decoders=$(ffmpeg -hide_banner -decoders 2>/dev/null | awk '{print $2}')

    for encoder in rawvideo libvpx_vp9 h264_nvenc hevc_nvenc av1_nvenc libopenh264; do
        if ! printf '%s\n' "$encoders" | grep -Fx -- "$encoder" >/dev/null; then
            echo "ERROR: ffmpeg encoder not found: $encoder" >&2
            status=1
        fi
    done
    for decoder in rawvideo libvpx_vp9 vp9 vp8 h264_cuvid hevc_cuvid av1_cuvid mpeg1video mpeg2video mpeg4 h264 hevc av1; do
        if ! printf '%s\n' "$decoders" | grep -Fx -- "$decoder" >/dev/null; then
            echo "ERROR: ffmpeg decoder not found: $decoder" >&2
            status=1
        fi
    done

    if [ "$status" -eq 0 ]; then
        echo "FFmpeg dependency check passed."
    fi
    return "$status"
}

if check_ffmpeg; then
    exit 0
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
    exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must be run as root to install FFmpeg dependencies." >&2
    exit 1
fi

echo "==> install_ffmpeg.sh: building ffmpeg ${FFMPEG_VERSION}"
echo "    Decoders: h264/hevc/av1 software + NVDEC variants"
echo "    Encoders: NVENC variants + libvpx-vp9 + libopenh264"
echo "    NOTE: OpenH264 license obligations are the user's responsibility."

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    libcrypt-dev \
    libnuma-dev \
    libopenh264-dev \
    libtool \
    libvpx-dev \
    nasm \
    pkg-config \
    wget \
    yasm \
    zlib1g-dev

if [ ! -f /usr/local/include/ffnvcodec/dynlink_loader.h ]; then
    wget -O /tmp/nv-codec-headers.tar.gz \
        "https://github.com/FFmpeg/nv-codec-headers/releases/download/n${NVCODEC_VERSION}/nv-codec-headers-${NVCODEC_VERSION}.tar.gz"
    tar xzf /tmp/nv-codec-headers.tar.gz -C /tmp/
    (cd "/tmp/nv-codec-headers-${NVCODEC_VERSION}" && make && make install)
fi

cd /tmp
rm -rf "ffmpeg-${FFMPEG_VERSION}" ffmpeg-snapshot.tar.bz2
wget -O /tmp/ffmpeg-snapshot.tar.bz2 \
    "https://www.ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2"
tar xjvf /tmp/ffmpeg-snapshot.tar.bz2 -C /tmp/
cd "/tmp/ffmpeg-${FFMPEG_VERSION}"

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
    --enable-encoder=rawvideo,libvpx_vp9,h264_nvenc,hevc_nvenc,av1_nvenc,libopenh264 \
    --enable-decoder=rawvideo,libvpx_vp9,vp9,vp8,h264_cuvid,hevc_cuvid,av1_cuvid,mpeg1video,mpeg2video,mpeg4,h264,hevc,av1 \
    --enable-muxer=mp4,rawvideo,image2pipe \
    --enable-demuxer=mov,mp4,m4a,3gp,3g2,mj2,avi,matroska,webm,image2,image2pipe \
    --enable-parser=h264,hevc,av1,vp8,vp9 \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb \
    --enable-protocol=file,pipe \
    --enable-filter=scale,format,null,copy \
    --enable-libvpx \
    --enable-libopenh264 \
    --enable-cuda \
    --enable-cuvid \
    --enable-nvdec \
    --enable-nvenc \
    --enable-ffnvcodec

make -j"$(nproc)"
make install
ldconfig

cd /
rm -rf /tmp/ffmpeg* /tmp/nv-codec-headers* /var/lib/apt/lists/*
check_ffmpeg
