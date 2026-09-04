#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

FFMPEG_VERSION="${CURATOR_BENCHMARK_FFMPEG_VERSION:-8.0.1}"
NVCODEC_VERSION="${CURATOR_BENCHMARK_NVCODEC_VERSION:-12.1.14.0}"
video_transcode_encoder="${CURATOR_BENCHMARK_VIDEO_ENCODER:-libopenh264}"

ffmpeg_supports_video_encoder() {
  local ffmpeg_bin="${1:-ffmpeg}"

  command -v "${ffmpeg_bin}" >/dev/null 2>&1 || return 1
  "${ffmpeg_bin}" -hide_banner -encoders 2>/dev/null | grep -Eq "[[:space:]]${video_transcode_encoder}[[:space:]]"
}

prefer_system_ffmpeg_if_supported() {
  if [ -x /usr/bin/ffmpeg ] && ffmpeg_supports_video_encoder /usr/bin/ffmpeg; then
    export PATH="/usr/bin:${PATH}"
    hash -r
    return 0
  fi
  return 1
}

install_distro_ffmpeg() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 1
  fi

  echo "INFO: installing FFmpeg for video benchmarks via apt-get" >&2
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends ffmpeg
  hash -r
}

install_build_dependencies() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: apt-get not found and FFmpeg with encoder ${video_transcode_encoder} is unavailable" >&2
    exit 1
  fi

  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
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
}

install_nvcodec_headers() {
  if [ -f /usr/local/include/ffnvcodec/dynlink_loader.h ]; then
    return 0
  fi

  wget -O /tmp/nv-codec-headers.tar.gz \
    "https://github.com/FFmpeg/nv-codec-headers/releases/download/n${NVCODEC_VERSION}/nv-codec-headers-${NVCODEC_VERSION}.tar.gz"
  tar xzf /tmp/nv-codec-headers.tar.gz -C /tmp/
  make -C "/tmp/nv-codec-headers-${NVCODEC_VERSION}"
  make -C "/tmp/nv-codec-headers-${NVCODEC_VERSION}" install
}

build_ffmpeg_with_openh264() {
  echo "INFO: building FFmpeg ${FFMPEG_VERSION} with ${video_transcode_encoder} for video benchmarks" >&2
  echo "INFO: users are responsible for any license obligations from the resulting binaries" >&2

  install_build_dependencies
  install_nvcodec_headers

  cd /tmp
  rm -rf "ffmpeg-${FFMPEG_VERSION}" ffmpeg-snapshot.tar.bz2
  wget -O /tmp/ffmpeg-snapshot.tar.bz2 "https://www.ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2"
  tar xjf /tmp/ffmpeg-snapshot.tar.bz2 -C /tmp/
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
    --enable-libopenh264 \
    --enable-libvpx \
    --enable-cuda \
    --enable-cuvid \
    --enable-nvdec \
    --enable-nvenc \
    --enable-ffnvcodec
  make -j"$(nproc)"
  make install
  ldconfig
  cd /
  rm -rf /tmp/ffmpeg* /tmp/nv-codec-headers*
  hash -r
}

if ffmpeg_supports_video_encoder || prefer_system_ffmpeg_if_supported; then
  exit 0
fi

install_distro_ffmpeg || true
if ffmpeg_supports_video_encoder || prefer_system_ffmpeg_if_supported; then
  exit 0
fi

build_ffmpeg_with_openh264

if ! ffmpeg_supports_video_encoder && ! prefer_system_ffmpeg_if_supported; then
  echo "ERROR: required FFmpeg encoder ${video_transcode_encoder} is unavailable" >&2
  exit 1
fi
