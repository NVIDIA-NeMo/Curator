#!/bin/bash
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

# Opt-in installer that adds software h264/hevc/av1 decoder support to a
# Curator container by building FFmpeg from source with those decoders added
# to the strict allowlist.
#
# The release Curator image may omit system ffmpeg entirely. When installed
# with install_ffmpeg.sh, Curator's strict FFmpeg build routes h264/hevc/av1
# decode through NVDEC only. That breaks ffprobe-based metadata extraction in
# CPU-only Ray actors (VideoReader, ClipWriter), which is why this opt-in
# exists.
#
# Run inside a running container, e.g.:
#   docker exec <container> bash /opt/Curator/docker/common/install_h264_support.sh
#
# Behaviour:
#   - Builds FFmpeg from the upstream release tarball, same version as install_ffmpeg.sh,
#     with the existing allowlist plus --enable-decoder=h264,hevc,av1.
#   - Optionally also enables the libopenh264 software h264 ENCODER (Cisco's
#     free-license OpenH264 binary; opt-in via --with-libopenh264).
#   - Replaces /usr/local/bin/{ffmpeg,ffprobe} in place.
#   - Default stays LGPLv3 (only FFmpeg-internal native decoders); with
#     --with-libopenh264 the resulting binary additionally links Cisco's
#     OpenH264 (BSD-2-Clause; see https://www.openh264.org/BINARY_LICENSE.txt).
#   - Takes ~5-10 min.
#
# Keep FFMPEG_VERSION and NVCODEC_VERSION in sync with docker/common/install_ffmpeg.sh.

set -euo pipefail

FFMPEG_VERSION=8.0.1
NVCODEC_VERSION=12.1.14.0
WITH_LIBOPENH264=0

usage() {
    cat <<'EOF'
Usage: install_h264_support.sh [--with-libopenh264] [--FFMPEG_VERSION=<version>]
                               [--NVCODEC_VERSION=<ver>]

Builds FFmpeg from source with software h264/hevc/av1 decoders enabled.

Options:
  --with-libopenh264         Also enable the libopenh264 software h264 ENCODER
                             (Cisco's OpenH264 binary; required by Curator's
                             --transcode-encoder=libopenh264 path).
  --FFMPEG_VERSION=<version> FFmpeg upstream release version (default: 8.0.1).
  --NVCODEC_VERSION=<ver>    nv-codec-headers release version (default: 12.1.14.0).
  -h, --help                 Show this help.

License notice:
  Default mode enables only FFmpeg's internal h264/hevc/av1 decoders (LGPL).
  With --with-libopenh264 the build additionally links Cisco's OpenH264
  binary (BSD-2-Clause + Cisco-distributed binary license; see
  https://www.openh264.org/BINARY_LICENSE.txt). By running this script you
  are responsible for any license obligations the resulting binaries impose
  on your distribution.
EOF
}

for arg in "$@"; do
    case $arg in
        --with-libopenh264)        WITH_LIBOPENH264=1 ;;
        --FFMPEG_VERSION=?*)       FFMPEG_VERSION="${arg#*=}" ;;
        --NVCODEC_VERSION=?*)      NVCODEC_VERSION="${arg#*=}" ;;
        -h|--help)                 usage; exit 0 ;;
        *)                         echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must be run as root inside the container." >&2
    exit 1
fi

echo "==> install_h264_support.sh: rebuilding ffmpeg ${FFMPEG_VERSION}"
echo "    Decoders added: h264, hevc, av1 (software)"
if [ "$WITH_LIBOPENH264" -eq 1 ]; then
    echo "    Encoder added:  libopenh264 (Cisco OpenH264 binary)"
fi
    echo "    NOTE: This expands the container's codec footprint beyond Curator's"
    echo "    default strict FFmpeg allowlist. License obligations of the resulting binaries"
echo "    are the user's responsibility."
echo

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt_packages=(
    autoconf automake build-essential ca-certificates cmake
    libcrypt-dev libnuma-dev libtool libvpx-dev nasm pkg-config
    wget yasm zlib1g-dev
)
if [ "$WITH_LIBOPENH264" -eq 1 ]; then
    apt_packages+=(libopenh264-dev)
fi
apt-get install -y "${apt_packages[@]}"

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

# Configure mirrors install_ffmpeg.sh exactly, with the decoder allowlist
# extended to include software h264/hevc/av1, and optionally the libopenh264
# software h264 encoder.
encoder_list="rawvideo,libvpx_vp9,h264_nvenc,hevc_nvenc,av1_nvenc"
extra_configure_flags=()
if [ "$WITH_LIBOPENH264" -eq 1 ]; then
    encoder_list="${encoder_list},libopenh264"
    extra_configure_flags+=(--enable-libopenh264)
fi

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
    --enable-encoder="${encoder_list}" \
    --enable-decoder=rawvideo,libvpx_vp9,vp9,vp8,h264_cuvid,hevc_cuvid,av1_cuvid,mpeg1video,mpeg2video,mpeg4,h264,hevc,av1 \
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
    --enable-ffnvcodec \
    "${extra_configure_flags[@]}"
make -j"$(nproc)"
make install
ldconfig

cd /
rm -rf /tmp/ffmpeg* /tmp/nv-codec-headers*
echo
if [ "$WITH_LIBOPENH264" -eq 1 ]; then
    echo "==> Done. /usr/local/bin/{ffmpeg,ffprobe} now include software h264/hevc/av1"
    echo "    decoders and the libopenh264 software h264 encoder."
else
    echo "==> Done. /usr/local/bin/{ffmpeg,ffprobe} now include software h264/hevc/av1 decoders."
fi
