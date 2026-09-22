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

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: check_ffmpeg.sh [--encoder <name>]... [--decoder <name>]... [--no-ffprobe]

Checks that ffmpeg is available, ffprobe is available by default, and any
requested encoders or decoders are exposed by the active ffmpeg build.

Examples:
  check_ffmpeg.sh
  check_ffmpeg.sh --encoder h264_nvenc --encoder libvpx-vp9
  check_ffmpeg.sh --decoder h264 --decoder hevc --decoder av1
EOF
}

encoders=()
decoders=()
check_ffprobe=1

while [ "$#" -gt 0 ]; do
    case "$1" in
        --encoder)
            encoders+=("${2:?--encoder requires a value}")
            shift 2
            ;;
        --encoder=*)
            encoders+=("${1#*=}")
            shift
            ;;
        --decoder)
            decoders+=("${2:?--decoder requires a value}")
            shift 2
            ;;
        --decoder=*)
            decoders+=("${1#*=}")
            shift
            ;;
        --no-ffprobe)
            check_ffprobe=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not found on PATH" >&2
    exit 1
fi

if [ "$check_ffprobe" -eq 1 ] && ! command -v ffprobe >/dev/null 2>&1; then
    echo "ERROR: ffprobe not found on PATH" >&2
    exit 1
fi

for encoder in "${encoders[@]}"; do
    if ! ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -Fx -- "$encoder" >/dev/null; then
        echo "ERROR: ffmpeg encoder not found: $encoder" >&2
        exit 1
    fi
done

for decoder in "${decoders[@]}"; do
    if ! ffmpeg -hide_banner -decoders 2>/dev/null | awk '{print $2}' | grep -Fx -- "$decoder" >/dev/null; then
        echo "ERROR: ffmpeg decoder not found: $decoder" >&2
        exit 1
    fi
done

echo "FFmpeg dependency check passed."
