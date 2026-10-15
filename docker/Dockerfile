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

ARG CUDA_VER=12.9.1
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
    software-properties-common \
    curl \
    git \
    lynx \
    libmagic1t64 \
    vim && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python-is-python3 && \
    apt-get install -y --only-upgrade gnupg libunbound8 openssl libssl3t64 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Replace any bundled Nsight Systems with the pinned, patched devtools release.
# Keep the repository key scoped to this source.
RUN rm -rf /usr/local/cuda*/NsightSystems-cli-* /opt/nvidia/nsight-systems* && \
    installed_nsys=$(dpkg-query -W -f='${binary:Package} ${db:Status-Status}\n' \
        | awk '$1 ~ /^nsight-systems/ && $2 == "installed" {print $1}') && \
    if [ -n "${installed_nsys}" ]; then \
        apt-get remove --purge -y --allow-change-held-packages ${installed_nsys}; \
    fi && \
    NSYS_VERSION="2026.5.1.161-265138896106v0" && \
    NSYS_RELEASE=$(echo "${NSYS_VERSION}" | cut -d. -f1-3) && \
    NSYS_REPO="https://developer.download.nvidia.com/devtools/repos/ubuntu$(. /etc/lsb-release; echo "${DISTRIB_RELEASE}" | tr -d .)/$(dpkg --print-architecture)" && \
    curl -fsSL "${NSYS_REPO}/nvidia.pub" -o /usr/share/keyrings/nvidia-devtools.asc && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.asc] ${NSYS_REPO}/ /" \
        > /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends "nsight-systems-cli-${NSYS_RELEASE}=${NSYS_VERSION}" && \
    nsys --version && \
    test "$(dpkg-query -W -f='${Version}' "nsight-systems-cli-${NSYS_RELEASE}")" = "${NSYS_VERSION}" && \
    rm -f /etc/apt/sources.list.d/nvidia-devtools.list /usr/share/keyrings/nvidia-devtools.asc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Install uv
ENV UV_VERSION="0.12.1"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_CACHE_DIR=/opt/uv_cache
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
ENV UV_LINK_MODE=copy
RUN uv venv ${UV_PROJECT_ENVIRONMENT} --python /usr/bin/python3.13 --system-site-packages --seed

FROM build AS nemo_curator_dep

# Install etcd + nats-server (required by the Dynamo inference backend)
COPY docker/common/install_etcd_nats.sh .
RUN bash install_etcd_nats.sh && \
    rm install_etcd_nats.sh

# Include software codecs and WAV resampling for examples and benchmarks.
COPY docker/common/install_h264_support.sh .
RUN bash install_h264_support.sh --with-libopenh264 && \
    rm install_h264_support.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM nemo_curator_dep AS nemo_curator

ARG CURATOR_EXTRA=all
ARG CURATOR_GROUPS="--all-groups"
WORKDIR /opt/Curator

# Selective file copying before dependency installation for better layer caching
COPY pyproject.toml uv.lock /opt/Curator/
COPY nemo_curator/__init__.py nemo_curator/package_info.py /opt/Curator/nemo_curator/

# Install Curator
RUN if [ -n "${CURATOR_EXTRA}" ]; then \
        uv sync --link-mode copy --locked --extra ${CURATOR_EXTRA} --extra cv2 ${CURATOR_GROUPS} --no-cache; \
    else \
        uv sync --link-mode copy --locked --extra cv2 ${CURATOR_GROUPS} --no-cache; \
    fi && \
    # Curator uses Python Ray workers only; remove Java/cross-language support.
    # Remove the directory, not just JAR files, so Ray disables Java workers.
    find /opt/venv -type d -path "*/site-packages/ray/jars" -prune -exec rm -rf {} + && \
    find /opt/venv -type d -path "*ray/_private/runtime_env/agent/thirdparty_files/aiohttp*" -exec rm -rf {} + && \
    # Fail the build if OpenCV cannot load its installed Python or native dependencies.
    python -c "import cv2"

# Fail the build if required codecs are missing or WAV resampling/probing fails.
# GPU codecs are checked for availability only; execution needs runtime hardware.
RUN python - <<'PY'
import json
import subprocess
import tempfile
import wave
from pathlib import Path

for option, required in (
    ("-encoders", {"libopenh264", "libvpx-vp9", "pcm_s16le", "h264_nvenc", "hevc_nvenc", "av1_nvenc"}),
    ("-decoders", {"h264", "hevc", "av1", "pcm_s16le"}),
):
    output = subprocess.check_output(["ffmpeg", "-hide_banner", option], text=True)
    available = {line.split()[1] for line in output.splitlines() if len(line.split()) >= 2}
    assert required <= available, f"Missing FFmpeg capabilities: {required - available}"

with tempfile.TemporaryDirectory() as directory:
    source, destination = (Path(directory) / name for name in ("input.wav", "output.wav"))
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 800)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", str(destination)], check=True)
    output = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels", "-of", "json", str(destination)], text=True)
    stream = json.loads(output)["streams"][0]
    assert (stream["codec_name"], stream["sample_rate"], stream["channels"]) == ("pcm_s16le", "16000", 1)
PY

# Install nemotron-ocr (C++/CUDA extension — requires no-build-isolation)
# GIT_LFS_SKIP_SMUDGE=1: clone source only, model weights are mounted at runtime
# TORCH_CUDA_ARCH_LIST: compile for SM 8.0/8.6/8.9/9.0 to cover A100, A10, RTX Ada, H100
ARG NEMOTRON_OCR_REV=86cacb0467fa4f7ce54342fdb250825e0d928ae7
RUN uv pip install hatchling ninja editables && \
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/nvidia/nemotron-ocr-v2 /tmp/nemotron-ocr-src && \
    git -C /tmp/nemotron-ocr-src checkout "${NEMOTRON_OCR_REV}" && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" uv pip install --no-build-isolation /tmp/nemotron-ocr-src/nemotron-ocr && \
    rm -rf /tmp/nemotron-ocr-src

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
