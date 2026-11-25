# syntax=docker/dockerfile:1
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

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_LINK_MODE=copy \
    LANGUAGEBIND_ROOT=/opt/LanguageBind \
    PATH="/opt/venv/bin:/usr/local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libmagic1t64 \
        libpython3.12t64 \
        libsndfile1 \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

RUN git clone --depth 1 https://github.com/PKU-YuanGroup/LanguageBind.git "${LANGUAGEBIND_ROOT}"

WORKDIR /workspace

# Install dependencies from pyproject.toml before copying scripts for better layer caching.
COPY pyproject.toml README.md ./
COPY omnifuse_tutorial ./omnifuse_tutorial
RUN uv sync --no-cache

COPY 0_validate_inputs.py 1_sns.py 1_2_parallel.py 2_embed.py 3_project.py 4_datablend.py utils.py .env.example .rayignore ./
COPY configs ./configs

CMD ["bash"]
