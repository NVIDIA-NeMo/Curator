# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3

RUN sed 's/NVIDIA_PRODUCT_NAME=Triton Server/NVIDIA_PRODUCT_NAME="Triton Server"/' </etc/environment >/etc/environment.new \
    && mv /etc/environment.new /etc/environment

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
    && apt-get install -y \
        nodejs \
        texlive-full

RUN apt-get install -y fonts-noto
RUN wget https://github.com/google/fonts/archive/main.zip \
    && unzip main.zip \
    && mv fonts-main /usr/share/fonts/google_fonts \
    && find /usr/share/fonts '(' -name '*[*' -or -name '*]*' -or -name '*,*' ')' -exec sh -c 'mv "$0" "${0//[\[\],]/_}"' {} ';' \
    && fc-cache -fv \
    && rm main.zip

RUN pip install --no-cache-dir \
        'beautifulsoup4>=4.14.2' \
        'build>=1.2.2.post1' \
        'megatron-energon>=5.0.0' \
        'numpy<2.0.0' \
        'pillow>=11.1.0' \
        'playwright>=1.51.0' \
        'pymupdf>=1.25.1' \
        'scipy>=1.14.1' \
        'sentencepiece>=0.2.0' \
        'sequence-align>=0.3.0' \
        'spacy>=3.7.5' \
        'tree-sitter>=0.23.2' \
        'tritonclient[all]' \
        'webdataset>=1.0.2' \
        'zstandard>=0.25.0' \
    && pip install --no-cache-dir --no-build-isolation 'flash-attn>=2.7.2.post1' \
    && playwright install --with-deps chromium

RUN wget https://github.com/latex-lsp/tree-sitter-latex/archive/master.zip \
    && unzip master.zip \
    && cd tree-sitter-latex-master \
    && wget https://github.com/tree-sitter/tree-sitter/releases/download/v0.24.4/tree-sitter-linux-x64.gz \
    && rm -f tree-sitter-linux-x64 \
    && gunzip tree-sitter-linux-x64.gz \
    && chmod 755 tree-sitter-linux-x64 \
    && ./tree-sitter-linux-x64 generate \
    && rm -rf dist \
    && python3 -m build --wheel \
    && pip install dist/tree_sitter_latex-*.whl \
    && cd .. \
    && rm -rf tree-sitter-latex-master master.zip

WORKDIR /workspace
