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

ARG CURATOR_IMAGE=nemo_curator
FROM ${CURATOR_IMAGE} AS nemo_curator_benchmarking

# Add system utilities useful for benchmark and debug
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    less \
    openssh-client \
    vim \
    wget \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Add dependencies for benchmarking to the Curator Python environment
RUN cd /opt/Curator \
 && uv sync --extra all \
 && uv add \
    GitPython \
    oauth2client \
    pydrive2 \
    pynvml \
    pyyaml \
    rich \
  && uv cache prune

# Add the Curator repo to the safe.directory list to avoid GitPython warnings
RUN git config --global --add safe.directory /opt/Curator

 # Set the entrypoint to the main benchmarking runner script
ENTRYPOINT ["python", "/opt/Curator/benchmarking/run.py"]
