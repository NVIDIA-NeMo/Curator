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

set -euo pipefail

: "${GPU_TEST_EXTRAS:?GPU_TEST_EXTRAS is required (e.g. deduplication_cuda12 text_cpu)}"
: "${GPU_TEST_PATHS:?GPU_TEST_PATHS is required (e.g. tests/stages/text)}"

EXTRA_FLAGS=""
for extra in $GPU_TEST_EXTRAS; do
  EXTRA_FLAGS="$EXTRA_FLAGS --extra $extra"
done

uv sync --link-mode copy --locked $EXTRA_FLAGS --group test

export CUSTOM_HF_DATASET=/home/TestData/HF_HOME

CUDA_VISIBLE_DEVICES="0,1" coverage run -a --source=nemo_curator -m pytest -m gpu $GPU_TEST_PATHS
