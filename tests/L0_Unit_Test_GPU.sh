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

GROUP="${1:?Usage: $0 <group> (e.g. dedup, text, image, audio, sdg)}"
CONFIG="tests/gpu_test_groups.json"

EXTRA=$(jq -r --arg g "$GROUP" '.[$g].extra // empty' "$CONFIG")
PATHS=$(jq -r --arg g "$GROUP" '.[$g].paths // [] | .[]' "$CONFIG")

if [ -z "$EXTRA" ] || [ -z "$PATHS" ]; then
  echo "Unknown group: $GROUP"
  echo "Available: $(jq -r 'keys | join(", ")' "$CONFIG")"
  exit 1
fi

uv sync --link-mode copy --locked --extra "$EXTRA" --group test

CUDA_VISIBLE_DEVICES="0,1" coverage run -a --source=nemo_curator -m pytest -m gpu $PATHS
