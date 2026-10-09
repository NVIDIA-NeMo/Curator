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

TUTORIAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${TUTORIAL_DIR}/configs/omni_fuse_hybrid.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

"${PYTHON_CMD[@]}" "${TUTORIAL_DIR}/0_validate_inputs.py" --config "${CONFIG}"
"${PYTHON_CMD[@]}" "${TUTORIAL_DIR}/1_sns.py" --config "${CONFIG}"
"${PYTHON_CMD[@]}" "${TUTORIAL_DIR}/2_embed.py" --config "${CONFIG}"
"${PYTHON_CMD[@]}" "${TUTORIAL_DIR}/3_project.py" --config "${CONFIG}"
"${PYTHON_CMD[@]}" "${TUTORIAL_DIR}/4_datablend.py" --config "${CONFIG}"
