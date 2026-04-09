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

FOLDER="${1:?Usage: $0 <folder> <python-version>}"
PY_VERSION="${2:?Usage: $0 <folder> <python-version>}"

FOLDER="${FOLDER/stages-/stages/}"

rm -rf .venv
uv venv --seed --python "${PY_VERSION}"
uv sync --link-mode copy --locked --extra audio_cpu --extra sdg_cpu --extra text_cpu --extra video_cpu --group test
source .venv/bin/activate
coverage run -a --branch --source=nemo_curator -m pytest -v "tests/$FOLDER" -m "not gpu"
