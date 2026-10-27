#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# Audio Python wheels load FFmpeg shared libraries, even for CPU decoding.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON=${PYTHON:-python}
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    echo "Usage: install_audio.sh [--check]"
    echo "Install Curator audio dependencies, or verify imports and decoding without installing."
    exit 0
fi
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != --check ]; }; then
    echo "Usage: install_audio.sh [--check]" >&2
    exit 2
fi
if "$PYTHON" "$script_dir/_install_audio.py" --check; then
    exit 0
fi
if [ "${1:-}" = --check ]; then
    exit 1
fi
bash "$script_dir/install_ffmpeg.sh"
bash "$script_dir/install_openfst.sh"
exec "$PYTHON" "$script_dir/_install_audio.py"
