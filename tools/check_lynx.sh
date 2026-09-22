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

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<'EOF'
Usage: check_lynx.sh

Checks that the lynx command-line browser is available for Curator math HTML
text extraction.
EOF
    exit 0
fi

if [ "$#" -ne 0 ]; then
    echo "Unknown argument: $1" >&2
    exit 2
fi

if ! command -v lynx >/dev/null 2>&1; then
    echo "ERROR: lynx not found on PATH" >&2
    exit 1
fi

echo "Lynx dependency check passed."
