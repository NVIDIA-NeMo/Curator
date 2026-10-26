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

CHECK_ONLY=0

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<'EOF'
Usage: install_lynx.sh [--check]

Installs lynx with apt-get when it is not already available. Curator math
HTML extraction uses lynx to convert HTML content to text.

Options:
  --check  Verify lynx is available without installing it.
EOF
    exit 0
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check)
            CHECK_ONLY=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if command -v lynx >/dev/null 2>&1; then
    echo "lynx already available."
    exit 0
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "ERROR: lynx not found on PATH" >&2
    exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: apt-get not found and lynx is unavailable" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends lynx
