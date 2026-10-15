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
PYTHON=${PYTHON:-python}
CURATOR_UNDER_TEST_REPO_DIR=${CURATOR_BENCHMARK_CURATOR_REPO_DIR:-/opt/Curator}
FSSPEC_VERSION=2026.3.0

usage() {
    cat <<'EOF'
Usage: install_cv2.sh [--check]

Installs the Curator-under-test cv2 extra. Several image, PDF, and video
workflows require opencv-python-headless, but Curator keeps it out of the
default install because its wheel vendors FFmpeg.

Options:
  --check  Verify cv2 and semdedup's fsspec parquet API without installing.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check)
            CHECK_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

check_cv2_environment() {
    "$PYTHON" - <<'PY'
import sys

try:
    import cv2
except Exception as exc:
    print(f"ERROR: cv2 not importable: {exc}", file=sys.stderr)
    raise SystemExit(1)

try:
    import fsspec
    from fsspec.parquet import open_parquet_files  # noqa: F401
except Exception as exc:
    print(f"ERROR: fsspec parquet API not importable: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"cv2 available: {cv2.__version__}")
print(f"fsspec parquet API available: {fsspec.__version__}")
PY
}

if check_cv2_environment; then
    exit 0
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
    exit 1
fi

if [ ! -d "$CURATOR_UNDER_TEST_REPO_DIR" ]; then
    echo "ERROR: Curator-under-test repo directory not found: $CURATOR_UNDER_TEST_REPO_DIR" >&2
    exit 1
fi

(
    cd "$CURATOR_UNDER_TEST_REPO_DIR" && \
        "$PYTHON" -m pip install --upgrade-strategy only-if-needed ".[cv2]"
)

# Installing Curator's cv2 extra currently lets pip choose an fsspec version
# that breaks semdedup's fsspec.parquet.open_parquet_files import. This is a
# temporary benchmark-environment repair; the real fix is to encode the correct
# fsspec constraint in Curator package metadata.
"$PYTHON" -m pip install --upgrade-strategy only-if-needed "fsspec==${FSSPEC_VERSION}"

check_cv2_environment
