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

ACTION=install
PYTHON=${PYTHON:-python}
CURATOR_UNDER_TEST_REPO_DIR=${CURATOR_BENCHMARK_CURATOR_REPO_DIR:-/opt/Curator}
BENCHMARK_SOURCE_MOUNT_DIR=${CURATOR_BENCHMARK_SOURCE_MOUNT_DIR:-/tmp/.curator-benchmark-source}
BENCHMARK_SOURCE_DIR=${CURATOR_BENCHMARK_SOURCE_DIR:-/opt/curator-benchmark-source}
BENCHMARK_SOURCE_SUBDIRS=(benchmarking tutorials)

usage() {
    cat <<'EOF'
Usage: setup_benchmark_env.sh [--check]

Prepares the benchmark runtime environment for Curator benchmarks.

Actions:
  default                 Install benchmark environment dependencies, then check.
  --check                 Verify benchmark Python dependencies and required system tools.

Environment:
  CURATOR_BENCHMARK_CURATOR_REPO_DIR
      Full Curator-under-test source checkout. Container helpers set this to
      /opt/Curator. Bare-metal users should set it when the Curator-under-test
      checkout is not /opt/Curator or when benchmark configs use
      {curator_repo_dir}.
  CURATOR_BENCHMARK_SOURCE_MOUNT_DIR
      Container-only read-only benchmark-source checkout mount. Default:
      /tmp/.curator-benchmark-source. Bare-metal users do not need this.
  CURATOR_BENCHMARK_SOURCE_DIR
      Container-only runtime copy of benchmark source files. Default:
      /opt/curator-benchmark-source. Bare-metal users do not need this.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check)
            ACTION=check
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

case "$ACTION" in
    check|install) ;;
    *)
        echo "ERROR: internal setup action must be one of: check, install" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
script_benchmarking_dir="$(cd "$script_dir/.." && pwd -P)"
script_repo_root="$(cd "$script_benchmarking_dir/.." && pwd -P)"

prepare_benchmark_source() {
    if [ ! -d "$BENCHMARK_SOURCE_MOUNT_DIR" ]; then
        return 0
    fi

    rm -rf "$BENCHMARK_SOURCE_DIR"
    mkdir -p "$BENCHMARK_SOURCE_DIR"
    for subdir in "${BENCHMARK_SOURCE_SUBDIRS[@]}"; do
        if [ ! -e "$BENCHMARK_SOURCE_MOUNT_DIR/$subdir" ]; then
            echo "ERROR: mounted benchmark source is missing $subdir" >&2
            exit 1
        fi
        cp -a "$BENCHMARK_SOURCE_MOUNT_DIR/$subdir" "$BENCHMARK_SOURCE_DIR/"
    done
}

benchmark_source_root() {
    if [ -f "$BENCHMARK_SOURCE_DIR/benchmarking/requirements.txt" ]; then
        echo "$BENCHMARK_SOURCE_DIR"
    elif [ -f "$script_repo_root/benchmarking/requirements.txt" ]; then
        echo "$script_repo_root"
    else
        echo "ERROR: could not find benchmarking/requirements.txt" >&2
        exit 1
    fi
}

check_python_deps() {
    local source_root=$1
    "$PYTHON" - "$source_root/benchmarking/requirements.txt" <<'PY'
import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

requirements = Path(sys.argv[1])


def requirement_name(line):
    line = line.split("#", 1)[0].split(";", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        line = line.split(separator, 1)[0]
    return line.split("[", 1)[0].strip()


status = 0
if importlib.util.find_spec("nemo_curator") is None:
    print("MISSING  nemo_curator")
    status = 1
else:
    print("OK       nemo_curator")

for raw_line in requirements.read_text(encoding="utf-8").splitlines():
    package = requirement_name(raw_line)
    if package is None:
        continue
    try:
        installed_version = version(package)
    except PackageNotFoundError:
        print(f"MISSING  {package}")
        status = 1
    else:
        print(f"OK       {package}: {installed_version}")
raise SystemExit(status)
PY
}

dependency_tools_dir() {
    if [ -d "$CURATOR_UNDER_TEST_REPO_DIR/tools" ]; then
        echo "$CURATOR_UNDER_TEST_REPO_DIR/tools"
        return
    fi

    local fallback
    for fallback in "$BENCHMARK_SOURCE_DIR/tools" "$BENCHMARK_SOURCE_MOUNT_DIR/tools" "$script_repo_root/tools"; do
        if [ -d "$fallback" ]; then
            echo "WARNING: Curator-under-test has no tools directory; using benchmark-source tools from $fallback" >&2
            echo "$fallback"
            return
        fi
    done

    echo "ERROR: could not find dependency setup tools" >&2
    exit 1
}

run_dependency_tool() {
    local script_path=$1
    shift

    if [ ! -f "$script_path" ]; then
        echo "ERROR: dependency setup script not found: $script_path" >&2
        exit 1
    fi
    bash "$script_path" "$@"
}

install_scripts() {
    local tools_dir=$1
    find "$tools_dir" -maxdepth 1 -type f -name 'install_*.sh' | sort
}

check_environment() {
    local source_root=$1
    local tools_dir=$2
    local status=0
    local script

    while IFS= read -r script; do
        run_dependency_tool "$script" --check || status=1
    done < <(install_scripts "$tools_dir")
    check_python_deps "$source_root" || status=1
    return "$status"
}

install_environment() {
    local source_root=$1
    local tools_dir=$2
    local script

    while IFS= read -r script; do
        run_dependency_tool "$script"
    done < <(install_scripts "$tools_dir")

    "$PYTHON" -m pip install --upgrade-strategy only-if-needed \
        -r "$source_root/benchmarking/requirements.txt"
}

prepare_benchmark_source
source_root=$(benchmark_source_root)
tools_dir=$(dependency_tools_dir)

case "$ACTION" in
    check)
        check_environment "$source_root" "$tools_dir"
        ;;
    install)
        install_environment "$source_root" "$tools_dir"
        check_environment "$source_root" "$tools_dir"
        ;;
esac
