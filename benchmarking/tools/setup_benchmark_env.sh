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

MODE=auto
PYTHON=${PYTHON:-python}
CURATOR_UNDER_TEST_REPO_DIR=${CURATOR_BENCHMARK_CURATOR_REPO_DIR:-/opt/Curator}
BENCHMARK_SOURCE_MOUNT_DIR=${CURATOR_BENCHMARK_SOURCE_MOUNT_DIR:-/tmp/.curator-benchmark-source}
BENCHMARK_SOURCE_DIR=${CURATOR_BENCHMARK_SOURCE_DIR:-/opt/curator-benchmark-source}
BENCHMARK_SOURCE_SUBDIRS=(benchmarking tutorials)
CURATOR_EXTRAS=()

usage() {
    cat <<'EOF'
Usage: setup_benchmark_env.sh [--mode check|install|auto] [--curator-extra <extra>]...

Prepares the benchmark runtime environment for Curator benchmarks.

Modes:
  check    Verify benchmark Python dependencies and required system tools.
  install  Install missing benchmark environment dependencies, then check.
  auto     Check first. If running in a container, install missing dependencies;
           otherwise fail with instructions to run install explicitly.

Environment:
  CURATOR_BENCHMARK_PATH_MODE=container
      Optional. Marks the current environment as container-managed for
      --mode auto. Container helpers set this automatically. Bare-metal users
      should usually leave it unset and run --mode check or --mode install
      explicitly.
  CURATOR_BENCHMARK_CURATOR_REPO_DIR
      Full Curator-under-test source checkout. Container helpers set this to
      /opt/Curator. Bare-metal users should set it when the Curator-under-test
      checkout is not /opt/Curator, when installing Curator extras, or when
      benchmark configs use {curator_repo_dir}.
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
        --mode)
            MODE="${2:?--mode requires a value}"
            shift 2
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --curator-extra)
            CURATOR_EXTRAS+=("${2:?--curator-extra requires a value}")
            shift 2
            ;;
        --curator-extra=*)
            CURATOR_EXTRAS+=("${1#*=}")
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

case "$MODE" in
    check|install|auto) ;;
    *)
        echo "ERROR: --mode must be one of: check, install, auto" >&2
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

run_first_available_script() {
    local script_name=$1
    shift
    local candidates=(
        "$CURATOR_UNDER_TEST_REPO_DIR/tools/$script_name"
        "$CURATOR_UNDER_TEST_REPO_DIR/docker/common/$script_name"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -f "$candidate" ]; then
            bash "$candidate" "$@"
            return
        fi
    done
    echo "ERROR: could not find $script_name" >&2
    exit 1
}

run_benchmark_check_script() {
    local script_name=$1
    shift
    local candidates=(
        "$BENCHMARK_SOURCE_DIR/tools/$script_name"
        "$BENCHMARK_SOURCE_MOUNT_DIR/tools/$script_name"
        "$script_repo_root/tools/$script_name"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -f "$candidate" ]; then
            bash "$candidate" "$@"
            return
        fi
    done
    echo "ERROR: could not find $script_name" >&2
    exit 1
}

check_environment() {
    local source_root=$1
    local status=0
    check_python_deps "$source_root" || status=1
    run_benchmark_check_script check_lynx.sh || status=1
    run_benchmark_check_script check_ffmpeg.sh \
        --decoder h264 --decoder hevc --decoder av1 --encoder libopenh264 || status=1
    return "$status"
}

install_environment() {
    local source_root=$1
    "$PYTHON" -m pip install --upgrade-strategy only-if-needed \
        -r "$source_root/benchmarking/requirements.txt"

    local extra
    for extra in "${CURATOR_EXTRAS[@]}"; do
        if [ ! -d "$CURATOR_UNDER_TEST_REPO_DIR" ]; then
            echo "ERROR: Curator-under-test repo directory not found: $CURATOR_UNDER_TEST_REPO_DIR" >&2
            exit 1
        fi
        (
            cd "$CURATOR_UNDER_TEST_REPO_DIR" && \
                "$PYTHON" -m pip install --upgrade-strategy only-if-needed ".[${extra}]"
        )
    done

    if ! run_benchmark_check_script check_lynx.sh; then
        run_first_available_script install_lynx.sh
    fi
    if ! run_benchmark_check_script check_ffmpeg.sh \
        --decoder h264 --decoder hevc --decoder av1 --encoder libopenh264; then
        run_first_available_script install_h264_support.sh --with-libopenh264
    fi
}

prepare_benchmark_source
source_root=$(benchmark_source_root)

case "$MODE" in
    check)
        check_environment "$source_root"
        ;;
    install)
        install_environment "$source_root"
        check_environment "$source_root"
        ;;
    auto)
        if check_environment "$source_root"; then
            exit 0
        fi
        if [ "${CURATOR_BENCHMARK_PATH_MODE:-}" != "container" ]; then
            echo "ERROR: benchmark environment is incomplete." >&2
            echo "Run setup_benchmark_env.sh --mode install to install dependencies explicitly." >&2
            exit 1
        fi
        install_environment "$source_root"
        check_environment "$source_root"
        ;;
esac
