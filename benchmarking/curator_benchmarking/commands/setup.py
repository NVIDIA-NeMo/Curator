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

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from curator_benchmarking.paths import benchmark_package_dir


def _package_spec(package_dir: Path, extras: list[str]) -> str:
    extra_suffix = f"[{','.join(extras)}]" if extras else ""
    return f"{package_dir}{extra_suffix}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install benchmark package dependencies in the current environment.")
    parser.add_argument(
        "--benchmark-suite-dir",
        type=Path,
        default=None,
        help="Curator checkout that provides the benchmark package.",
    )
    parser.add_argument(
        "--benchmark-extra",
        action="append",
        dest="benchmark_extras",
        default=None,
        help="Benchmark package extra to install. Can be specified multiple times. Defaults to all.",
    )
    args = parser.parse_args(argv)

    package_dir = benchmark_package_dir(args.benchmark_suite_dir)
    package_spec = _package_spec(package_dir, args.benchmark_extras or ["all"])
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", package_spec]
    else:
        cmd = [sys.executable, "-m", "pip", "install", package_spec]
    return subprocess.call(cmd)  # noqa: S603


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
