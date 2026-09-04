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

from curator_benchmarking.config import load_benchmark_config
from curator_benchmarking.dependencies import (
    dependency_groups_from_config,
    python_extras_for_dependency_groups,
    validate_dependency_groups,
)
from curator_benchmarking.paths import benchmark_package_dir
from curator_benchmarking.system_tools import install_system_dependencies


def _package_spec(package_dir: Path, extras: list[str]) -> str:
    extra_suffix = f"[{','.join(extras)}]" if extras else ""
    return f"{package_dir}{extra_suffix}"


def _comma_separated_values(values: list[str] | None) -> list[str]:
    result = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install benchmark package dependencies in the current environment.")
    parser.add_argument(
        "--benchmark-suite-dir",
        type=Path,
        default=None,
        help="Benchmarking package directory. A Curator checkout root is also accepted.",
    )
    parser.add_argument(
        "--entry-name",
        action="append",
        dest="entry_names",
        default=None,
        help=(
            "Benchmark entry name to select from --config when resolving "
            "dependency groups. Can be specified multiple times or as a "
            "comma-separated list."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="Benchmark YAML config used to resolve entry dependency groups.",
    )
    parser.add_argument(
        "--install-system-tools",
        action="store_true",
        help=(
            "Also run matching non-Python system dependency install scripts. "
            "This is never enabled by default for the current environment."
        ),
    )
    args = parser.parse_args(argv)

    package_dir = benchmark_package_dir(args.benchmark_suite_dir)
    dependency_groups = _dependency_groups_from_args(args)
    validate_dependency_groups(dependency_groups, suite_dir=package_dir)

    python_extras = list(python_extras_for_dependency_groups(dependency_groups, suite_dir=package_dir))
    package_spec = _package_spec(package_dir, python_extras)
    if shutil.which("uv"):
        cmd = ["uv", "--no-config", "pip", "install", package_spec]
    else:
        cmd = [sys.executable, "-m", "pip", "install", package_spec]
    status = subprocess.call(cmd)  # noqa: S603
    if status != 0 or not args.install_system_tools:
        return status

    return install_system_dependencies(
        dependency_groups,
        suite_dir=package_dir,
    )


def _dependency_groups_from_args(args: argparse.Namespace) -> list[str]:
    if args.config:
        config = load_benchmark_config(args.config, drop_disabled=True)
        entry_names = _comma_separated_values(args.entry_names)
        return list(
            dependency_groups_from_config(
                config,
                entry_names=entry_names if entry_names else None,
            )
        )
    if args.entry_names:
        msg = "--entry-name requires --config so dependency groups can be resolved from YAML"
        raise ValueError(msg)
    return ["all"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
