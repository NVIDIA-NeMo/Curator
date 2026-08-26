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
import importlib
import importlib.metadata
import importlib.util
import shutil
import sys
from pathlib import Path

from runner.path_resolver import PathResolver
from runner.utils import assert_valid_config_dict, merge_config_files, resolve_env_vars


def _print_result(ok: bool, label: str, detail: str = "") -> bool:
    status = "OK" if ok else "MISSING"
    suffix = f": {detail}" if detail else ""
    print(f"{status:8} {label}{suffix}")
    return ok


def _module_version(module_name: str, package_name: str | None = None) -> str:
    package_name = package_name or module_name
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "unknown"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the benchmark environment.")
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--strict-config-check", action="store_true")
    args = parser.parse_args(argv)

    failures = []
    for module_name, package_name in [
        ("nemo_curator", "nemo-curator"),
        ("curator_benchmarking", "nemo-curator-benchmarking"),
        ("runner", None),
        ("yaml", "pyyaml"),
        ("loguru", None),
    ]:
        if importlib.util.find_spec(module_name) is None:
            failures.append(module_name)
            _print_result(False, module_name)
        else:
            _print_result(True, module_name, _module_version(module_name, package_name))

    for tool_name in ["docker", "uv", "ffmpeg", "ffprobe"]:
        tool_path = shutil.which(tool_name)
        _print_result(tool_path is not None, tool_name, tool_path or "")

    if args.config:
        try:
            config_dict = resolve_env_vars(merge_config_files(args.config), strict=args.strict_config_check)
            assert_valid_config_dict(config_dict)
            path_resolver = PathResolver(config_dict)
            for path_name, path in sorted(path_resolver.path_map.items()):
                exists = path.exists()
                _print_result(exists, f"path:{path_name}", str(path))
        except Exception as exc:
            failures.append("config")
            _print_result(False, "config", str(exc))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
