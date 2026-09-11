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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from curator_benchmarking.config import load_benchmark_config
from curator_benchmarking.dependencies import (
    dependency_groups_from_config,
    pyproject_dependency_requirements,
    pyproject_optional_dependency_requirements,
    validate_dependency_groups,
)
from curator_benchmarking.paths import resolve_benchmark_suite_dir
from curator_benchmarking.system_tools import check_system_dependencies
from runner.path_resolver import PathResolver, set_path_mode
from runner.utils import assert_valid_config_dict, merge_config_files, resolve_env_vars

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class PythonDependencyCheckResult:
    """Result from checking one Python package requirement."""

    ok: bool
    label: str
    detail: str
    failure_status: str = "MISSING"


def _print_result(ok: bool, label: str, detail: str = "", *, failure_status: str = "MISSING") -> bool:
    status = "OK" if ok else failure_status
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


def _check_python_dependency_requirements(
    requirements: Sequence[tuple[str, str]],
) -> tuple[PythonDependencyCheckResult, ...]:
    try:
        # Keep this import local so minimal host installs can launch Docker
        # targets without installing check-only requirement parsing support.
        from packaging.requirements import InvalidRequirement, Requirement
    except ModuleNotFoundError:
        return (
            PythonDependencyCheckResult(
                ok=False,
                label="python-dep:packaging",
                detail="required to evaluate pyproject.toml requirements",
            ),
        )

    results = []
    for group_name, requirement_string in requirements:
        try:
            requirement = Requirement(requirement_string)
        except InvalidRequirement as exc:
            results.append(
                PythonDependencyCheckResult(
                    ok=False,
                    label=f"python-dep:{group_name}:{requirement_string}",
                    detail=f"invalid requirement: {exc}",
                    failure_status="FAILED",
                )
            )
            continue

        if requirement.marker is not None and not requirement.marker.evaluate():
            continue

        label = _python_dependency_label(group_name, requirement.name)
        try:
            installed_version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            results.append(PythonDependencyCheckResult(ok=False, label=label, detail=requirement_string))
            continue

        if requirement.specifier and not requirement.specifier.contains(installed_version, prereleases=True):
            results.append(
                PythonDependencyCheckResult(
                    ok=False,
                    label=label,
                    detail=f"installed {installed_version} does not satisfy {requirement.specifier}",
                    failure_status="FAILED",
                )
            )
            continue

        results.append(PythonDependencyCheckResult(ok=True, label=label, detail=installed_version))
    return tuple(results)


def _python_dependency_label(group_name: str, package_name: str) -> str:
    if group_name == "core":
        return f"python-dep:{package_name}"
    return f"python-dep:{group_name}:{package_name}"


def _python_dependency_requirements(
    dependency_groups: Sequence[str],
    *,
    suite_dir: str | Path | None,
) -> tuple[tuple[str, str], ...]:
    requirements = [("core", requirement) for requirement in pyproject_dependency_requirements(suite_dir)]
    for extra, extra_requirements in pyproject_optional_dependency_requirements(
        dependency_groups,
        suite_dir=suite_dir,
    ):
        requirements.extend((extra, requirement) for requirement in extra_requirements)
    return tuple(requirements)


def _comma_separated_values(values: list[str] | None) -> list[str]:
    result = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the benchmark environment.")
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--strict-config-check", action="store_true")
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
        "--path-mode",
        choices=["auto", "host", "container"],
        default=None,
        help=(
            "Select whether configured paths resolve to host_path or container_path. "
            "Defaults to CURATOR_BENCHMARK_PATH_MODE, then auto."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.path_mode:
        set_path_mode(args.path_mode)

    failures = []
    _check_runtime_modules(failures)
    _check_tools()
    _check_config_paths(args, failures)
    dependency_groups = _resolve_dependency_groups(args, failures)
    _check_python_dependencies(args, dependency_groups, failures)
    _check_system_dependency_scripts(args, dependency_groups, failures)
    return 1 if failures else 0


def _check_runtime_modules(failures: list[str]) -> None:
    for module_name, package_name in [
        ("nemo_curator", "nemo-curator"),
        ("curator_benchmarking", "nemo-curator-benchmarking"),
        ("runner", None),
    ]:
        if importlib.util.find_spec(module_name) is None:
            failures.append(module_name)
            _print_result(False, module_name)
        else:
            _print_result(True, module_name, _module_version(module_name, package_name))


def _check_tools() -> None:
    for tool_name in ["docker", "uv"]:
        tool_path = shutil.which(tool_name)
        _print_result(tool_path is not None, tool_name, tool_path or "")


def _check_config_paths(args: argparse.Namespace, failures: list[str]) -> None:
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


def _resolve_dependency_groups(args: argparse.Namespace, failures: list[str]) -> Sequence[str]:
    if not args.config and not args.entry_names:
        return []
    if args.entry_names and not args.config:
        failures.append("dependencies")
        _print_result(
            False,
            "dependencies",
            "--entry-name requires --config so dependency groups can be resolved from YAML",
        )
        return []

    try:
        suite_dir = resolve_benchmark_suite_dir(args.benchmark_suite_dir)
        dependency_config = load_benchmark_config(args.config, drop_disabled=True)
        entry_names = _comma_separated_values(args.entry_names)
        dependency_groups = dependency_groups_from_config(
            dependency_config,
            entry_names=entry_names if entry_names else None,
        )
        validate_dependency_groups(dependency_groups, suite_dir=suite_dir)
    except Exception as exc:
        failures.append("dependencies")
        _print_result(False, "dependencies", str(exc))
    else:
        return dependency_groups
    return []


def _check_python_dependencies(
    args: argparse.Namespace,
    dependency_groups: Sequence[str],
    failures: list[str],
) -> None:
    try:
        suite_dir = resolve_benchmark_suite_dir(args.benchmark_suite_dir)
        for result in _check_python_dependency_requirements(
            _python_dependency_requirements(dependency_groups, suite_dir=suite_dir)
        ):
            _print_result(result.ok, result.label, result.detail, failure_status=result.failure_status)
            if not result.ok:
                failures.append(result.label)
    except Exception as exc:
        failures.append("python-dependencies")
        _print_result(False, "python-dependencies", str(exc))


def _check_system_dependency_scripts(
    args: argparse.Namespace,
    dependency_groups: Sequence[str],
    failures: list[str],
) -> None:
    if args.config or args.entry_names:
        try:
            suite_dir = resolve_benchmark_suite_dir(args.benchmark_suite_dir)
            validate_dependency_groups(dependency_groups, suite_dir=suite_dir)
            for result in check_system_dependencies(dependency_groups, suite_dir=suite_dir):
                _print_result(result.ok, f"system-dep:{result.dependency}", result.output)
                if not result.ok:
                    failures.append(f"system-dep:{result.dependency}")
        except Exception as exc:
            failures.append("dependencies")
            _print_result(False, "dependencies", str(exc))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
