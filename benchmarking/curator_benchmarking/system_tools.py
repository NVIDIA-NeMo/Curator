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

"""Generic system dependency script discovery and execution.

System dependencies are declared as dependency group names in YAML. If
``benchmarking/system_deps/<group>/check.sh`` or ``install.sh`` exists, the
benchmarking tools can run it without knowing what the group means.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from curator_benchmarking.paths import benchmark_package_dir

if TYPE_CHECKING:
    from collections.abc import Sequence

SYSTEM_DEPS_DIR_NAME = "system_deps"
CHECK_SCRIPT = "check.sh"
INSTALL_SCRIPT = "install.sh"
BASH = "/bin/bash"


@dataclass(frozen=True)
class SystemDependencyCheckResult:
    """Result from running one system dependency check script."""

    dependency: str
    script_path: Path
    returncode: int
    output: str

    @property
    def ok(self) -> bool:
        """Return whether the check script reported success."""
        return self.returncode == 0


def system_dependency_names(suite_dir: str | Path | None = None) -> tuple[str, ...]:
    """Return dependency groups with a system dependency script directory."""
    deps_dir = system_dependencies_dir(suite_dir)
    if not deps_dir.exists():
        return ()
    return tuple(sorted(path.name for path in deps_dir.iterdir() if path.is_dir()))


def system_dependency_install_command(
    dependency_groups: Sequence[str],
    *,
    suite_dir: str | Path | None,
    source_suite_dir: str | Path | None = None,
) -> str:
    """Return a shell command that runs install scripts for dependency groups."""
    discovery_suite_dir = source_suite_dir if source_suite_dir is not None else suite_dir
    commands = [
        _script_command(
            _command_script_path(
                script_path,
                discovery_suite_dir=discovery_suite_dir,
                command_suite_dir=suite_dir,
            )
        )
        for script_path in _script_paths(dependency_groups, suite_dir=discovery_suite_dir, script_name=INSTALL_SCRIPT)
    ]
    return " && ".join(commands)


def install_system_dependencies(
    dependency_groups: Sequence[str],
    *,
    suite_dir: str | Path | None,
) -> int:
    """Run system dependency install scripts in the current environment."""
    command = system_dependency_install_command(
        dependency_groups,
        suite_dir=suite_dir,
    )
    if not command:
        return 0
    return subprocess.call([BASH, "-lc", command])  # noqa: S603


def check_system_dependencies(
    dependency_groups: Sequence[str],
    *,
    suite_dir: str | Path | None,
) -> tuple[SystemDependencyCheckResult, ...]:
    """Run system dependency check scripts in the current environment."""
    results = []
    for script_path in _script_paths(dependency_groups, suite_dir=suite_dir, script_name=CHECK_SCRIPT):
        completed = subprocess.run(  # noqa: S603
            [BASH, str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        results.append(
            SystemDependencyCheckResult(
                dependency=script_path.parent.name,
                script_path=script_path,
                returncode=completed.returncode,
                output=output,
            )
        )
    return tuple(results)


def system_dependencies_dir(suite_dir: str | Path | None = None) -> Path:
    """Return the source directory that contains system dependency scripts."""
    return benchmark_package_dir(suite_dir) / SYSTEM_DEPS_DIR_NAME


def _script_paths(
    dependency_groups: Sequence[str],
    *,
    suite_dir: str | Path | None,
    script_name: str,
) -> tuple[Path, ...]:
    deps_dir = system_dependencies_dir(suite_dir)
    script_paths = []
    for dependency_group in _expanded_dependency_groups(dependency_groups, suite_dir=suite_dir):
        script_path = deps_dir / dependency_group / script_name
        if script_path.exists():
            script_paths.append(script_path)
    return tuple(script_paths)


def _expanded_dependency_groups(
    dependency_groups: Sequence[str],
    *,
    suite_dir: str | Path | None,
) -> tuple[str, ...]:
    requested = {dependency_group.strip() for dependency_group in dependency_groups if dependency_group.strip()}
    if "all" in requested:
        return system_dependency_names(suite_dir)
    return tuple(sorted(requested))


def _script_command(script_path: Path) -> str:
    return " ".join(["bash", shlex.quote(str(script_path))])


def _command_script_path(
    script_path: Path,
    *,
    discovery_suite_dir: str | Path | None,
    command_suite_dir: str | Path | None,
) -> Path:
    if command_suite_dir is None or discovery_suite_dir is None:
        return script_path
    discovery_package_dir = benchmark_package_dir(discovery_suite_dir)
    command_package_dir = Path(command_suite_dir)
    return command_package_dir / script_path.relative_to(discovery_package_dir)
