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

"""Helpers for resolving benchmark dependency groups.

Benchmark entries declare logical dependency group names in YAML. A group can
be backed by a Python optional dependency in ``benchmarking/pyproject.toml``,
a system dependency script directory under ``benchmarking/system_deps/``, or
both. The Python code here knows that convention, but it does not know what any
particular group installs.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from curator_benchmarking.paths import benchmark_package_dir
from curator_benchmarking.system_tools import system_dependency_names

if TYPE_CHECKING:
    from pathlib import Path

_DEPENDENCY_GROUP_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def known_extras() -> tuple[str, ...]:
    """Return benchmark package extras from the default benchmark suite."""
    return pyproject_optional_dependency_names()


def pyproject_optional_dependency_names(suite_dir: str | Path | None = None) -> tuple[str, ...]:
    """Return optional dependency group names from benchmark ``pyproject.toml``."""
    optional_dependencies = _pyproject_data(suite_dir).get("project", {}).get("optional-dependencies", {})
    return tuple(sorted(optional_dependencies))


def pyproject_dependency_requirements(suite_dir: str | Path | None = None) -> tuple[str, ...]:
    """Return required dependency strings from benchmark ``pyproject.toml``."""
    dependencies = _pyproject_data(suite_dir).get("project", {}).get("dependencies", [])
    return tuple(dependencies)


def pyproject_optional_dependency_requirements(
    dependency_groups: Sequence[str],
    *,
    suite_dir: str | Path | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return selected optional dependency requirement strings grouped by extra."""
    requested_extras = python_extras_for_dependency_groups(dependency_groups, suite_dir=suite_dir)
    optional_dependencies = _pyproject_data(suite_dir).get("project", {}).get("optional-dependencies", {})
    return tuple((extra, tuple(optional_dependencies.get(extra, []))) for extra in requested_extras)


def dependency_groups_from_config(
    config: Mapping[str, Any],
    *,
    entry_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return dependency groups declared by a benchmark config.

    Top-level ``dependencies`` apply to the whole run. Enabled sink configs,
    data setup entries, and benchmark entries can also declare dependencies.
    When ``entry_names`` is provided, only matching benchmark entries are used;
    sinks and data setups are still included because they are run-scoped.
    """
    groups = set(_dependency_groups_from_mapping(config, "top-level config"))
    groups.update(_dependency_groups_from_named_items(config.get("sinks", []), "sinks"))
    groups.update(_dependency_groups_from_named_items(config.get("data_setups", []), "data_setups"))
    selected_entry_names = set(entry_names) if entry_names is not None else None
    groups.update(
        _dependency_groups_from_named_items(
            config.get("entries", []),
            "entries",
            selected_names=selected_entry_names,
        )
    )
    if selected_entry_names is not None:
        missing = selected_entry_names - _enabled_item_names(config.get("entries", []), "entries")
        if missing:
            msg = f"benchmark config does not contain enabled entry name(s): {sorted(missing)}"
            raise ValueError(msg)
    return tuple(sorted(groups))


def python_extras_for_dependency_groups(
    dependency_groups: Sequence[str],
    *,
    suite_dir: str | Path | None = None,
) -> tuple[str, ...]:
    """Return dependency groups that correspond to benchmark package extras."""
    requested = _normalize_dependency_groups(dependency_groups)
    if "all" in requested:
        return ("all",)

    available_extras = set(pyproject_optional_dependency_names(suite_dir))
    return tuple(sorted(group for group in requested if group in available_extras))


def system_dependency_groups_for_dependency_groups(
    dependency_groups: Sequence[str],
    *,
    suite_dir: str | Path | None = None,
) -> tuple[str, ...]:
    """Return dependency groups that have system dependency scripts."""
    requested = _normalize_dependency_groups(dependency_groups)
    available_system_deps = set(system_dependency_names(suite_dir))
    if "all" in requested:
        return tuple(sorted(available_system_deps))
    return tuple(sorted(group for group in requested if group in available_system_deps))


def validate_dependency_groups(
    dependency_groups: Sequence[str],
    *,
    suite_dir: str | Path | None = None,
) -> None:
    """Raise if any dependency group is not backed by an extra or script dir."""
    requested = _normalize_dependency_groups(dependency_groups)
    if "all" in requested:
        return

    known_groups = set(pyproject_optional_dependency_names(suite_dir))
    known_groups.update(system_dependency_names(suite_dir))
    unknown = requested - known_groups
    if unknown:
        msg = f"unknown benchmark dependency group(s): {sorted(unknown)}"
        raise ValueError(msg)


def _normalize_dependency_groups(dependency_groups: Sequence[str]) -> set[str]:
    normalized = {group.strip() for group in dependency_groups if group.strip()}
    invalid = sorted(group for group in normalized if not _DEPENDENCY_GROUP_PATTERN.match(group))
    if invalid:
        msg = f"invalid benchmark dependency group name(s): {invalid}"
        raise ValueError(msg)
    return normalized


def _pyproject_data(suite_dir: str | Path | None = None) -> Mapping[str, Any]:
    pyproject_path = benchmark_package_dir(suite_dir) / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def _dependency_groups_from_named_items(
    items: object,
    field_name: str,
    *,
    selected_names: set[str] | None = None,
) -> set[str]:
    if items is None:
        return set()
    if not isinstance(items, list):
        msg = f"benchmark config field '{field_name}' must be a list"
        raise TypeError(msg)

    groups = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            msg = f"benchmark config field '{field_name}' item at index {index} must be a mapping"
            raise TypeError(msg)
        if item.get("enabled", True) is False:
            continue
        name = item.get("name")
        if selected_names is not None and name not in selected_names:
            continue
        groups.update(_dependency_groups_from_mapping(item, f"{field_name}.{name or index}"))
    return groups


def _enabled_item_names(items: object, field_name: str) -> set[str]:
    if items is None:
        return set()
    if not isinstance(items, list):
        msg = f"benchmark config field '{field_name}' must be a list"
        raise TypeError(msg)
    names = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            msg = f"benchmark config field '{field_name}' item at index {index} must be a mapping"
            raise TypeError(msg)
        if item.get("enabled", True) is False:
            continue
        name = item.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _dependency_groups_from_mapping(item: Mapping[str, Any], context: str) -> set[str]:
    dependencies = item.get("dependencies", [])
    if dependencies is None:
        return set()
    if not isinstance(dependencies, list):
        msg = f"benchmark config field '{context}.dependencies' must be a list"
        raise TypeError(msg)
    invalid = [dependency for dependency in dependencies if not isinstance(dependency, str)]
    if invalid:
        msg = f"benchmark config field '{context}.dependencies' must contain only strings"
        raise ValueError(msg)
    return _normalize_dependency_groups(dependencies)
