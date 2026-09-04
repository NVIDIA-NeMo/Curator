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

import os
from pathlib import Path

from runner.path_resolver import PathResolver
from runner.utils import assert_valid_config_dict, merge_config_files, resolve_env_vars

BENCHMARK_SUITE_DIR_ENV = "CURATOR_BENCHMARK_SUITE_DIR"


def resolve_benchmark_suite_dir(path: str | Path | None = None) -> Path:
    """Return the benchmark suite directory that contains ``pyproject.toml``.

    Callers may pass either the ``benchmarking/`` directory or a Curator
    checkout root that contains it. The returned path is always the
    self-contained benchmark suite/package directory.
    """
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    elif env_path := os.environ.get(BENCHMARK_SUITE_DIR_ENV):
        candidates.append(Path(env_path))

    this_file = Path(__file__).resolve()
    candidates.extend(
        [
            this_file.parents[1],
            Path.cwd(),
            Path.cwd() / "benchmarking",
            Path.cwd().parent,
            Path.cwd().parent / "benchmarking",
        ]
    )

    for candidate in candidates:
        if suite_dir := _benchmark_suite_dir_from_candidate(candidate):
            return suite_dir

    if path is not None:
        msg = f"Path does not contain the benchmark suite package: {path}"
        raise ValueError(msg)

    msg = "Could not infer the benchmark suite directory. Run from a Curator checkout or pass --benchmark-suite-dir."
    raise ValueError(msg)


def benchmark_package_dir(suite_dir: str | Path | None = None) -> Path:
    """Return the benchmark package directory."""
    return resolve_benchmark_suite_dir(suite_dir)


def _benchmark_suite_dir_from_candidate(candidate: str | Path) -> Path | None:
    expanded = Path(candidate).expanduser().resolve()
    if (expanded / "pyproject.toml").exists() and (expanded / "curator_benchmarking").is_dir():
        return expanded
    benchmark_dir = expanded / "benchmarking"
    if (benchmark_dir / "pyproject.toml").exists() and (benchmark_dir / "curator_benchmarking").is_dir():
        return benchmark_dir
    return None


def volume_mount_pairs_from_config(
    config: dict,
    *,
    resolve_environment: bool = True,
    strict_environment: bool = False,
    validate: bool = True,
) -> list[tuple[Path, Path]]:
    """Return host/container volume mounts required by a benchmark config."""
    if resolve_environment:
        config = resolve_env_vars(config, strict=strict_environment)
    if validate:
        assert_valid_config_dict(config)

    path_resolver = PathResolver(config)
    pairs = []
    for host_path, container_path in path_resolver.volume_mount_pairs():
        if not host_path.is_absolute():
            msg = f"Configured host path must be absolute: {host_path}"
            raise ValueError(msg)
        pairs.append((host_path, container_path))
    return pairs


def volume_mount_pairs_from_configs(
    config_files: list[str | Path],
    *,
    resolve_environment: bool = True,
    strict_environment: bool = False,
    validate: bool = True,
) -> list[tuple[Path, Path]]:
    """Return host/container volume mounts required by merged config files."""
    if not config_files:
        return []
    config = merge_config_files([Path(config_file) for config_file in config_files])
    return volume_mount_pairs_from_config(
        config,
        resolve_environment=resolve_environment,
        strict_environment=strict_environment,
        validate=validate,
    )
