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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

import pytest
from runner.path_resolver import (
    CURATOR_BENCHMARK_PATH_MODE_ENV,
    DEFAULT_CONTAINER_PATH_PREFIX,
    PathResolver,
    set_path_mode,
)


def test_unmap_container_path_returns_host_visible_path() -> None:
    path_resolver = PathResolver(
        {
            "paths": [
                {
                    "name": "results_path",
                    "host_path": "/host/results",
                    "container_path": "/container/results",
                }
            ]
        }
    )

    assert path_resolver.unmap_container_path(Path("/container/results/run")) == Path("/host/results/run")


def test_unmap_container_path_returns_unmapped_path_unchanged() -> None:
    path_resolver = PathResolver(
        {
            "paths": [
                {
                    "name": "results_path",
                    "host_path": "/host/results",
                    "container_path": "/container/results",
                }
            ]
        }
    )

    assert path_resolver.unmap_container_path(Path("/other/results/run")) == Path("/other/results/run")


def test_path_resolver_uses_host_paths_when_path_mode_is_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CURATOR_BENCHMARK_PATH_MODE_ENV, "host")

    path_resolver = PathResolver(
        {
            "paths": [
                {
                    "name": "datasets_path",
                    "host_path": "/host/datasets",
                    "container_path": "/container/datasets",
                }
            ]
        }
    )

    assert path_resolver.resolve("datasets_path") == Path("/host/datasets")


def test_path_resolver_uses_container_paths_when_path_mode_is_container(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CURATOR_BENCHMARK_PATH_MODE_ENV, "container")

    path_resolver = PathResolver(
        {
            "paths": [
                {
                    "name": "datasets_path",
                    "host_path": "/host/datasets",
                    "container_path": "/container/datasets",
                }
            ]
        }
    )

    assert path_resolver.resolve("datasets_path") == Path("/container/datasets")


def test_path_resolver_defaults_container_path_under_mount_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CURATOR_BENCHMARK_PATH_MODE_ENV, "container")

    path_resolver = PathResolver({"paths": [{"name": "datasets_path", "host_path": "/host/datasets"}]})

    assert path_resolver.resolve("datasets_path") == Path(f"{DEFAULT_CONTAINER_PATH_PREFIX}/host/datasets")


def test_path_resolver_rejects_invalid_path_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CURATOR_BENCHMARK_PATH_MODE_ENV, "invalid")

    with pytest.raises(ValueError, match=CURATOR_BENCHMARK_PATH_MODE_ENV):
        PathResolver({"paths": [{"name": "datasets_path", "host_path": "/host/datasets"}]})


def test_set_path_mode_updates_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CURATOR_BENCHMARK_PATH_MODE_ENV, raising=False)

    set_path_mode("container")

    assert os.environ[CURATOR_BENCHMARK_PATH_MODE_ENV] == "container"
