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

import importlib
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking"))

from curator_benchmarking.cli import main
from curator_benchmarking.targets.docker import (
    DockerTarget,
    _containerize_command_args,
    _docker_env_args,
)
from runner.path_resolver import CURATOR_BENCHMARK_PATH_MODE_ENV


def test_run_command_module_import_does_not_load_runtime_dependencies() -> None:
    sys.modules.pop("curator_benchmarking.commands.run", None)
    sys.modules.pop("curator_benchmarking.commands.run_impl", None)
    nemo_curator_loaded_before = "nemo_curator" in sys.modules

    importlib.import_module("curator_benchmarking.commands.run")

    assert "curator_benchmarking.commands.run_impl" not in sys.modules
    assert ("nemo_curator" in sys.modules) is nemo_curator_loaded_before


def test_cli_defaults_to_run_for_legacy_args(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    run_module = types.ModuleType("curator_benchmarking.commands.run")
    run_module.main = lambda argv: calls.append(argv) or 0
    monkeypatch.setitem(sys.modules, "curator_benchmarking.commands.run", run_module)

    assert main(["--config", "config.yaml"]) == 0

    assert calls == [["--config", "config.yaml"]]


def test_cli_list_dispatches_to_run_with_list_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    run_module = types.ModuleType("curator_benchmarking.commands.run")
    run_module.main = lambda argv: calls.append(argv) or 0
    monkeypatch.setitem(sys.modules, "curator_benchmarking.commands.run", run_module)

    assert main(["list", "--config", "config.yaml"]) == 0

    assert calls == [["--list", "--config", "config.yaml"]]


def test_cli_image_target_runs_inside_container(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run_in_target(target: DockerTarget, command_args: list[str]) -> int:
        calls.append((target, command_args))
        return 0

    monkeypatch.setattr("curator_benchmarking.cli.run_in_target", fake_run_in_target)

    assert main(["run", "--image", "curator:test", "--config", "config.yaml"]) == 0

    target, command_args = calls[0]
    assert target.image == "curator:test"
    assert command_args == ["run", "--config", "config.yaml"]


def test_containerize_command_args_rewrites_config_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  - name: results_path
    host_path: {tmp_path}
entries: []
"""
    )

    target = DockerTarget(image="curator:test", benchmark_suite_dir=Path.cwd())

    args = _containerize_command_args(target, ["run", "--config", str(config_path)])

    assert args == ["run", "--config", str(Path(f"/MOUNT/{config_path}"))]


def test_docker_env_args_sets_container_path_mode() -> None:
    env_args = _docker_env_args(DockerTarget(), image_digest="sha256:test")

    assert f"{CURATOR_BENCHMARK_PATH_MODE_ENV}=container" in env_args
