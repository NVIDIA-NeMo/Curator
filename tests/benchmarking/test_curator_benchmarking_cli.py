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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking"))

from curator_benchmarking.cli import main
from curator_benchmarking.targets.docker import (
    DockerTarget,
    _containerize_command_args,
    _docker_env_args,
    _setup_command,
    run_start,
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


def test_cli_start_dispatches_to_docker_target(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run_start(target: DockerTarget, command_args: list[str]) -> int:
        calls.append((target, command_args))
        return 0

    monkeypatch.setattr("curator_benchmarking.cli.run_start", fake_run_start)

    assert main(["start", "--image", "curator:test", "--name", "bench-dev", "--config", "config.yaml"]) == 0

    target, command_args = calls[0]
    assert target.image == "curator:test"
    assert target.name == "bench-dev"
    assert command_args == ["--config", "config.yaml"]


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


def test_containerize_command_args_keeps_container_only_config_path() -> None:
    target = DockerTarget(container="bench-dev", benchmark_suite_dir=Path.cwd())

    args = _containerize_command_args(
        target,
        ["run", "--config", "/opt/curator-benchmark-suite/benchmarking/benchmarks.yaml"],
    )

    assert args == ["run", "--config", "/opt/curator-benchmark-suite/benchmarking/benchmarks.yaml"]


def test_containerize_command_args_does_not_require_suite_dir_for_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolve_benchmark_suite_dir(_path: str | Path | None = None) -> Path:
        msg = "no suite"
        raise ValueError(msg)

    monkeypatch.setattr(
        "curator_benchmarking.targets.docker.resolve_benchmark_suite_dir",
        fail_resolve_benchmark_suite_dir,
    )
    target = DockerTarget(container="bench-dev")

    args = _containerize_command_args(
        target,
        ["run", "--config", "/container-only/config.yaml"],
    )

    assert args == ["run", "--config", "/container-only/config.yaml"]


def test_containerize_command_args_rewrites_suite_config_for_existing_container(tmp_path: Path) -> None:
    suite_dir = tmp_path / "Curator"
    package_dir = suite_dir / "benchmarking"
    package_dir.mkdir(parents=True)
    (package_dir / "pyproject.toml").write_text("")
    config_path = package_dir / "benchmarks.yaml"

    target = DockerTarget(
        container="bench-dev",
        benchmark_suite_dir=suite_dir,
        benchmark_suite_container_dir=Path("/suite"),
    )

    args = _containerize_command_args(target, ["run", "--config", str(config_path)])

    assert args == ["run", "--config", "/suite/benchmarking/benchmarks.yaml"]


def test_containerize_command_args_rejects_missing_image_config(tmp_path: Path) -> None:
    target = DockerTarget(image="curator:test", benchmark_suite_dir=Path.cwd())

    with pytest.raises(ValueError, match="Config file does not exist"):
        _containerize_command_args(target, ["run", "--config", str(tmp_path / "missing.yaml")])


def test_docker_env_args_sets_container_path_mode() -> None:
    env_args = _docker_env_args(DockerTarget(), image_digest="sha256:test")

    assert f"{CURATOR_BENCHMARK_PATH_MODE_ENV}=container" in env_args


def test_setup_command_checks_requested_extra_imports() -> None:
    command = _setup_command(DockerTarget(benchmark_extras=["sinks"]))

    assert "curator_benchmarking" in command
    assert "runner" in command
    assert "mlflow" in command
    assert "pydrive2" in command
    assert "slack_sdk" in command


def test_setup_command_all_checks_every_known_extra() -> None:
    command = _setup_command(DockerTarget(benchmark_extras=["all"]))

    assert "pyloudnorm" in command
    assert "slack_sdk" in command
    assert "cv2" in command


def test_run_start_creates_detached_named_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  - name: results_path
    host_path: {tmp_path}
entries: []
"""
    )
    calls = []

    def fake_call(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    monkeypatch.setattr("curator_benchmarking.targets.docker.subprocess.call", fake_call)
    monkeypatch.setattr("curator_benchmarking.targets.docker._image_digest", lambda _image: "sha256:test")

    status = run_start(
        DockerTarget(
            image="curator:test",
            name="bench-dev",
            benchmark_setup="never",
            benchmark_suite_dir=Path.cwd(),
            memory="1g",
            shm_size="512m",
        ),
        ["--config", str(config_path)],
    )

    assert status == 0
    assert len(calls) == 1
    command = calls[0]
    assert command[:4] == ["docker", "run", "--detach", "--name"]
    assert command[4] == "bench-dev"
    assert "--rm" not in command
    assert "--volume" in command
    assert command[-5:] == ["--entrypoint", "bash", "curator:test", "-lc", "sleep infinity"]
