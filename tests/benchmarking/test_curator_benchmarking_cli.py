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
from curator_benchmarking.commands import check, setup
from curator_benchmarking.paths import BENCHMARK_SUITE_DIR_ENV
from curator_benchmarking.targets.docker import (
    DEFAULT_BENCHMARK_CONFIG_ENV,
    DockerTarget,
    _containerize_command_args,
    _dependency_groups_from_command_args,
    _docker_env_args,
    _setup_command,
    run_setup,
    run_shell,
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


def test_cli_container_target_does_not_set_docker_run_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run_in_target(target: DockerTarget, command_args: list[str]) -> int:
        calls.append((target, command_args))
        return 0

    monkeypatch.setattr("curator_benchmarking.cli.run_in_target", fake_run_in_target)

    assert main(["run", "--container", "bench-dev", "--config", "config.yaml"]) == 0

    target, command_args = calls[0]
    assert target.container == "bench-dev"
    assert target.gpus is None
    assert target.memory is None
    assert target.shm_size is None
    assert target.network is None
    assert command_args == ["run", "--config", "config.yaml"]


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        ("--gpus", "all"),
        ("--container-memory", "1t"),
        ("--shm-size", "512g"),
        ("--network", "host"),
    ],
)
def test_cli_rejects_docker_run_options_with_container(
    option_name: str,
    option_value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = main(
        [
            "run",
            "--container",
            "bench-dev",
            option_name,
            option_value,
            "--config",
            "config.yaml",
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert option_name in captured.err
    assert "cannot be combined with --container" in captured.err


def test_cli_parses_setup_benchmark_env_option(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run_in_target(target: DockerTarget, command_args: list[str]) -> int:
        calls.append((target, command_args))
        return 0

    monkeypatch.setattr("curator_benchmarking.cli.run_in_target", fake_run_in_target)

    assert (
        main(
            [
                "run",
                "--image",
                "curator:test",
                "--setup-benchmark-env",
                "no",
                "--config",
                "config.yaml",
            ]
        )
        == 0
    )

    target, _command_args = calls[0]
    assert target.setup_benchmark_env == "no"


def test_cli_start_dispatches_to_docker_target(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run_start(target: DockerTarget, command_args: list[str]) -> int:
        calls.append((target, command_args))
        return 0

    monkeypatch.setattr("curator_benchmarking.cli.run_start", fake_run_start)

    assert (
        main(
            [
                "start",
                "--image",
                "curator:test",
                "--name",
                "bench-dev",
                "--config",
                "config.yaml",
            ]
        )
        == 0
    )

    target, command_args = calls[0]
    assert target.image == "curator:test"
    assert target.name == "bench-dev"
    assert command_args == ["--config", "config.yaml"]


def test_cli_setup_passes_setup_args_to_docker_target(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run_setup(target: DockerTarget, setup_args: list[str]) -> int:
        calls.append((target, setup_args))
        return 0

    monkeypatch.setattr("curator_benchmarking.cli.run_setup", fake_run_setup)

    assert (
        main(
            [
                "setup",
                "--image",
                "curator:test",
                "--entry-name",
                "audio_readspeech_xenna",
            ]
        )
        == 0
    )

    target, setup_args = calls[0]
    assert target.image == "curator:test"
    assert setup_args == ["--entry-name", "audio_readspeech_xenna"]


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


def test_containerize_command_args_rewrites_suite_config_for_image(tmp_path: Path) -> None:
    suite_dir = tmp_path / "Curator" / "benchmarking"
    suite_dir.mkdir(parents=True)
    (suite_dir / "curator_benchmarking").mkdir()
    (suite_dir / "pyproject.toml").write_text("")
    config_path = suite_dir / "benchmarks.yaml"
    config_path.write_text("entries: []\n")

    target = DockerTarget(
        image="curator:test",
        benchmark_suite_dir=suite_dir,
    )

    args = _containerize_command_args(target, ["run", "--config", str(config_path)])

    assert args == ["run", "--config", "/opt/curator-benchmark-suite/benchmarks.yaml"]


def test_containerize_command_args_keeps_container_only_config_path() -> None:
    target = DockerTarget(container="bench-dev", benchmark_suite_dir=Path.cwd())

    args = _containerize_command_args(
        target,
        ["run", "--config", "/opt/curator-benchmark-suite/benchmarks.yaml"],
    )

    assert args == ["run", "--config", "/opt/curator-benchmark-suite/benchmarks.yaml"]


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
    benchmark_package_dir = suite_dir / "benchmarking"
    benchmark_package_dir.mkdir(parents=True)
    (benchmark_package_dir / "curator_benchmarking").mkdir()
    (benchmark_package_dir / "pyproject.toml").write_text("")
    config_path = benchmark_package_dir / "benchmarks.yaml"

    target = DockerTarget(
        container="bench-dev",
        benchmark_suite_dir=suite_dir,
        benchmark_suite_container_dir=Path("/suite"),
    )

    args = _containerize_command_args(target, ["run", "--config", str(config_path)])

    assert args == ["run", "--config", "/suite/benchmarks.yaml"]


def test_containerize_command_args_rejects_missing_image_config(tmp_path: Path) -> None:
    target = DockerTarget(image="curator:test", benchmark_suite_dir=Path.cwd())

    with pytest.raises(ValueError, match="Config file does not exist"):
        _containerize_command_args(target, ["run", "--config", str(tmp_path / "missing.yaml")])


def test_docker_env_args_sets_container_path_mode() -> None:
    env_args = _docker_env_args(DockerTarget(), image_digest="sha256:test")

    assert f"{CURATOR_BENCHMARK_PATH_MODE_ENV}=container" in env_args
    assert f"{BENCHMARK_SUITE_DIR_ENV}=/opt/curator-benchmark-suite" in env_args
    assert f"{DEFAULT_BENCHMARK_CONFIG_ENV}=/opt/curator-benchmark-suite/benchmarks.yaml" in env_args


def test_setup_command_installs_dependency_group_extra_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
sinks:
  - name: slack
    enabled: true
    dependencies:
      - sinks
entries: []
""".lstrip()
    )

    command = _setup_command(DockerTarget(), command_args=["run", "--config", str(config_path)])

    assert "uv --no-config pip install" in command
    assert "/opt/curator-benchmark-suite[sinks]" in command


def test_setup_command_all_installs_all_extra() -> None:
    command = _setup_command(DockerTarget(), command_args=["setup"], force=True)

    assert "/opt/curator-benchmark-suite[all]" in command


def test_forced_setup_command_all_installs_all_system_dependency_scripts() -> None:
    command = _setup_command(DockerTarget(), command_args=["setup"], force=True)

    assert "system_deps/audio/install.sh" in command
    assert "system_deps/math/install.sh" in command
    assert "system_deps/video/install.sh" in command


def test_setup_command_installs_audio_system_dependency_script_for_audio_entry(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  - name: results_path
    host_path: /results
entries:
  - name: audio_readspeech_xenna
    script: audio.py
    dependencies:
      - audio
""".lstrip()
    )

    command = _setup_command(
        DockerTarget(),
        command_args=["run", "--config", str(config_path), "--entries-exact", "audio_readspeech_xenna"],
    )

    assert "system_deps/audio/install.sh" in command
    assert "system_deps/video/install.sh" not in command


def test_setup_command_installs_video_system_dependency_script_for_video_entry(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  - name: results_path
    host_path: /results
entries:
  - name: video_transcoding_xenna
    script: video.py
    dependencies:
      - video
""".lstrip()
    )

    command = _setup_command(
        DockerTarget(),
        command_args=["run", "--config", str(config_path), "--entries-exact", "video_transcoding_xenna"],
    )

    assert "system_deps/video/install.sh" in command
    assert "system_deps/audio/install.sh" not in command


def test_setup_command_does_not_install_system_dependencies_for_check() -> None:
    command = _setup_command(DockerTarget(), command_args=["check"])

    assert "system_deps/" not in command


def test_check_reports_pyproject_python_dependency_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package_dir = tmp_path / "benchmarking"
    (package_dir / "curator_benchmarking").mkdir(parents=True)
    (package_dir / "pyproject.toml").write_text(
        """
[project]
dependencies = [
    "core-ok>=1",
    "core-missing",
    "core-too-old>=2",
]

[project.optional-dependencies]
video = [
    "video-ok==2.0",
    "video-too-old>=3",
]
""".lstrip()
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
entries:
  - name: video_transcoding_xenna
    dependencies:
      - video
""".lstrip()
    )
    versions = {
        "core-ok": "1.2.0",
        "core-too-old": "1.0.0",
        "nemo-curator": "1.0.0",
        "nemo-curator-benchmarking": "0.1.0",
        "runner": "0.1.0",
        "video-ok": "2.0",
        "video-too-old": "2.5",
    }

    def fake_version(package_name: str) -> str:
        if package_name in versions:
            return versions[package_name]
        raise check.importlib.metadata.PackageNotFoundError(package_name)

    monkeypatch.setattr(check.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(check.importlib.util, "find_spec", lambda _module_name: object())

    status = check.main(
        [
            "--benchmark-suite-dir",
            str(package_dir),
            "--config",
            str(config_path),
            "--entry-name",
            "video_transcoding_xenna",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "OK       python-dep:core-ok: 1.2.0" in captured.out
    assert "MISSING  python-dep:core-missing: core-missing" in captured.out
    assert "FAILED   python-dep:core-too-old: installed 1.0.0 does not satisfy >=2" in captured.out
    assert "OK       python-dep:video:video-ok: 2.0" in captured.out
    assert "FAILED   python-dep:video:video-too-old: installed 2.5 does not satisfy >=3" in captured.out


def test_dependency_groups_uses_config_entry_names(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  - name: results_path
    host_path: /results
entries:
  - name: audio_readspeech_xenna
    script: audio.py
    dependencies:
      - audio
  - name: math_preprocess_xenna
    script: math.py
    dependencies:
      - math
""".lstrip()
    )

    assert _dependency_groups_from_command_args(["run", "--config", str(config_path)]) == ["audio", "math"]


def test_dependency_groups_uses_config_data_setup_names(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  - name: results_path
    host_path: /results
entries: []
data_setups:
  - name: audio_sortformer_librispeech_450h
    script: prepare_audio.py
    dependencies:
      - audio
""".lstrip()
    )

    assert _dependency_groups_from_command_args(["run", "--config", str(config_path)]) == ["audio"]


def test_setup_command_entry_name_selects_minimal_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_call(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    package_dir = tmp_path / "benchmarking"
    package_dir.mkdir()
    (package_dir / "pyproject.toml").write_text(
        """
[project.optional-dependencies]
audio = []
""".lstrip()
    )

    monkeypatch.setattr(setup, "benchmark_package_dir", lambda _suite_dir: package_dir)
    monkeypatch.setattr(setup.shutil, "which", lambda _name: None)
    monkeypatch.setattr(setup.subprocess, "call", fake_call)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
entries:
  - name: audio_readspeech_xenna
    dependencies:
      - audio
""".lstrip()
    )

    assert setup.main(["--config", str(config_path), "--entry-name", "audio_readspeech_xenna"]) == 0

    assert calls == [[sys.executable, "-m", "pip", "install", f"{package_dir}[audio]"]]


def test_local_setup_uses_uv_without_project_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_call(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    package_dir = tmp_path / "benchmarking"
    package_dir.mkdir()
    (package_dir / "pyproject.toml").write_text(
        """
[project.optional-dependencies]
sinks = []
""".lstrip()
    )

    monkeypatch.setattr(setup, "benchmark_package_dir", lambda _suite_dir: package_dir)
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(setup.subprocess, "call", fake_call)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
sinks:
  - name: slack
    dependencies:
      - sinks
entries: []
""".lstrip()
    )

    assert setup.main(["--config", str(config_path)]) == 0

    assert calls == [["uv", "--no-config", "pip", "install", f"{package_dir}[sinks]"]]


def test_setup_command_installs_system_tools_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    system_tool_calls = []

    def fake_call(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    def fake_install_system_dependencies(
        dependency_groups: list[str],
        *,
        suite_dir: Path,
    ) -> int:
        system_tool_calls.append((dependency_groups, suite_dir))
        return 0

    package_dir = tmp_path / "benchmarking"
    package_dir.mkdir()
    (package_dir / "pyproject.toml").write_text(
        """
[project.optional-dependencies]
video = []
""".lstrip()
    )

    monkeypatch.setattr(setup, "benchmark_package_dir", lambda _suite_dir: package_dir)
    monkeypatch.setattr(setup.shutil, "which", lambda _name: None)
    monkeypatch.setattr(setup.subprocess, "call", fake_call)
    monkeypatch.setattr(setup, "install_system_dependencies", fake_install_system_dependencies)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
entries:
  - name: video_transcoding_xenna
    dependencies:
      - video
""".lstrip()
    )

    assert setup.main(["--config", str(config_path), "--entry-name", "video_transcoding_xenna"]) == 0
    assert system_tool_calls == []

    assert (
        setup.main(
            [
                "--install-system-tools",
                "--config",
                str(config_path),
                "--entry-name",
                "video_transcoding_xenna",
            ]
        )
        == 0
    )
    assert system_tool_calls == [
        (["video"], package_dir),
    ]


def test_run_setup_uses_entry_name_for_docker_system_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  - name: results_path
    host_path: /results
entries:
  - name: audio_readspeech_xenna
    script: audio.py
    dependencies:
      - audio
""".lstrip()
    )
    calls = []

    def fake_run_shell_command(target: DockerTarget, shell_command: str, command_args: list[str]) -> int:
        calls.append((target, shell_command, command_args))
        return 0

    monkeypatch.setattr(
        "curator_benchmarking.targets.docker._run_shell_command",
        fake_run_shell_command,
    )

    assert (
        run_setup(
            DockerTarget(container="bench-dev"),
            ["--config", str(config_path), "--entry-name", "audio_readspeech_xenna"],
        )
        == 0
    )

    _, shell_command, command_args = calls[0]
    assert command_args == []
    assert "system_deps/audio/install.sh" in shell_command
    assert "system_deps/video/install.sh" not in shell_command


def test_run_shell_with_image_uses_context_config_for_mounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  - name: results_path
    host_path: {tmp_path}
entries: []
""".lstrip()
    )
    calls = []

    def fake_call(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    monkeypatch.setattr("curator_benchmarking.targets.docker.subprocess.call", fake_call)
    monkeypatch.setattr("curator_benchmarking.targets.docker._image_digest", lambda _image: "sha256:test")

    status = run_shell(
        DockerTarget(
            image="curator:test",
            setup_benchmark_env="no",
            benchmark_suite_dir=Path.cwd(),
            gpus="none",
            memory="1g",
            shm_size="512m",
        ),
        ["--config", str(config_path)],
    )

    assert status == 0
    command = calls[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--workdir" not in command
    assert f"{Path.cwd() / 'benchmarking'}:/opt/curator-benchmark-suite" in command
    assert f"{tmp_path}:{Path(f'/MOUNT/{tmp_path}')}" in command
    assert f"{config_path}:{Path(f'/MOUNT/{config_path}')}" in command
    assert command[-5:-2] == ["--entrypoint", "bash", "curator:test"]
    assert "Curator benchmark shell" in command[-1]
    assert "/opt/curator-benchmark-suite/benchmarks.yaml" in command[-1]
    assert str(Path(f"/MOUNT/{config_path}")) in command[-1]


def test_run_shell_with_image_passes_command_after_separator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  - name: results_path
    host_path: {tmp_path}
entries: []
""".lstrip()
    )
    calls = []

    def fake_run_shell_command(
        target: DockerTarget,
        shell_command: str,
        command_args: list[str],
    ) -> int:
        calls.append((target, shell_command, command_args))
        return 0

    monkeypatch.setattr(
        "curator_benchmarking.targets.docker._run_shell_command",
        fake_run_shell_command,
    )

    status = run_shell(
        DockerTarget(
            image="curator:test",
            setup_benchmark_env="no",
            benchmark_suite_dir=Path.cwd(),
        ),
        ["--config", str(config_path), "--", "echo", "hello"],
    )

    assert status == 0
    _target, shell_command, command_args = calls[0]
    assert shell_command.endswith(" && echo hello")
    assert command_args == ["--config", str(config_path)]


def test_run_shell_with_existing_container_prints_mount_reminder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_call(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    monkeypatch.setattr("curator_benchmarking.targets.docker.subprocess.call", fake_call)

    status = run_shell(
        DockerTarget(container="bench-dev", setup_benchmark_env="no"),
        [],
    )

    assert status == 0
    command = calls[0]
    assert command[:2] == ["docker", "exec"]
    assert command[-3:-1] == ["bash", "-lc"]
    assert "Curator benchmark shell" in command[-1]
    assert "Existing containers must already have required mounts." in command[-1]


def test_cli_shell_passes_context_args_and_command_after_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run_shell(target: DockerTarget, shell_args: list[str]) -> int:
        calls.append((target, shell_args))
        return 0

    monkeypatch.setattr("curator_benchmarking.cli.run_shell", fake_run_shell)

    assert (
        main(
            [
                "shell",
                "--image",
                "curator:test",
                "--config",
                "config.yaml",
                "--",
                "echo",
                "hello",
            ]
        )
        == 0
    )

    target, shell_args = calls[0]
    assert target.image == "curator:test"
    assert shell_args == ["--config", "config.yaml", "--", "echo", "hello"]


def test_run_shell_rejects_context_args_for_existing_container(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("entries: []\n")

    with pytest.raises(ValueError, match="docker exec cannot add mounts"):
        run_shell(
            DockerTarget(container="bench-dev"),
            ["--config", str(config_path)],
        )


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
            setup_benchmark_env="no",
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
    assert "--workdir" not in command
    assert command[-5:] == ["--entrypoint", "bash", "curator:test", "-lc", "sleep infinity"]
