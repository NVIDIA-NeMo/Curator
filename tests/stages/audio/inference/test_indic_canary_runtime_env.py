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

"""Tests for the independently locked Indic Canary runtime environment."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_curator.stages.audio.inference import indic_canary_runtime_env as runtime_env


def test_packaged_runtime_project_is_complete() -> None:
    project = runtime_env.runtime_project_dir()

    assert (project / "pyproject.toml").is_file()
    assert (project / "uv.lock").is_file()
    assert len(runtime_env.runtime_lock_digest()) == 64


def test_resolve_runtime_python_prefers_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.touch(mode=0o755)
    validate = MagicMock(return_value=python)
    monkeypatch.setattr(runtime_env, "_validate_runtime_python", validate)

    assert runtime_env.resolve_runtime_python(str(python)) == python
    validate.assert_called_once_with(python)


def test_resolve_runtime_python_reports_provisioning_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEMO_CURATOR_INDIC_CANARY_RUNTIME_PYTHON", raising=False)
    monkeypatch.setattr(runtime_env, "_SYSTEM_RUNTIME_ROOT", tmp_path / "system")
    monkeypatch.setattr(runtime_env, "default_runtime_root", lambda: tmp_path / "cache")

    with pytest.raises(FileNotFoundError, match="install_indic_canary_trtllm_runtime"):
        runtime_env.resolve_runtime_python()


def test_ensure_runtime_python_provisions_only_implicit_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = tmp_path / "runtime" / "bin" / "python"
    install = MagicMock(return_value=installed)
    monkeypatch.delenv("NEMO_CURATOR_INDIC_CANARY_RUNTIME_PYTHON", raising=False)
    monkeypatch.setattr(runtime_env, "_SYSTEM_RUNTIME_ROOT", tmp_path / "system")
    monkeypatch.setattr(runtime_env, "install_runtime", install)

    assert runtime_env.ensure_runtime_python() == installed
    install.assert_called_once_with()


@pytest.mark.parametrize("source", ["configured", "environment"])
def test_ensure_runtime_python_does_not_mask_invalid_explicit_runtime(
    source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = "/configured/python" if source == "configured" else None
    if source == "environment":
        monkeypatch.setenv("NEMO_CURATOR_INDIC_CANARY_RUNTIME_PYTHON", "/environment/python")
    resolve = MagicMock(side_effect=FileNotFoundError("missing"))
    install = MagicMock()
    monkeypatch.setattr(runtime_env, "resolve_runtime_python", resolve)
    monkeypatch.setattr(runtime_env, "install_runtime", install)

    with pytest.raises(FileNotFoundError, match="missing"):
        runtime_env.ensure_runtime_python(configured)

    install.assert_not_called()


def test_ensure_runtime_python_keeps_prebuilt_system_runtime_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_root = tmp_path / "system"
    system_python = system_root / "bin" / "python"
    system_python.parent.mkdir(parents=True)
    system_python.touch(mode=0o755)
    validate = MagicMock(side_effect=RuntimeError("invalid system runtime"))
    install = MagicMock()
    monkeypatch.delenv("NEMO_CURATOR_INDIC_CANARY_RUNTIME_PYTHON", raising=False)
    monkeypatch.setattr(runtime_env, "_SYSTEM_RUNTIME_ROOT", system_root)
    monkeypatch.setattr(runtime_env, "_validate_runtime_python", validate)
    monkeypatch.setattr(runtime_env, "install_runtime", install)

    with pytest.raises(RuntimeError, match="invalid system runtime"):
        runtime_env.ensure_runtime_python()

    validate.assert_called_once_with(system_python)
    install.assert_not_called()


def test_runtime_subprocess_environment_isolates_python_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "runtime" / "bin" / "python"
    monkeypatch.setattr(runtime_env, "_validate_runtime_python", lambda path: path)
    monkeypatch.setenv("VIRTUAL_ENV", "/curator/main")
    monkeypatch.setenv("PYTHONPATH", "/curator/main/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/curator/python")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/curator/main")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    monkeypatch.setenv("PMI_SIZE", "8")
    monkeypatch.setenv("PMIX_RANK", "3")
    monkeypatch.setenv("MPI_LOCALRANKID", "3")
    monkeypatch.setenv("I_MPI_HYDRA_TOPOLIB", "hwloc")
    monkeypatch.setenv("MPICH_INTERFACE_HOSTNAME", "compute-3")
    monkeypatch.setenv("MV2_COMM_WORLD_RANK", "3")
    monkeypatch.setenv("HYDRA_PROXY_ID", "3")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("PATH", "/curator/main/bin:/usr/local/cuda/bin:/usr/bin")
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        "/curator/main/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64",
    )

    environment = runtime_env.runtime_subprocess_environment(python)

    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "UV_PROJECT_ENVIRONMENT" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["NVIDIA_VISIBLE_DEVICES"] == "1"
    for variable in (
        "OMPI_COMM_WORLD_RANK",
        "PMI_SIZE",
        "PMIX_RANK",
        "MPI_LOCALRANKID",
        "I_MPI_HYDRA_TOPOLIB",
        "MPICH_INTERFACE_HOSTNAME",
        "MV2_COMM_WORLD_RANK",
        "HYDRA_PROXY_ID",
        "SLURM_PROCID",
        "RANK",
        "WORLD_SIZE",
    ):
        assert variable not in environment
    assert environment["VIRTUAL_ENV"] == str(tmp_path / "runtime")
    assert environment["OPAL_PREFIX"] == str(tmp_path / "runtime")
    assert environment["CUDA_HOME"] == str(
        tmp_path / "runtime" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    )
    assert environment["PATH"].split(os.pathsep) == [str(tmp_path / "runtime" / "bin"), "/usr/bin"]
    library_path = environment["LD_LIBRARY_PATH"].split(os.pathsep)
    assert (
        str(tmp_path / "runtime" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "lib") in library_path
    )
    assert "/curator/main/lib" not in library_path
    assert "/usr/local/cuda/lib64" not in library_path
    assert library_path[-1] == "/usr/local/nvidia/lib64"


def test_validate_runtime_python_rejects_wrong_locked_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "runtime" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch(mode=0o755)
    run = MagicMock(return_value=MagicMock(returncode=1, stderr="torch==2.11.0 (expected 2.9.1+cu128)"))
    monkeypatch.setattr(runtime_env.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="does not match its locked stack"):
        runtime_env._validate_runtime_python(python)

    assert run.call_args.args[0][1:3] == ["-I", "-c"]


def test_validate_runtime_layout_requires_bundled_native_libraries(tmp_path: Path) -> None:
    python = tmp_path / "runtime" / "bin" / "python"

    with pytest.raises(RuntimeError, match=r"libmpi\.so\.40"):
        runtime_env._validate_runtime_layout(python)


def test_install_runtime_uses_frozen_lock_and_reuses_completed_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    uv = tmp_path / "uv"
    uv.touch(mode=0o755)
    monkeypatch.setattr(runtime_env.shutil, "which", lambda _name: str(uv))
    monkeypatch.setattr(runtime_env, "runtime_lock_digest", lambda: "a" * 64)
    monkeypatch.setattr(runtime_env, "_validate_runtime_python", lambda path: path)
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> MagicMock:
        calls.append(command)
        if "sync" in command:
            (runtime_root / "bin").mkdir(parents=True)
            (runtime_root / "bin" / "python").touch(mode=0o755)
        return MagicMock(returncode=0)

    monkeypatch.setattr(runtime_env.subprocess, "run", run)

    first = runtime_env.install_runtime(runtime_root)
    second = runtime_env.install_runtime(runtime_root)

    assert first == runtime_root / "bin" / "python"
    assert second == first
    assert len(calls) == 2
    assert calls[0][1:4] == ["sync", "--frozen", "--no-dev"]
    assert calls[1][1:4] == ["pip", "check", "--python"]
    assert (runtime_root / ".nemo_curator_runtime_lock_sha256").read_text().strip() == "a" * 64


def test_install_runtime_finds_uv_next_to_unactivated_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curator_bin = tmp_path / "curator" / "bin"
    curator_bin.mkdir(parents=True)
    python = curator_bin / "python"
    uv = curator_bin / "uv"
    python.touch(mode=0o755)
    uv.touch(mode=0o755)
    runtime_root = tmp_path / "runtime"
    calls: list[list[str]] = []

    monkeypatch.setattr(runtime_env.sys, "executable", str(python))
    monkeypatch.setattr(runtime_env.shutil, "which", lambda _name: None)
    monkeypatch.setattr(runtime_env, "runtime_lock_digest", lambda: "b" * 64)
    monkeypatch.setattr(runtime_env, "_validate_runtime_python", lambda path: path)

    def run(command: list[str], **_kwargs: object) -> MagicMock:
        calls.append(command)
        if "sync" in command:
            (runtime_root / "bin").mkdir(parents=True)
            (runtime_root / "bin" / "python").touch(mode=0o755)
        return MagicMock(returncode=0)

    monkeypatch.setattr(runtime_env.subprocess, "run", run)

    assert runtime_env.install_runtime(runtime_root) == runtime_root / "bin" / "python"
    assert calls[0][0] == str(uv)


def test_install_runtime_repairs_corrupt_completed_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    python = runtime_root / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch(mode=0o755)
    (runtime_root / ".nemo_curator_runtime_lock_sha256").write_text(f"{'c' * 64}\n")
    uv = tmp_path / "uv"
    uv.touch(mode=0o755)
    validate = MagicMock(side_effect=[RuntimeError("corrupt"), python])
    calls: list[list[str]] = []

    monkeypatch.setattr(runtime_env.shutil, "which", lambda _name: str(uv))
    monkeypatch.setattr(runtime_env, "runtime_lock_digest", lambda: "c" * 64)
    monkeypatch.setattr(runtime_env, "_validate_runtime_python", validate)
    monkeypatch.setattr(
        runtime_env.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or MagicMock(returncode=0),
    )

    assert runtime_env.install_runtime(runtime_root) == python
    assert "--reinstall" in calls[0]
    assert calls[1][1:4] == ["pip", "check", "--python"]
    assert validate.call_count == 2
