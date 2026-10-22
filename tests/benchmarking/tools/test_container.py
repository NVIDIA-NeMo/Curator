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

"""Container lifecycle tests using real source preparation and setup subprocesses."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/tools"))

import container as container_tool


@pytest.mark.parametrize(
    ("options", "expected_memory", "expected_shm"),
    [
        ([], 1024, 512),
        (["--container-memory", "256"], 256, 128),
        (["--shm-size", "64"], 1024, 64),
        (["--container-memory", "256", "--shm-size", "64"], 256, 64),
    ],
)
def test_start_memory_options(
    monkeypatch: pytest.MonkeyPatch, options: list[str], expected_memory: int, expected_shm: int
) -> None:
    run = Mock()
    monkeypatch.setattr(container_tool, "get_total_memory_bytes", lambda: 1024)
    monkeypatch.setattr(container_tool, "_run", run)
    monkeypatch.setattr(container_tool, "_docker_env_args", lambda _image: [])
    monkeypatch.setattr(container_tool, "_run_tool_script", Mock())

    assert container_tool.main(["start", "--setup-benchmark-env", "no", *options]) == 0
    command = run.call_args.args[0]
    assert command[command.index("--memory") + 1] == str(expected_memory)
    assert command[command.index("--shm-size") + 1] == str(expected_shm)


@pytest.mark.parametrize("setup", ["yes", "no"])
def test_start_copies_run_refreshes_and_check_does_not_modify_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: str
) -> None:
    source = tmp_path / "mount"
    runtime = tmp_path / "runtime"
    tools = source / "benchmarking/tools"
    tools.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[3]
    for name in ("prepare_benchmark_source.py", "setup_benchmark_env.sh"):
        shutil.copyfile(repo / "benchmarking/tools" / name, tools / name)
    (source / "benchmarking/requirements.txt").write_text("")
    (source / "benchmarking/config.yaml").write_text("original")
    (source / "benchmarking/obsolete.py").touch()
    (source / "tutorials").mkdir()
    (source / "tutorials/example.py").write_text("original tutorial")
    (source / "nemo_curator").mkdir()
    (source / "benchmarking/run.py").write_text(
        "from pathlib import Path\nprint(Path(__file__).with_name('config.yaml').read_text())\n"
    )
    (source / "docker/common").mkdir(parents=True)
    pip_log = tmp_path / "pip.log"
    python = tmp_path / "python"
    python.write_text(
        '#!/bin/bash\nif [ "$1" = "-m" ]; then\n'
        f'  printf "%s\\n" "$*" >> {shlex.quote(str(pip_log))}\n'
        "  exit 0\nfi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PYTHON": str(python),
        "CURATOR_BENCHMARK_SOURCE_MOUNT_DIR": str(source),
        "CURATOR_BENCHMARK_SOURCE_DIR": str(runtime),
        "CURATOR_BENCHMARK_CURATOR_REPO_DIR": str(source),
    }

    def docker_boundary(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(cmd, 0)
        assert cmd[:3] == ["docker", "exec", "test-container"]
        return subprocess.run(cmd[3:], env=env, capture_output=True, text=True, check=check)  # noqa: S603

    monkeypatch.setattr(container_tool, "HOST_REPO_ROOT", source)
    monkeypatch.setattr(container_tool, "CONTAINER_BENCHMARK_SOURCE_MOUNT_DIR", source)
    monkeypatch.setattr(container_tool, "CONTAINER_BENCHMARK_SOURCE_DIR", runtime)
    monkeypatch.setattr(container_tool, "CONTAINER_TOOL_DIRS", (tools, runtime / "benchmarking/tools"))
    monkeypatch.setattr(container_tool, "_run", docker_boundary)
    monkeypatch.setattr(container_tool, "_docker_env_args", lambda _image: [])

    assert (
        container_tool.main(["start", "--name", "test-container", "--gpus", "none", "--setup-benchmark-env", setup])
        == 0
    )
    assert (runtime / "benchmarking/config.yaml").read_text() == "original"
    assert pip_log.exists() is (setup == "yes")
    assert not (runtime / "nemo_curator").exists()

    (source / "benchmarking/config.yaml").write_text("updated")
    (source / "benchmarking/obsolete.py").unlink()
    (source / "tutorials/example.py").write_text("updated tutorial")
    assert container_tool.main(["check", "--container", "test-container"]) == 0
    assert (runtime / "benchmarking/config.yaml").read_text() == "original"
    assert (runtime / "benchmarking/obsolete.py").exists()
    assert container_tool.main(["run", "--container", "test-container"]) == 0
    assert (runtime / "benchmarking/config.yaml").read_text() == "updated"
    assert (runtime / "tutorials/example.py").read_text() == "updated tutorial"
    assert not (runtime / "benchmarking/obsolete.py").exists()
    assert pip_log.exists() is (setup == "yes")
    if pip_log.exists():
        assert len(pip_log.read_text().splitlines()) == 1
