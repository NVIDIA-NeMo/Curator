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

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import packaging
import pytest


@pytest.mark.parametrize("compatible_under_test", [False, True])
@pytest.mark.parametrize("check_only", [False, True])
def test_setup_discovers_compatible_installers_and_skips_legacy_helpers(
    tmp_path: Path, compatible_under_test: bool, check_only: bool
) -> None:
    source = tmp_path / "source"
    under_test = tmp_path / "under-test"
    setup = source / "benchmarking/tools/setup_benchmark_env.sh"
    setup.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[3] / "benchmarking/tools/setup_benchmark_env.sh", setup)
    (source / "benchmarking/requirements.txt").write_text("")
    log = tmp_path / "installers.log"
    for root in (source, under_test):
        common = root / "docker/common"
        common.mkdir(parents=True)
        (common / "install_example.sh").write_text(f'printf "%s\\n" "{root.name}:$*" >> "$TEST_INSTALLERS_LOG"\n')
        for name in ("install_etcd_nats.sh", "install_h264_support.sh"):
            (common / name).write_text('echo "Legacy helper must not run" >&2\nexit 1\n')
    if compatible_under_test:
        marker = under_test / "benchmarking/tools/setup_benchmark_env.sh"
        marker.parent.mkdir(parents=True)
        marker.touch()

    # Isolate installer discovery from pip mutation; run the real Python dependency check.
    python = tmp_path / "python"
    python.write_text(f'#!/bin/bash\nif [ "$1" = "-m" ]; then exit 0; fi\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(setup), *(["--check"] if check_only else [])],
        env={
            **os.environ,
            "PYTHON": str(python),
            "TEST_INSTALLERS_LOG": str(log),
            "CURATOR_BENCHMARK_CURATOR_REPO_DIR": str(under_test),
            "CURATOR_BENCHMARK_SOURCE_DIR": str(source),
            "CURATOR_BENCHMARK_SOURCE_MOUNT_DIR": str(tmp_path / "not-mounted"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    selected = "under-test" if compatible_under_test else "source"
    expected = [f"{selected}:--check"] if check_only else [f"{selected}:", f"{selected}:--check"]
    assert log.read_text().splitlines() == expected
    assert ("WARNING:" in result.stderr) is not compatible_under_test


def test_check_reports_missing_pytest_in_minimal_python(tmp_path: Path) -> None:
    source = tmp_path / "source"
    setup = source / "benchmarking/tools/setup_benchmark_env.sh"
    setup.parent.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[3]
    shutil.copyfile(repo / "benchmarking/tools/setup_benchmark_env.sh", setup)
    shutil.copyfile(repo / "benchmarking/requirements.txt", source / "benchmarking/requirements.txt")
    (source / "docker/common").mkdir(parents=True)
    python = tmp_path / "python"
    # -S excludes installed third-party packages without creating another environment.
    python.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} -S "$@"\n')
    python.chmod(0o755)
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "packaging").symlink_to(Path(packaging.__file__).parent, target_is_directory=True)

    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(setup), "--check"],
        env={
            **os.environ,
            "PYTHON": str(python),
            "PYTHONPATH": str(packages),
            "CURATOR_BENCHMARK_CURATOR_REPO_DIR": str(source),
            "CURATOR_BENCHMARK_SOURCE_DIR": str(source),
            "CURATOR_BENCHMARK_SOURCE_MOUNT_DIR": str(tmp_path / "not-mounted"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "MISSING  pytest" in result.stdout


@pytest.mark.parametrize(
    ("requirement", "package", "installed", "expected_status", "expected_output"),
    [
        ("albumentations==2.0.8", "albumentations", None, 1, "MISSING  albumentations"),
        ("albumentations==2.0.8", "albumentations", "1.4.0", 1, "required ==2.0.8"),
        ("albumentations==2.0.8", "albumentations", "2.0.8", 0, "OK       albumentations: 2.0.8"),
        ("albumentations>=2.0.8", "albumentations", "2.1.0rc1", 0, "OK       albumentations: 2.1.0rc1"),
        ("rich", "rich", "1.0", 0, "OK       rich: 1.0"),
        ("# comment\n\nalbumentations==2.0.8 # pinned", "albumentations", "2.0.8", 0, "OK       albumentations"),
        ('albumentations==2.0.8; python_version < "1.0"', "albumentations", None, 0, "OK       nemo_curator"),
    ],
)
def test_check_validates_installed_requirement_versions(  # noqa: PLR0913
    tmp_path: Path, requirement: str, package: str, installed: str | None, expected_status: int, expected_output: str
) -> None:
    source = tmp_path / "source"
    setup = source / "benchmarking/tools/setup_benchmark_env.sh"
    setup.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[3] / "benchmarking/tools/setup_benchmark_env.sh", setup)
    (source / "benchmarking/requirements.txt").write_text(requirement + "\n")
    (source / "docker/common").mkdir(parents=True)
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "packaging").symlink_to(Path(packaging.__file__).parent, target_is_directory=True)
    (packages / "nemo_curator.py").touch()
    if installed is not None:
        metadata = packages / f"{package}-{installed}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Name: {package}\nVersion: {installed}\n")
    python = tmp_path / "python"
    python.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} -S "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(setup), "--check"],
        env={
            **os.environ,
            "PYTHON": str(python),
            "PYTHONPATH": str(packages),
            "CURATOR_BENCHMARK_CURATOR_REPO_DIR": str(source),
            "CURATOR_BENCHMARK_SOURCE_DIR": str(source),
            "CURATOR_BENCHMARK_SOURCE_MOUNT_DIR": str(tmp_path / "not-mounted"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected_status, result.stderr
    assert expected_output in result.stdout


def test_check_reports_missing_packaging(tmp_path: Path) -> None:
    setup = Path(__file__).resolve().parents[3] / "benchmarking/tools/setup_benchmark_env.sh"
    source = tmp_path / "source"
    (source / "benchmarking").mkdir(parents=True)
    (source / "benchmarking/requirements.txt").write_text("packaging\n")
    (source / "docker/common").mkdir(parents=True)
    python = tmp_path / "python"
    python.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} -S "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(setup), "--check"],
        env={
            **os.environ,
            "PYTHON": str(python),
            "PYTHONPATH": "",
            "CURATOR_BENCHMARK_CURATOR_REPO_DIR": str(source),
            "CURATOR_BENCHMARK_SOURCE_DIR": str(source),
            "CURATOR_BENCHMARK_SOURCE_MOUNT_DIR": str(tmp_path / "not-mounted"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "MISSING  packaging" in result.stdout
    assert "Traceback" not in result.stderr
