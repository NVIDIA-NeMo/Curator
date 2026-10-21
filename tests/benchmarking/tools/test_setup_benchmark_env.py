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
