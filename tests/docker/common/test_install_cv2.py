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

"""Verify cv2 installer dispatch against real Python distribution metadata."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[3] / "docker/common/install_cv2.sh"


@pytest.mark.parametrize("editable", [True, False, None], ids=["editable", "local-wheel", "wheel"])
def test_install_preserves_curator_installation_mode(tmp_path: Path, editable: bool | None) -> None:
    metadata = tmp_path / "nemo_curator-0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: nemo-curator\nVersion: 0.0\n")
    if editable is not None:
        (metadata / "direct_url.json").write_text(
            json.dumps({"url": tmp_path.as_uri(), "dir_info": {"editable": editable}})
        )
    ready = tmp_path / "installed"
    (tmp_path / "cv2.py").write_text(
        "from pathlib import Path\n"
        f"if not Path({str(ready)!r}).exists():\n"
        "    raise ImportError('cv2 not installed')\n"
        "__version__ = 'test'\n"
    )
    log = tmp_path / "pip.log"
    python = tmp_path / "python"
    # Stub only pip mutation; run the installer checks and metadata lookup with Python.
    python.write_text(
        '#!/bin/bash\nif [ "$1" = "-m" ]; then\n'
        f'  printf "%s\\n" "$*" >> {shlex.quote(str(log))}\n'
        f"  touch {shlex.quote(str(ready))}\n"
        "  exit 0\nfi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PYTHON": str(python),
        "PYTHONPATH": str(tmp_path),
        "CURATOR_BENCHMARK_CURATOR_REPO_DIR": str(tmp_path),
    }

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            ["/bin/bash", str(INSTALLER), *args], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
        )

    assert run("--check").returncode == 1
    assert not log.exists()
    installed = run()
    assert installed.returncode == 0, installed.stderr
    commands = [shlex.split(line) for line in log.read_text().splitlines()]
    assert commands == [
        [
            "-m",
            "pip",
            "install",
            "--upgrade-strategy",
            "only-if-needed",
            *(["--editable"] if editable else []),
            ".[cv2]",
        ],
        ["-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "fsspec==2026.3.0"],
    ]
    assert run("--check").returncode == 0
    assert run().returncode == 0
    assert len(log.read_text().splitlines()) == 2
