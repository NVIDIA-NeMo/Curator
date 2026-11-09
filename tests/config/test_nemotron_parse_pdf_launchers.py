# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Launchers fail promptly and clean up when their owned replica exits."""

import os
import subprocess
from pathlib import Path

import pytest

TUTORIAL = Path(__file__).parents[2] / "tutorials/interleaved/nemotron_parse_pdf"


@pytest.mark.parametrize("script", ["run_8gpu.sh", "run_8gpu_x100_direct.sh", "run_extractions.sh"])
def test_replica_failure_from_outside_checkout_is_bounded(tmp_path: Path, script: str):
    commands = tmp_path / "bin"
    commands.mkdir()
    for name, body in {
        "nvidia-smi": "#!/bin/bash\nprintf '%s\\n' {0..7}\n",
        "curl": "#!/bin/bash\nexit 22\n",
        "python": '#!/bin/bash\nif [[ "${1:-}" == -m ]]; then echo launched >> "$STARTED"; exit 7; fi\nexit 0\n',
    }.items():
        executable = commands / name
        executable.write_text(body)
        executable.chmod(0o755)
    model, pdfs, results = (tmp_path / name for name in ("model", "pdfs", "results"))
    for directory in (model, pdfs, results):
        directory.mkdir()
    for name in ("manifest.jsonl", "manifest-capped50.jsonl", "manifest-uncapped-window50.jsonl"):
        (results / name).write_text("{}\n")
    started = tmp_path / "started"
    environment = {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "ROOT": str(results),
        "MODEL": str(model),
        "PDF_DIR": str(pdfs),
        "MANIFEST": str(results / "manifest.jsonl"),
        "PROXY": str(TUTORIAL / "round_robin_proxy.py"),
        "PYTHON": str(commands / "python"),
        "STARTED": str(started),
    }
    result = subprocess.run(  # noqa: S603
        ["bash", str(TUTORIAL / script)],  # noqa: S607
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "exited before readiness" in result.stderr
    assert started.is_file()
    assert not list(results.glob("*/run-summary.json"))
