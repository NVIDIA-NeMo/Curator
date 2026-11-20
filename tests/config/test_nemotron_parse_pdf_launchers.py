# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Launchers fail promptly and clean up when their owned replica exits."""

import os
import subprocess
from pathlib import Path

import pytest

TUTORIAL = Path(__file__).parents[2] / "tutorials/interleaved/nemotron_parse_pdf"


@pytest.mark.parametrize(
    ("script", "uncapped_manifest"),
    [
        ("run_8gpu.sh", "manifest-uncapped-window50.jsonl"),
        ("run_8gpu_x100_direct.sh", "manifest-uncapped-window50.jsonl"),
        ("run_extractions.sh", "manifest-uncapped-window50.jsonl"),
        ("run_extractions.sh", "manifest-uncapped.jsonl"),
    ],
)
def test_preflight_and_replica_failures_from_outside_checkout_are_bounded(
    tmp_path: Path, script: str, uncapped_manifest: str, launcher_env: dict[str, str]
) -> None:
    results = Path(launcher_env["ROOT"])
    (results / uncapped_manifest).write_text("{}\n")
    result = _run_launcher(tmp_path, script, launcher_env)
    started = Path(launcher_env["STARTED"])
    assert result.returncode != 0
    if script == "run_extractions.sh" and uncapped_manifest == "manifest-uncapped.jsonl":
        assert not started.exists()
        assert "manifest-uncapped-window50.jsonl" in result.stderr
    else:
        assert "exited before readiness" in result.stderr
        assert started.is_file()
    assert not list(results.glob("*/run-summary.json"))


@pytest.fixture
def launcher_env(tmp_path: Path) -> dict[str, str]:
    commands = tmp_path / "bin"
    commands.mkdir()
    for name, body in {
        "nvidia-smi": (
            "#!/bin/bash\n"
            "for ((gpu=0; gpu<9; gpu++)); do\n"
            '  printf \'%d, GPU-00000000-0000-0000-0000-%012d\\n\' "$gpu" "$gpu"\n'
            '  if [[ "$gpu" == "${LAST_GPU_INDEX:-7}" ]]; then break; fi\n'
            "done\n"
        ),
        "curl": "#!/bin/bash\nexit 22\n",
        "python": (
            '#!/bin/bash\nif [[ "${1:-}" == -m ]]; then echo "$CUDA_VISIBLE_DEVICES" >> "$STARTED"; exit 7; fi\nexit 0\n'
        ),
    }.items():
        executable = commands / name
        executable.write_text(body)
        executable.chmod(0o755)
    model, pdfs, results = (tmp_path / name for name in ("model", "pdfs", "results"))
    for directory in (model, pdfs, results):
        directory.mkdir()
    for name in ("manifest.jsonl", "manifest-capped50.jsonl"):
        (results / name).write_text("{}\n")
    environment = {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "ROOT": str(results),
        "MODEL": str(model),
        "PDF_DIR": str(pdfs),
        "MANIFEST": str(results / "manifest.jsonl"),
        "PROXY": str(TUTORIAL / "round_robin_proxy.py"),
        "PYTHON": str(commands / "python"),
        "STARTED": str(tmp_path / "started"),
    }
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    return environment


def _run_launcher(directory: Path, script: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["bash", str(TUTORIAL / script)],  # noqa: S607
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize("script", ["run_8gpu.sh", "run_8gpu_x100_direct.sh", "run_extractions.sh"])
@pytest.mark.parametrize("selection", ["", "0,4,5,6", "0,1,2,3,4,5,6,6", "0,1,2,3,4,5,6,9"])
def test_direct_launchers_refuse_insufficient_or_invalid_assignments(
    tmp_path: Path, launcher_env: dict[str, str], script: str, selection: str
) -> None:
    result = _run_launcher(tmp_path, script, {**launcher_env, "CUDA_VISIBLE_DEVICES": selection})
    assert result.returncode == 1
    assert "assigned GPU" in result.stderr or "distinct, available GPU" in result.stderr
    assert not Path(launcher_env["STARTED"]).exists()


@pytest.mark.parametrize("script", ["run_8gpu.sh", "run_8gpu_x100_direct.sh", "run_extractions.sh"])
@pytest.mark.parametrize("use_uuids", [False, True])
def test_direct_launchers_keep_the_assigned_devices(
    tmp_path: Path, launcher_env: dict[str, str], script: str, use_uuids: bool
) -> None:
    results = Path(launcher_env["ROOT"])
    (results / "manifest-uncapped-window50.jsonl").write_text("{}\n")
    indices = [8, 2, 1, 3, 4, 5, 6, 7]
    selectors = [f"GPU-00000000-0000-0000-0000-{gpu:012d}" if use_uuids else str(gpu) for gpu in indices]
    result = _run_launcher(
        tmp_path, script, {**launcher_env, "CUDA_VISIBLE_DEVICES": ",".join(selectors), "LAST_GPU_INDEX": "8"}
    )
    assert result.returncode != 0
    assert "exited before readiness" in result.stderr
    started = set(Path(launcher_env["STARTED"]).read_text().splitlines())
    assert selectors[0] in started
    assert started <= set(selectors)
