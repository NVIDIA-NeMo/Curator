# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Exercise the archived direct launchers without starting real model servers."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TUTORIAL = Path(__file__).parents[2] / "tutorials/interleaved/nemotron_parse_pdf"
TUTORIAL_RELATIVE = "tutorials/interleaved/nemotron_parse_pdf"
DIRECT_RUN = "direct-r16-m45-s96-t8192-p16-w36-c64-nemotron-parse-pdf"

MOCK_PYTHON = """\
import json
import os
import sys
import time
from pathlib import Path

args = sys.argv[1:]
if args[0] in ("-c", "-"):
    sys.exit(0)
root = Path(os.environ["MOCK_ROOT"])
selector = os.environ.get("CUDA_VISIBLE_DEVICES")
if args[0] == "-m":
    with Path(os.environ["STARTED"]).open("a") as started:
        started.write(f"{selector}\\n")
    if not os.environ.get("MOCK_READY"):
        sys.exit(7)
    role = "replica"
    port = int(args[args.index("--port") + 1])
    if port % 2 == 0:
        first_wave = set(range(8101, 8117, 2))
        if not first_wave <= set(map(int, (root / "health").read_text().splitlines())):
            sys.exit(8)
elif Path(args[0]).name == "round_robin_proxy.py":
    role = "proxy"
    port = int(args[args.index("--listen-port") + 1])
else:
    role = "recorder"
    port = None

with (root / "records.jsonl").open("a") as out:
    record = dict(
        role=role, port=port, args=args, selector=selector, pid=os.getpid(), cwd=os.getcwd(),
        start_method=os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
    )
    out.write(json.dumps(record) + "\\n")
if port is not None:
    (root / f"ready-{port}").touch()
    time.sleep(120)
"""


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
        "curl": (
            '#!/bin/bash\nport="${@: -1}"; port="${port%/health}"; port="${port##*:}"\n'
            '[[ -f "$MOCK_ROOT/ready-$port" ]] || exit 22\n'
            'echo "$port" >> "$MOCK_ROOT/health"\n'
        ),
        "python": f"#!{sys.executable}\n{MOCK_PYTHON}",
        "sleep": "#!/bin/bash\nexec /usr/bin/sleep 0.02\n",
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
        "MOCK_ROOT": str(tmp_path),
        "LAST_GPU_INDEX": "7",
    }
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("MOCK_READY", None)
    return environment


def _run_launcher(
    directory: Path, script: str, environment: dict[str, str], *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["bash", str(TUTORIAL / script), *arguments],  # noqa: S607
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


@pytest.mark.parametrize(
    ("script", "arguments", "use_uuids", "expected"),
    [
        ("run_8gpu.sh", (), False, [(DIRECT_RUN, "manifest.jsonl", "nemotron-parse-pdf-direct-r16-w36", "50")]),
        (
            "run_8gpu_x100_direct.sh",
            (),
            True,
            [(f"{DIRECT_RUN}-x100", "manifest.jsonl", "nemotron-parse-pdf-x100-direct-r16-w36", "50")],
        ),
        (
            "run_extractions.sh",
            (),
            True,
            [
                ("extraction/capped50", "manifest-capped50.jsonl", "dc10k-capped50", "50"),
                ("extraction/uncapped", "manifest-uncapped-window50.jsonl", "dc10k-uncapped", "1000000"),
            ],
        ),
        (
            "run_extractions.sh",
            ("capped50",),
            False,
            [("extraction/capped50", "manifest-capped50.jsonl", "dc10k-capped50", "50")],
        ),
        (
            "run_extractions.sh",
            ("uncapped",),
            True,
            [("extraction/uncapped", "manifest-uncapped-window50.jsonl", "dc10k-uncapped", "1000000")],
        ),
    ],
)
def test_direct_launchers_preserve_the_recipe_and_sample_the_assigned_devices(
    launcher_env: dict[str, str],
    script: str,
    arguments: tuple[str, ...],
    use_uuids: bool,
    expected: list[tuple[str, str, str, str]],
) -> None:
    tmp_path = Path(launcher_env["MOCK_ROOT"])
    results = Path(launcher_env["ROOT"])
    (results / "manifest-uncapped-window50.jsonl").write_text("{}\n")
    indices = [8, 2, 1, 3, 4, 5, 6, 7]
    selectors = [f"GPU-00000000-0000-0000-0000-{gpu:012d}" if use_uuids else str(gpu) for gpu in indices]
    result = _run_launcher(
        tmp_path,
        script,
        {**launcher_env, "CUDA_VISIBLE_DEVICES": ",".join(selectors), "LAST_GPU_INDEX": "8", "MOCK_READY": "1"},
        *arguments,
    )
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    replicas = sorted((item for item in records if item["role"] == "replica"), key=lambda item: item["port"])
    assert [item["port"] for item in replicas] == list(range(8101, 8117))
    for replica in replicas:
        port = replica["port"]
        assert replica["selector"] == selectors[(port - 8101) // 2]
        assert replica["start_method"] == "spawn"
        assert replica["args"] == [
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            launcher_env["MODEL"],
            "--served-model-name",
            "nemotron-parse",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--trust-remote-code",
            "--gpu-memory-utilization",
            "0.45",
            "--max-model-len",
            "16384",
            "--max-num-seqs",
            "96",
            "--max-num-batched-tokens",
            "8192",
            "--attention-config",
            '{"flash_attn_version":3}',
            "--no-enable-log-requests",
        ]
    (proxy,) = [item for item in records if item["role"] == "proxy"]
    assert proxy["args"] == [
        launcher_env["PROXY"],
        "--listen-port",
        "8000",
        *(arg for port in range(8101, 8117) for arg in ("--upstream", f"http://127.0.0.1:{port}")),
    ]
    actual = [item for item in records if item["role"] == "recorder"]
    assert len(actual) == len(expected)
    for command, (run, manifest, dataset, max_pages) in zip(actual, expected, strict=True):
        assert Path(command["cwd"]) == TUTORIAL.parents[2]
        assert command["args"] == [
            f"{TUTORIAL_RELATIVE}/benchmark_run.py",
            "--run-dir",
            str(results / run),
            *(arg for gpu in indices for arg in ("--gpu-index", str(gpu))),
            "--",
            launcher_env["PYTHON"],
            f"{TUTORIAL_RELATIVE}/main.py",
            "--manifest",
            str(results / manifest),
            "--pdf-dir",
            launcher_env["PDF_DIR"],
            "--output-dir",
            str(results / run / "output"),
            "--dataset-name",
            dataset,
            "--model-path",
            launcher_env["MODEL"],
            "--inference-server-endpoint",
            "http://127.0.0.1:8000/v1",
            "--inference-server-model-name",
            "nemotron-parse",
            "--parse-mode",
            "single",
            "--inference-batch-size",
            "64",
            "--client-num-workers",
            "36",
            "--pdfs-per-task",
            "16",
            "--dpi",
            "300",
            "--max-pages",
            max_pages,
            "--executor",
            "ray-data",
        ]
    for item in [*replicas, proxy]:
        with pytest.raises(ProcessLookupError):
            os.kill(item["pid"], 0)
