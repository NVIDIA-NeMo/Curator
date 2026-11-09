#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run the uncapped DC10K Parse 3.5 pipeline on the best warm Dynamo topology.

Model/Dynamo startup is measured separately.  The one-second GPU sampler and
pages/second clock begin immediately before ``pipeline.run`` and stop when it
returns, so neither metric includes model startup or shutdown.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark_run import GPUSampler, summarize_gpus

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.core.serve.server import InferenceServer
    from nemo_curator.pipeline import Pipeline

CURATOR_ROOT = Path(__file__).resolve().parents[3]
GPU_INDEXES = tuple(range(8))


def ensure_cuda13_python(args: argparse.Namespace) -> None:
    """Select the CUDA 13 venv and actor image without a shell wrapper."""
    desired = Path(os.path.abspath(args.python or args.curator_root / ".venv-parse35-cuda13/bin/python"))
    if not desired.is_file():
        msg = f"Python interpreter not found: {desired}. Pass --python or create Curator's .venv-parse35-cuda13 environment."
        raise FileNotFoundError(msg)
    os.environ["NEMO_CURATOR_DYNAMO_CUDA_TAG"] = args.cuda_tag
    os.environ["PATH"] = str(desired.parent) + os.pathsep + os.environ.get("PATH", "")
    if Path(os.path.abspath(sys.executable)) != desired:
        os.execve(  # noqa: S606
            str(desired),
            [str(desired), str(Path(__file__).resolve()), *sys.argv[1:]],
            os.environ,
        )


def child_command(mode: str) -> list[str]:
    """Recreate this command with one concrete eager-mode value."""
    forwarded = []
    skip_next = False
    for value in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if value == "--eager-mode":
            skip_next = True
            continue
        if value.startswith("--eager-mode="):
            continue
        forwarded.append(value)
    return [sys.executable, str(Path(__file__).resolve()), *forwarded, "--eager-mode", mode]


def run_both(args: argparse.Namespace) -> None:
    """Run off and on in isolated processes, matching the former wrapper."""
    args.output_root.mkdir(parents=True, exist_ok=True)
    for mode in ("off", "on"):
        run_dir = args.output_root / f"{args.run_name}-eager-{mode}"
        if run_dir.exists():
            msg = f"Refusing to overwrite existing run: {run_dir}"
            raise FileExistsError(msg)
        log_path = args.output_root / f"{args.run_name}-eager-{mode}.command.log"
        print(f"Running eager mode {mode}; log: {log_path}", flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(  # noqa: S603
                child_command(mode),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            try:
                if process.wait() != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


def execute_run(  # noqa: PLR0913
    server: InferenceServer,
    pipeline_factory: Callable[[], Pipeline],
    executor: RayDataExecutor,
    *,
    run_dir: Path,
    topology: dict,
    compute_metrics: Callable,
    count_pdfs: Callable,
) -> None:
    """Measure the pipeline and clean up even if startup or construction fails."""
    gpu_csv = run_dir / "gpu-samples.csv"
    sampler = GPUSampler(gpu_csv, set(GPU_INDEXES))
    started_utc = finished_utc = None
    started = None
    elapsed = startup_seconds = 0.0
    exit_code = 1
    error = None
    output_tasks = []
    cleanup_errors = []
    try:
        startup_started = time.perf_counter()
        server.start()
        startup_seconds = time.perf_counter() - startup_started
        pipeline = pipeline_factory()
        started_utc = datetime.now(UTC)
        started = time.perf_counter()
        sampler.start()
        output_tasks = pipeline.run(executor)
        exit_code = 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if started is not None:
            elapsed = time.perf_counter() - started
            finished_utc = datetime.now(UTC)
        try:
            sampler.stop()
        except RuntimeError as exc:
            cleanup_errors.append(str(exc))
        try:
            # A signal during start() can leave a backend before _started becomes true.
            if not server._started and server._backend_impl is not None:
                server._backend_impl.stop()
            server.stop()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"Server cleanup failed: {exc}")
        if cleanup_errors:
            exit_code = 1
        summary = {
            "schema": "curator.parse35.dynamo-pipeline-only.v1",
            "exit_code": exit_code,
            "error": error,
            "cleanup_errors": cleanup_errors,
            "started_utc": started_utc.isoformat() if started_utc else None,
            "finished_utc": finished_utc.isoformat() if finished_utc else None,
            "inference_server_startup_seconds_excluded": startup_seconds,
            "pipeline_elapsed_seconds": elapsed,
            "num_pdfs_processed": count_pdfs(output_tasks) if output_tasks else 0,
            **(compute_metrics(output_tasks, elapsed, len(GPU_INDEXES), 16) if output_tasks else {}),
            **summarize_gpus(gpu_csv, set(GPU_INDEXES), started_utc=started_utc, finished_utc=finished_utc),
            "topology": topology,
        }
        (run_dir / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if cleanup_errors and error is None:
            raise RuntimeError("; ".join(cleanup_errors))


def main() -> None:
    cli = argparse.ArgumentParser(description="Run DC10K through Curator's managed Dynamo Parse 3.5 pipeline.")
    cli.add_argument("--eager-mode", choices=("on", "off", "both"), default="both")
    cli.add_argument("--curator-root", type=Path, default=CURATOR_ROOT)
    cli.add_argument("--python", type=Path)
    cli.add_argument("--cuda-tag", choices=("cu129", "cu130"), default="cu130")
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--pdf-dir", type=Path, required=True)
    cli.add_argument("--model-path", type=Path, required=True)
    cli.add_argument("--output-root", type=Path, required=True)
    cli.add_argument(
        "--run-name",
        default="uncapped-dynamo",
        help="Output prefix; choose a new value for each reproduction.",
    )
    cli_args = cli.parse_args()
    if Path(cli_args.run_name).name != cli_args.run_name or cli_args.run_name in {".", ".."}:
        cli.error("--run-name must be a directory name without path separators")
    for path in (cli_args.manifest, cli_args.pdf_dir, cli_args.model_path):
        if not path.exists():
            cli.error(f"Input path does not exist: {path}")
    ensure_cuda13_python(cli_args)
    if cli_args.eager_mode == "both":
        run_both(cli_args)
        return

    eager_mode = cli_args.eager_mode == "on"
    run_dir = cli_args.output_root / f"{cli_args.run_name}-eager-{cli_args.eager_mode}"
    output_dir = run_dir / "output"
    if run_dir.exists():
        msg = f"Refusing to overwrite existing run: {run_dir}"
        raise FileExistsError(msg)
    run_dir.mkdir(parents=True)
    output_dir.mkdir()

    curator = cli_args.curator_root.resolve()
    sys.path.insert(0, str(curator))
    sys.path.insert(0, str(curator / "tutorials/interleaved/nemotron_parse_pdf"))
    sys.path.insert(0, str(curator / "benchmarking/scripts"))

    from nemotron_parse_pdf_benchmark import _compute_pdf_parse_metrics, _count_unique_pdfs
    from pipeline_utils import create_nemotron_parse_pdf_argparser, create_nemotron_parse_pdf_pipeline

    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.stages.interleaved.pdf.nemotron_parse import create_nemotron_parse_inference_server

    engine_kwargs = {
        "distributed_executor_backend": "uni",
        "tensor_parallel_size": 1,
        "max_model_len": 16384,
        "max_num_seqs": 96,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.45,
        # Fixed KV capacity leaves graph-capture headroom for two replicas/GPU.
        "kv_cache_memory_bytes": 8 * 1024**3,
        "attention_config": {"flash_attn_version": 3},
        "enforce_eager": eager_mode,
    }
    topology = {
        "physical_gpus": 8,
        "replicas": 16,
        "replicas_per_gpu": 2,
        "client_workers": 16,
        "inference_batch_size": 4,
        "pdfs_per_task": 16,
        "dpi": 300,
        "eager_mode": eager_mode,
        "engine_kwargs": engine_kwargs,
    }
    (run_dir / "topology.json").write_text(json.dumps(topology, indent=2) + "\n", encoding="utf-8")

    parser = create_nemotron_parse_pdf_argparser()
    args = parser.parse_args(
        [
            "--manifest",
            str(cli_args.manifest.resolve()),
            "--pdf-dir",
            str(cli_args.pdf_dir.resolve()),
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            f"{cli_args.run_name}-eager-{cli_args.eager_mode}",
            "--model-path",
            str(cli_args.model_path.resolve()),
            "--parse-mode",
            "single",
            "--inference-batch-size",
            "4",
            "--client-num-workers",
            "16",
            "--pdfs-per-task",
            "16",
            "--dpi",
            "300",
            "--max-pages",
            "1000000",
        ]
    )

    server = create_nemotron_parse_inference_server(
        model_path=str(cli_args.model_path.resolve()),
        model_name="nemotron-parse",
        backend="dynamo",
        num_replicas=16,
        engine_kwargs=engine_kwargs,
        request_timeout_s=300.0,
        health_check_timeout_s=1800,
        nemotron_parse_path=str(cli_args.model_path.resolve()),
    )

    server.models = [replace(model, gpu_fraction=0.5) for model in server.models]
    server.name = f"parse35-b200-{cli_args.run_name}-{cli_args.eager_mode}"

    def pipeline_factory() -> Pipeline:
        return create_nemotron_parse_pdf_pipeline(
            args,
            inference_server_endpoint=server.endpoint,
            inference_server_model_name="nemotron-parse",
            inference_server_client_num_workers=16,
        )

    execute_run(
        server,
        pipeline_factory,
        RayDataExecutor(),
        run_dir=run_dir,
        topology=topology,
        compute_metrics=_compute_pdf_parse_metrics,
        count_pdfs=_count_unique_pdfs,
    )


if __name__ == "__main__":

    def interrupted(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    main()
