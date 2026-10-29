#!/usr/bin/env python3
"""Run the uncapped DC10K Parse 3.5 pipeline on the best warm Dynamo topology.

Model/Dynamo startup is measured separately.  The one-second GPU sampler and
pages/second clock begin immediately before ``pipeline.run`` and stop when it
returns, so neither metric includes model startup or shutdown.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time


CURATOR_ROOT = Path(__file__).resolve().parents[3]
GPU_INDEXES = tuple(range(8))
FIELDS = ("timestamp_utc", "gpu_index", "gpu_uuid", "utilization_gpu_pct", "memory_used_mib", "power_draw_w")


def ensure_cuda13_python(args: argparse.Namespace) -> None:
    """Select the CUDA 13 venv and actor image without a shell wrapper."""
    desired = (args.python or args.curator_root / ".venv-parse35-cuda13/bin/python").resolve()
    if not desired.is_file():
        raise FileNotFoundError(
            f"Python interpreter not found: {desired}. Pass --python or create "
            "Curator's .venv-parse35-cuda13 environment."
        )
    os.environ["NEMO_CURATOR_DYNAMO_CUDA_TAG"] = args.cuda_tag
    os.environ["PATH"] = str(desired.parent) + os.pathsep + os.environ.get("PATH", "")
    if Path(sys.executable).resolve() != desired:
        os.execve(
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
            raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
        log_path = args.output_root / f"{args.run_name}-eager-{mode}.command.log"
        print(f"Running eager mode {mode}; log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                child_command(mode),
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )


def sample_gpus(stop: threading.Event, destination: Path) -> None:
    query = "index,uuid,utilization.gpu,memory.used,power.draw"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        while True:
            sampled_at = datetime.now(UTC).isoformat()
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                index, uuid, utilization, memory, power = (value.strip() for value in line.split(","))
                if int(index) in GPU_INDEXES:
                    writer.writerow(dict(zip(FIELDS, (sampled_at, index, uuid, utilization, memory, power), strict=True)))
            stream.flush()
            if stop.wait(1.0):
                return


def main() -> None:
    cli = argparse.ArgumentParser(
        description="Run DC10K through Curator's managed Dynamo Parse 3.5 pipeline."
    )
    cli.add_argument("--eager-mode", choices=("on", "off", "both"), default="both")
    cli.add_argument("--curator-root", type=Path, default=CURATOR_ROOT)
    cli.add_argument("--python", type=Path)
    cli.add_argument("--cuda-tag", default="cu130")
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
    ensure_cuda13_python(cli_args)
    if cli_args.eager_mode == "both":
        run_both(cli_args)
        return

    eager_mode = cli_args.eager_mode == "on"
    run_dir = cli_args.output_root / f"{cli_args.run_name}-eager-{cli_args.eager_mode}"
    output_dir = run_dir / "output"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    output_dir.mkdir()

    curator = cli_args.curator_root.resolve()
    sys.path.insert(0, str(curator / "tutorials/interleaved/nemotron_parse_pdf"))
    sys.path.insert(0, str(curator / "benchmarking/scripts"))

    import dataclasses

    import nemo_curator.core.serve.dynamo.backend as dynamo_backend
    import nemo_curator.core.serve.dynamo.vllm as dynamo_vllm
    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.stages.interleaved.pdf.nemotron_parse import create_nemotron_parse_inference_server
    from nemotron_parse_pdf_benchmark import _compute_pdf_parse_metrics, _count_unique_pdfs
    from pipeline_utils import create_nemotron_parse_pdf_argparser, create_nemotron_parse_pdf_pipeline

    # Two schedulers per B200 was the best tested Dynamo active-pipeline
    # topology. Ray must reserve half a physical GPU for each actor.
    original_plan = dynamo_vllm.plan_replica_bundle_shape
    dynamo_vllm.plan_replica_bundle_shape = lambda tp_size, **kwargs: dataclasses.replace(
        original_plan(tp_size, **kwargs), bundles=[{"CPU": 1, "GPU": 0.5}], per_node_gpus=0.5
    )
    dynamo_backend.DynamoBackend._validate_gpu_requirements = lambda *args, **kwargs: None

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
            "--manifest", str(cli_args.manifest.resolve()),
            "--pdf-dir", str(cli_args.pdf_dir.resolve()),
            "--output-dir", str(output_dir),
            "--dataset-name", f"{cli_args.run_name}-eager-{cli_args.eager_mode}",
            "--model-path", str(cli_args.model_path.resolve()),
            "--parse-mode", "single",
            "--inference-batch-size", "4",
            "--client-num-workers", "16",
            "--pdfs-per-task", "16",
            "--dpi", "300",
            "--max-pages", "1000000",
            "--executor", "ray-data",
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
    )

    startup_started = time.perf_counter()
    server.start()
    startup_seconds = time.perf_counter() - startup_started
    pipeline = create_nemotron_parse_pdf_pipeline(
        args,
        inference_server_endpoint=server.endpoint,
        inference_server_model_name="nemotron-parse",
        inference_server_client_num_workers=16,
    )
    executor = RayDataExecutor()

    gpu_csv = run_dir / "gpu-samples.csv"
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpus, args=(stop, gpu_csv), daemon=True)
    started_utc = datetime.now(UTC)
    started = time.perf_counter()
    sampler.start()
    exit_code = 0
    error = None
    output_tasks = []
    try:
        output_tasks = pipeline.run(executor)
    except BaseException as exc:
        exit_code = 1
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        sampler.join()
        finished_utc = datetime.now(UTC)
        server.stop()

        with gpu_csv.open(newline="") as stream:
            samples = list(csv.DictReader(stream))
        per_gpu = {}
        for index in GPU_INDEXES:
            values = [float(row["utilization_gpu_pct"]) for row in samples if int(row["gpu_index"]) == index]
            per_gpu[str(index)] = {
                "samples": len(values),
                "mean_utilization_pct": statistics.fmean(values) if values else None,
                "min_utilization_pct": min(values) if values else None,
                "max_utilization_pct": max(values) if values else None,
            }
        values = [float(row["utilization_gpu_pct"]) for row in samples]
        metrics = (
            _compute_pdf_parse_metrics(output_tasks, elapsed, len(GPU_INDEXES), 16) if output_tasks else {}
        )
        summary = {
            "schema": "curator.parse35.dynamo-pipeline-only.v1",
            "exit_code": exit_code,
            "error": error,
            "started_utc": started_utc.isoformat(),
            "finished_utc": finished_utc.isoformat(),
            "inference_server_startup_seconds_excluded": startup_seconds,
            "pipeline_elapsed_seconds": elapsed,
            "num_pdfs_processed": _count_unique_pdfs(output_tasks) if output_tasks else 0,
            **metrics,
            "sample_interval_seconds": 1,
            "gpu_sample_rows": len(samples),
            "mean_gpu_utilization_pct": statistics.fmean(values) if values else None,
            "per_gpu": per_gpu,
            "topology": topology,
        }
        (run_dir / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
