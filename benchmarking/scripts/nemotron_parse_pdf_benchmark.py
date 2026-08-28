# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# ruff: noqa: ANN401, E402, PLR0915

"""Nemotron-Parse PDF pipeline benchmarking script.

Reuses the pipeline and argparser from
tutorials/interleaved/nemotron_parse_pdf/main.py with comprehensive
metrics collection.
"""

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from inference_server_utils import InferenceServerBackend, parse_json_object, start_inference_server
from loguru import logger
from utils import setup_executor, write_benchmark_results

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tutorials" / "interleaved" / "nemotron_parse_pdf"))

from main import (
    create_nemotron_parse_pdf_argparser,
    create_nemotron_parse_pdf_pipeline,
)

from nemo_curator.backends.utils import get_available_cpu_gpu_resources
from nemo_curator.tasks.utils import TaskPerfUtils


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _resolve_num_replicas(configured_num_replicas: int | None) -> int:
    num_replicas = int(configured_num_replicas) if configured_num_replicas is not None else _available_gpu_count()
    if num_replicas < 1:
        msg = f"The PDF inference server needs at least one GPU, found {num_replicas}."
        raise RuntimeError(msg)
    return num_replicas


def _resolve_client_workers_per_replica(configured_workers: int) -> int:
    if configured_workers < 1:
        msg = "--inference-server-client-workers-per-replica must be at least 1"
        raise ValueError(msg)
    return configured_workers


def _available_gpu_count() -> int:
    num_gpus = int(get_available_cpu_gpu_resources(init_and_shutdown=True)[1])
    if num_gpus < 1:
        msg = f"Nemotron-Parse inference needs at least one GPU, found {num_gpus}."
        raise RuntimeError(msg)
    return num_gpus


def _server_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    # Keep the server scheduler at its vLLM default unless the benchmark
    # explicitly overrides it through --engine-kwargs. The CLI max_num_seqs
    # setting belongs to the in-process stage.
    engine_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "limit_mm_per_prompt": {"image": 1},
        "enable_prefix_caching": False,
        "disable_hybrid_kv_cache_manager": False,
    }
    if args.enforce_eager:
        engine_kwargs["enforce_eager"] = True
    engine_kwargs.update(parse_json_object(args.engine_kwargs, argument="--engine-kwargs"))
    return engine_kwargs


def _validate_inference_server_backend(backend: str) -> None:
    if backend != "vllm":
        msg = f"--inference-server-type requires --backend=vllm, got {backend!r}."
        raise ValueError(msg)


def _sample_ids_from_table(data: Any) -> set[str]:
    """Pull sample ids from an in-memory Arrow table, if that is what the task carries."""
    if data is None or not hasattr(data, "column"):
        return set()
    try:
        return set(data.column("sample_id").to_pylist())
    except Exception:  # column may be absent for non-interleaved payloads
        return set()


def _sample_ids_from_metadata(task: Any) -> set[str]:
    """Derive sample ids from the manifest entries recorded in task metadata.

    The parquet writer returns a ``FileGroupTask`` whose ``data`` is a list of
    written file paths, so the sample ids only survive in ``_metadata``.
    """
    metadata = getattr(task, "_metadata", None) or {}
    ids: set[str] = set()
    for entry in metadata.get("source_files") or []:
        if not isinstance(entry, str):
            continue
        name = entry
        try:
            record = json.loads(entry)
        except ValueError:
            pass  # plain filename rather than a serialized manifest record
        else:
            if isinstance(record, dict) and record.get("file_name"):
                name = record["file_name"]
        # Mirror PDFPreprocessStage's sample_id convention exactly: strip only the
        # extension, keeping any directory prefix. Using Path().stem here would
        # collapse "a/x.pdf" and "b/x.pdf" onto the same id and undercount.
        ids.add(name.rsplit(".", 1)[0])
    return ids


def _count_unique_pdfs(output_tasks: list) -> int:
    """Count distinct source PDFs represented in the pipeline output.

    Handles both shapes the pipeline emits: a materialized Arrow table carrying
    ``sample_id``, and the default parquet-writer output where the ids are only
    recoverable from ``_metadata['source_files']``.
    """
    unique: set[str] = set()
    for task in output_tasks:
        ids = _sample_ids_from_table(getattr(task, "data", None))
        if not ids:
            ids = _sample_ids_from_metadata(task)
        unique |= ids
    return len(unique)


def _compute_pdf_parse_metrics(
    output_tasks: list,
    run_time_taken: float,
    num_inference_gpus: int,
    inference_stage_parallelism: int,
) -> dict[str, float]:
    """Compute benchmark-level throughput metrics from additive task stats."""
    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")
    metric_prefix = "task_nemotron_parse_inference_custom"

    num_valid_pages = task_metrics.get(f"{metric_prefix}.num_valid_pages_sum", 0.0)
    total_input_tokens = task_metrics.get(f"{metric_prefix}.total_prompt_tokens_sum", 0.0)
    total_output_tokens = task_metrics.get(f"{metric_prefix}.total_output_tokens_sum", 0.0)
    num_request_errors = task_metrics.get(f"{metric_prefix}.num_request_errors_sum", 0.0)
    inference_stage_process_time_sum_s = task_metrics.get("task_nemotron_parse_inference_process_time_sum", 0.0)

    throughput_pages_per_sec = _safe_div(num_valid_pages, run_time_taken)
    throughput_output_tokens_per_sec = _safe_div(total_output_tokens, run_time_taken)
    inference_stage_active_time_s = _safe_div(inference_stage_process_time_sum_s, inference_stage_parallelism)
    inference_stage_gpu_time_s = inference_stage_active_time_s * num_inference_gpus
    return {
        # Surfaced as first-class metrics (not just throughput denominators) so
        # entries can assert on work actually completed rather than on wall-clock
        # rates, which vary with cluster load. Page count is exactly reproducible
        # for a fixed input; token count is not (dynamic batching shifts where the
        # model emits EOS), so it is asserted as a band rather than an exact value.
        "num_pages_processed": num_valid_pages,
        "num_output_tokens": total_output_tokens,
        "num_request_errors": num_request_errors,
        # Stage process time excludes model/server setup. Normalizing the sum
        # across the stage's concurrent workers estimates the active inference
        # wall time for both in-process and HTTP inference. These intermediate
        # values are intentionally not exposed as top-level metrics.
        "inference_stage_pages_per_sec_per_gpu": _safe_div(num_valid_pages, inference_stage_gpu_time_s),
        "inference_stage_input_tokens_per_sec_per_gpu": _safe_div(total_input_tokens, inference_stage_gpu_time_s),
        "inference_stage_output_tokens_per_sec_per_gpu": _safe_div(total_output_tokens, inference_stage_gpu_time_s),
        "throughput_pages_per_sec": throughput_pages_per_sec,
        "throughput_output_tokens_per_sec": throughput_output_tokens_per_sec,
        "throughput_pages_per_sec_per_gpu": _safe_div(throughput_pages_per_sec, num_inference_gpus),
        "throughput_output_tokens_per_sec_per_gpu": _safe_div(throughput_output_tokens_per_sec, num_inference_gpus),
    }


def run_nemotron_parse_pdf_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the Nemotron-Parse PDF benchmark and collect metrics."""
    executor = setup_executor(args.executor)

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_server = None
    inference_server_startup_s = 0.0
    num_replicas = 0
    num_inference_gpus = 0
    inference_stage_parallelism = 0
    server_engine_kwargs: dict[str, Any] | None = None
    server_type: InferenceServerBackend | None = args.inference_server_type

    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"PDF source: zip_base_dir={args.zip_base_dir}, pdf_dir={args.pdf_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {args.model_path}, backend={args.backend}")
    logger.info(f"PDFs per task: {args.pdfs_per_task}, max PDFs: {args.max_pdfs}")

    run_start_time = time.perf_counter()
    success = False
    output_tasks: list = []

    try:
        if server_type is not None:
            _validate_inference_server_backend(args.backend)
            num_replicas = _resolve_num_replicas(args.num_replicas)
            client_workers_per_replica = _resolve_client_workers_per_replica(
                args.inference_server_client_workers_per_replica
            )
            num_inference_gpus = num_replicas
            inference_stage_parallelism = client_workers_per_replica * num_replicas
            model_name = args.model_id or args.model_path
            server_engine_kwargs = _server_engine_kwargs(args)
            logger.info(
                f"Starting {server_type} inference server with {num_replicas} replicas; "
                f"PDF client stage workers={inference_stage_parallelism}"
            )
            server_start = time.perf_counter()
            inference_server = start_inference_server(
                backend=server_type,
                model_id=model_name,
                model_path=args.model_path,
                num_replicas=num_replicas,
                engine_kwargs=server_engine_kwargs,
                model_runtime_env={"uv": {"packages": ["albumentations==2.0.8"]}},
                dynamo_kwargs={"enable_multimodal": True},
                dynamo_router_kwargs={"trust_remote_code": True},
                dynamo_subprocess_env={"DYN_TCP_REQUEST_TIMEOUT": str(int(args.inference_server_request_timeout_s))},
                health_check_timeout_s=args.inference_server_health_timeout_s,
            )
            inference_server_startup_s = time.perf_counter() - server_start
            pipeline = create_nemotron_parse_pdf_pipeline(
                args,
                inference_server_endpoint=inference_server.endpoint,
                inference_server_model_name=model_name,
                inference_server_client_num_workers=inference_stage_parallelism,
            )
            logger.info(
                f"Inference server ready at {inference_server.endpoint} after {inference_server_startup_s:.2f}s"
            )
        else:
            num_inference_gpus = _available_gpu_count()
            inference_stage_parallelism = num_inference_gpus
            pipeline = create_nemotron_parse_pdf_pipeline(args)

        run_start_time = time.perf_counter()
        logger.info("Running Nemotron-Parse PDF pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        num_pdfs_processed = _count_unique_pdfs(output_tasks)
        pdf_parse_metrics = _compute_pdf_parse_metrics(
            output_tasks,
            run_time_taken,
            num_inference_gpus,
            inference_stage_parallelism,
        )

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_pdfs_processed} PDFs")
        logger.success(f"Page throughput: {pdf_parse_metrics['throughput_pages_per_sec']:.2f} pages/s")
        logger.success(
            f"Output token throughput: {pdf_parse_metrics['throughput_output_tokens_per_sec']:.2f} tokens/s"
        )
        logger.success(
            "Inference-stage per-GPU throughput: "
            f"{pdf_parse_metrics['inference_stage_pages_per_sec_per_gpu']:.2f} pages/s/GPU, "
            f"{pdf_parse_metrics['inference_stage_output_tokens_per_sec_per_gpu']:.2f} output tokens/s/GPU, "
            f"{pdf_parse_metrics['inference_stage_input_tokens_per_sec_per_gpu']:.2f} input tokens/s/GPU"
        )
        if not num_pdfs_processed or not pdf_parse_metrics["num_pages_processed"]:
            logger.error("Benchmark produced no PDFs or pages")
        elif pdf_parse_metrics["num_request_errors"]:
            logger.error(f"Inference server request errors: {pdf_parse_metrics['num_request_errors']:.0f}")
        else:
            success = True

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        run_time_taken = time.perf_counter() - run_start_time
        num_pdfs_processed = 0
        # Keep the metric keys stable across success and failure so entry
        # requirements always have a value to compare against.
        pdf_parse_metrics = {
            "num_pages_processed": 0.0,
            "num_output_tokens": 0.0,
            "num_request_errors": 0.0,
            "inference_stage_pages_per_sec_per_gpu": 0.0,
            "inference_stage_input_tokens_per_sec_per_gpu": 0.0,
            "inference_stage_output_tokens_per_sec_per_gpu": 0.0,
            "throughput_pages_per_sec": 0.0,
            "throughput_output_tokens_per_sec": 0.0,
            "throughput_pages_per_sec_per_gpu": 0.0,
            "throughput_output_tokens_per_sec_per_gpu": 0.0,
        }

    finally:
        if inference_server is not None:
            with contextlib.suppress(Exception):
                inference_server.stop()

    return {
        "params": {
            "executor": args.executor,
            "manifest": args.manifest,
            "pdf_dir": args.pdf_dir,
            "zip_base_dir": args.zip_base_dir,
            "output_dir": str(output_dir),
            "benchmark_results_path": str(args.benchmark_results_path),
            "model_path": args.model_path,
            "backend": args.backend,
            "inference_server_type": server_type,
            "num_replicas": num_replicas,
            "num_inference_gpus": num_inference_gpus,
            "inference_server_client_workers_per_replica": (
                args.inference_server_client_workers_per_replica if server_type is not None else None
            ),
            "pdfs_per_task": args.pdfs_per_task,
            "max_pdfs": args.max_pdfs,
            "dpi": args.dpi,
            "max_pages": args.max_pages,
            "inference_batch_size": args.inference_batch_size,
            "max_num_seqs": (
                args.max_num_seqs if server_type is None else (server_engine_kwargs or {}).get("max_num_seqs")
            ),
            "max_tokens": args.max_tokens,
            "enforce_eager": args.enforce_eager,
            "server_engine_kwargs": server_engine_kwargs,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "inference_server_startup_s": inference_server_startup_s,
            "num_pdfs_processed": num_pdfs_processed,
            "num_output_tasks": len(output_tasks),
            "throughput_pdfs_per_sec": num_pdfs_processed / run_time_taken if run_time_taken > 0 else 0,
            **pdf_parse_metrics,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = create_nemotron_parse_pdf_argparser()

    parser.add_argument(
        "--benchmark-results-path",
        type=Path,
        required=True,
        help="Path to write benchmark results",
    )
    parser.add_argument(
        "--executor",
        default="xenna",
        choices=["xenna", "ray_data"],
        help="Executor to use for pipeline execution",
    )
    parser.add_argument(
        "--inference-server-type",
        choices=["ray-serve", "dynamo"],
        default=None,
        help="Run PDF inference through a managed vLLM Ray Serve or Dynamo server; requires --backend=vllm",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=None,
        help="Inference-server replicas; defaults to the GPU count reported by Ray",
    )
    parser.add_argument(
        "--inference-server-client-workers-per-replica",
        type=int,
        default=4,
        help="Parallel HTTP client stage workers per inference-server replica",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Served model name; defaults to --model-path",
    )
    parser.add_argument(
        "--engine-kwargs",
        default=None,
        help="JSON object of additional vLLM engine arguments for the inference server",
    )
    parser.add_argument(
        "--inference-server-health-timeout-s",
        type=int,
        default=900,
        help="Seconds to wait for the inference server to become healthy",
    )
    parser.add_argument(
        "--inference-server-request-timeout-s",
        type=float,
        default=300.0,
        help="Timeout for each page inference request",
    )
    parser.add_argument(
        "--inference-server-max-retries",
        type=int,
        default=3,
        help="Retries after a failed page inference request",
    )
    args = parser.parse_args()

    logger.info("=== Nemotron-Parse PDF Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results: dict[str, Any] = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    try:
        results = run_nemotron_parse_pdf_benchmark(args)
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
