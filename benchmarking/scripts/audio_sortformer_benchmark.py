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

"""Audio Sortformer diarization benchmarking script.

This script runs Streaming Sortformer diarization benchmarks with
comprehensive metrics collection including real-time factor (RTF),
per-file segment counts, and throughput.
"""

import argparse
import time
import traceback
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import ManifestReader
from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage


def _collect_diarization_metrics(tasks: list, elapsed_s: float) -> dict[str, Any]:
    """Extract diarization-specific metrics from output tasks."""
    num_files = len(tasks) if tasks else 0
    total_audio_duration_s = 0.0
    total_segments = 0

    for task in tasks or []:
        data = task.data if hasattr(task, "data") else {}
        total_audio_duration_s += float(data.get("duration", 0))
        segments = data.get("diar_segments", [])
        total_segments += len(segments)

    throughput = num_files / elapsed_s if elapsed_s > 0 else 0.0
    rtf = elapsed_s / total_audio_duration_s if total_audio_duration_s > 0 else 0.0

    return {
        "is_success": num_files > 0,
        "num_files_processed": num_files,
        "exec_time_s": round(elapsed_s, 2),
        "total_audio_duration_s": round(total_audio_duration_s, 2),
        "total_segments_detected": total_segments,
        "real_time_factor": round(rtf, 4),
        "throughput_files_per_sec": round(throughput, 4),
    }


def run_audio_sortformer_benchmark(
    manifest_path: str,
    model_name: str,
    rttm_out_dir: str | None = None,
    executor: str = "xenna",
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the audio Sortformer diarization benchmark and collect metrics."""
    logger.info("Starting audio Sortformer diarization benchmark")
    logger.info(f"Executor: {executor}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Manifest: {manifest_path}")

    executor_obj = setup_executor(executor)
    pipeline = Pipeline(
        name="audio_sortformer_diarization",
        description="Streaming Sortformer speaker diarization inference.",
    )

    pipeline.add_stage(ManifestReader(manifest_path=manifest_path))
    pipeline.add_stage(
        InferenceSortformerStage(
            model_name=model_name,
            rttm_out_dir=rttm_out_dir,
        ),
    )

    t0 = time.perf_counter()
    results = pipeline.run(executor_obj)
    elapsed_s = time.perf_counter() - t0

    metrics = _collect_diarization_metrics(results, elapsed_s)

    logger.success(
        f"Benchmark completed: {metrics['num_files_processed']} files in {elapsed_s:.1f}s "
        f"(RTF={metrics['real_time_factor']:.3f}, {metrics['throughput_files_per_sec']:.2f} files/sec)"
    )

    return {
        "params": {
            "executor": executor,
            "manifest_path": manifest_path,
            "model_name": model_name,
            "rttm_out_dir": rttm_out_dir,
        },
        "metrics": metrics,
        "tasks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio Sortformer diarization benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--manifest-path", required=True, help="Path to input JSONL manifest")
    parser.add_argument(
        "--model-name",
        default="nvidia/diar_streaming_sortformer_4spk-v2.1",
        help="HF Sortformer model id",
    )
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"], help="Executor to use")
    parser.add_argument("--rttm-out-dir", default=None, help="Optional directory to write RTTM output files")

    args = parser.parse_args()

    logger.info("=== Audio Sortformer Diarization Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    success_code = 1
    result_dict: dict[str, Any] = {
        "params": vars(args),
        "metrics": {
            "is_success": False,
        },
        "tasks": [],
    }
    try:
        result_dict.update(run_audio_sortformer_benchmark(**vars(args)))
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
