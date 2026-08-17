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

"""Benchmark English ASR and WER on a pre-staged LibriSpeech manifest."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, ManifestReader, PreserveByValueStage
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.audio.metrics.wer import GetPairwiseWerStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.writer import JsonlWriter


def _collect_output_metrics(results_dir: Path, num_input_tasks: int) -> dict[str, float | int]:
    num_tasks = 0
    tasks_with_wer = 0
    total_duration_s = 0.0
    for output_path in results_dir.glob("*.jsonl"):
        with output_path.open(encoding="utf-8") as output_file:
            for line_number, line in enumerate(output_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    duration_s = float(row["duration"])
                    wer_pct = float(row["wer_pct"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    msg = f"Invalid benchmark output {output_path}:{line_number}"
                    raise RuntimeError(msg) from e
                if not math.isfinite(duration_s) or duration_s <= 0 or not math.isfinite(wer_pct):
                    msg = f"Invalid benchmark output {output_path}:{line_number}"
                    raise RuntimeError(msg)
                num_tasks += 1
                tasks_with_wer += 1
                total_duration_s += duration_s
    if num_tasks == 0:
        msg = f"LibriSpeech pipeline wrote no JSONL rows under {results_dir}"
        raise RuntimeError(msg)
    return {
        "num_tasks_processed": num_tasks,
        "total_audio_duration_hours": total_duration_s / 3600,
        "wer_output_coverage_ratio": tasks_with_wer / num_input_tasks,
    }


def run_audio_librispeech_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    input_manifest: str,
    model_name: str,
    wer_threshold: float,
    gpus: int,
    executor: str = "xenna",
    execution_mode: str | None = None,
    asr_stage_num_workers: int | None = None,
) -> dict[str, Any]:
    """Run the timed LibriSpeech pipeline and collect output-derived metrics."""
    benchmark_results_path = Path(benchmark_results_path)
    results_dir = benchmark_results_path / "results"
    run_start_time = time.perf_counter()

    try:
        if results_dir.exists():
            msg = f"Result directory {results_dir} already exists."
            raise ValueError(msg)  # noqa: TRY301

        logger.info("Starting audio LibriSpeech benchmark")
        logger.info(f"Input manifest: {input_manifest}")
        logger.info(f"Executor: {executor}")
        if execution_mode:
            logger.info(f"Execution mode: {execution_mode}")
        logger.info(f"Model: {model_name}")
        logger.info(f"WER threshold: {wer_threshold}")
        logger.info(f"GPUs per ASR worker: {gpus}")
        worker_count = asr_stage_num_workers if asr_stage_num_workers is not None else "executor default"
        logger.info(f"ASR stage workers: {worker_count}")
        with Path(input_manifest).open(encoding="utf-8") as input_file:
            num_input_tasks = sum(bool(line.strip()) for line in input_file)
        if num_input_tasks == 0:
            msg = f"Input manifest contains no rows: {input_manifest}"
            raise RuntimeError(msg)  # noqa: TRY301

        pipeline = Pipeline(name="audio_librispeech", description="LibriSpeech ASR, WER, and duration pipeline")
        asr_batch_size = 16
        pipeline.add_stage(
            ManifestReader(manifest_path=input_manifest).with_(
                {
                    "manifest_reader_stage": {
                        "ray_stage_spec": {
                            RayStageSpecKeys.FANOUT_TARGET_ROWS_PER_BLOCK: asr_batch_size,
                        }
                    }
                }
            )
        )
        pipeline.add_stage(
            ASRStage(
                adapter_target="nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                model_id=model_name,
                audio_filepath_key="audio_filepath",
                batch_size=asr_batch_size,
                fail_on_audio_error=True,
                adapter_kwargs={"use_cuda_graph_decoder": False},
            ).with_(resources=Resources(gpus=gpus), num_workers=asr_stage_num_workers)
        )
        pipeline.add_stage(GetPairwiseWerStage(text_key="text", pred_text_key="pred_text", wer_key="wer_pct"))
        pipeline.add_stage(GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration"))
        pipeline.add_stage(PreserveByValueStage(input_value_key="wer_pct", target_value=wer_threshold, operator="le"))
        pipeline.add_stage(AudioToDocumentStage())
        pipeline.add_stage(JsonlWriter(path=results_dir, write_kwargs={"force_ascii": False}))

        executor_config = {"execution_mode": execution_mode} if execution_mode else None
        output_tasks = pipeline.run(setup_executor(executor, config=executor_config))
        run_time_taken = time.perf_counter() - run_start_time
        output_metrics = _collect_output_metrics(results_dir, num_input_tasks)
        success = True
        logger.success(
            f"Processed {output_metrics['num_tasks_processed']} clips / "
            f"{output_metrics['total_audio_duration_hours']:.4f} hours in {run_time_taken:.2f}s"
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        output_metrics = {
            "num_tasks_processed": 0,
            "total_audio_duration_hours": 0.0,
            "wer_output_coverage_ratio": 0.0,
        }
        success = False

    num_tasks_processed = int(output_metrics["num_tasks_processed"])
    return {
        "params": {
            "executor": executor,
            "execution_mode": execution_mode,
            "input_manifest": input_manifest,
            "model_name": model_name,
            "wer_threshold": wer_threshold,
            "gpus": gpus,
            "asr_stage_num_workers": asr_stage_num_workers,
            "benchmark_results_path": str(benchmark_results_path),
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            **output_metrics,
            "throughput_tasks_per_sec": num_tasks_processed / run_time_taken if run_time_taken > 0 else 0,
            "throughput_audio_hours_per_hour": (
                float(output_metrics["total_audio_duration_hours"]) * 3600 / run_time_taken
                if run_time_taken > 0
                else 0
            ),
        },
        "tasks": output_tasks or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--model-name", default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument(
        "--wer-threshold",
        type=float,
        default=1000.0,
        help="Permissive workload-preservation threshold; WER remains measured for every output.",
    )
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"])
    parser.add_argument("--gpus", type=int, choices=[0, 1], default=1, help="GPUs per NeMo ASR worker")
    parser.add_argument("--asr-stage-num-workers", type=int, default=None)
    parser.add_argument("--execution-mode", choices=["streaming", "batch"], default=None)
    args = parser.parse_args()

    results: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        results.update(run_audio_librispeech_benchmark(**vars(args)))
    finally:
        write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
