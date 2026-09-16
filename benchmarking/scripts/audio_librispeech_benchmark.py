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

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, ManifestReader, PreserveByValueStage
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.audio.metrics.wer import GetPairwiseWerStage
from nemo_curator.stages.text.io.writer import JsonlWriter


def _load_input_audio_paths(manifest_path: Path) -> list[str]:
    audio_paths: list[str] = []
    seen: set[str] = set()
    with manifest_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                audio_path = row["audio_filepath"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                msg = f"Invalid input manifest row {manifest_path}:{line_number}"
                raise RuntimeError(msg) from e
            if not isinstance(audio_path, str) or not audio_path:
                msg = f"Invalid audio_filepath in {manifest_path}:{line_number}"
                raise RuntimeError(msg)
            if audio_path in seen:
                msg = f"Duplicate input audio_filepath: {audio_path}"
                raise RuntimeError(msg)
            seen.add(audio_path)
            audio_paths.append(audio_path)
    if not audio_paths:
        msg = "Input manifest contains no rows"
        raise RuntimeError(msg)
    return audio_paths


def _parse_output_row(line: str, output_path: Path, line_number: int) -> tuple[str, dict[str, Any], float]:
    try:
        row = json.loads(line)
        audio_path = row["audio_filepath"]
        duration_s = float(row["duration"])
        wer_pct = float(row["wer_pct"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        msg = f"Invalid benchmark output {output_path}:{line_number}"
        raise RuntimeError(msg) from e
    if (
        not isinstance(audio_path, str)
        or not audio_path
        or not math.isfinite(duration_s)
        or duration_s <= 0
        or not math.isfinite(wer_pct)
    ):
        msg = f"Invalid benchmark output {output_path}:{line_number}"
        raise RuntimeError(msg)
    return audio_path, row, duration_s


def _finalize_output(
    writer_output_dir: Path,
    output_path: Path,
    input_audio_paths: list[str],
) -> dict[str, float | int]:
    output_rows: dict[str, dict[str, Any]] = {}
    total_duration_s = 0.0
    shards = sorted(writer_output_dir.glob("*.jsonl"))
    if not shards:
        msg = f"LibriSpeech pipeline wrote no temporary JSONL outputs under {writer_output_dir}"
        raise RuntimeError(msg)

    for shard_path in shards:
        with shard_path.open(encoding="utf-8") as output_file:
            for line_number, line in enumerate(output_file, start=1):
                if not line.strip():
                    continue
                audio_path, row, duration_s = _parse_output_row(line, shard_path, line_number)
                if audio_path in output_rows:
                    msg = f"Duplicate output audio_filepath: {audio_path}"
                    raise RuntimeError(msg)
                output_rows[audio_path] = row
                total_duration_s += duration_s

    expected_paths = set(input_audio_paths)
    actual_paths = set(output_rows)
    if actual_paths != expected_paths:
        msg = (
            "LibriSpeech output audio paths do not match input: "
            f"missing={len(expected_paths - actual_paths)}, extra={len(actual_paths - expected_paths)}"
        )
        raise RuntimeError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_output.open("w", encoding="utf-8") as output_file:
        for audio_path in input_audio_paths:
            output_file.write(json.dumps(output_rows[audio_path], ensure_ascii=False) + "\n")
    temporary_output.replace(output_path)
    if sorted(output_path.parent.glob("*.jsonl")) != [output_path]:
        msg = f"Expected only the final LibriSpeech output at {output_path}"
        raise RuntimeError(msg)
    for shard in shards:
        shard.unlink()
    writer_output_dir.rmdir()

    num_tasks = len(output_rows)
    return {
        "num_tasks_processed": num_tasks,
        "total_audio_duration_hours": total_duration_s / 3600,
        "wer_output_coverage_ratio": num_tasks / len(input_audio_paths),
    }


def run_audio_librispeech_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    input_manifest: str,
    model_name: str,
    wer_threshold: float,
    executor: str = "xenna",
    execution_mode: str | None = None,
) -> dict[str, Any]:
    """Run the timed LibriSpeech pipeline and collect output-derived metrics."""
    benchmark_results_path = Path(benchmark_results_path)
    results_dir = benchmark_results_path / "results"
    # Only writer output is temporary; the input manifest remains untouched.
    writer_output_dir = benchmark_results_path / "scratch" / "librispeech_writer_output"
    final_output = results_dir / "librispeech_output.jsonl"
    run_start_time = time.perf_counter()

    try:
        if results_dir.exists() or writer_output_dir.exists():
            msg = f"Benchmark output already exists under {benchmark_results_path}"
            raise ValueError(msg)  # noqa: TRY301

        logger.info("Starting audio LibriSpeech benchmark")
        logger.info(f"Input manifest: {input_manifest}")
        logger.info(f"Executor: {executor}")
        if execution_mode:
            logger.info(f"Execution mode: {execution_mode}")
        logger.info(f"Model: {model_name}")
        logger.info(f"WER threshold: {wer_threshold}")
        input_manifest_path = Path(input_manifest)
        input_audio_paths = _load_input_audio_paths(input_manifest_path)

        pipeline = Pipeline(name="audio_librispeech", description="LibriSpeech ASR, WER, and duration pipeline")
        asr_batch_size = 16
        pipeline.add_stage(ManifestReader(manifest_path=input_manifest))
        pipeline.add_stage(
            ASRStage(
                adapter_target="nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                model_id=model_name,
                audio_filepath_key="audio_filepath",
                batch_size=asr_batch_size,
                fail_on_audio_error=True,
                adapter_kwargs={"use_cuda_graph_decoder": False},
            )
        )
        pipeline.add_stage(
            GetPairwiseWerStage(text_key="text", pred_text_key="pred_text", wer_key="wer_pct").with_(
                batch_size=asr_batch_size
            )
        )
        pipeline.add_stage(
            GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration").with_(
                batch_size=asr_batch_size
            )
        )
        pipeline.add_stage(
            PreserveByValueStage(input_value_key="wer_pct", target_value=wer_threshold, operator="le").with_(
                batch_size=asr_batch_size
            )
        )
        pipeline.add_stage(AudioToDocumentStage())
        pipeline.add_stage(
            JsonlWriter(path=writer_output_dir, write_kwargs={"force_ascii": False}).with_(num_workers=1)
        )

        executor_config = {"execution_mode": execution_mode} if execution_mode else None
        pipeline.run(setup_executor(executor, config=executor_config))
        output_metrics = _finalize_output(writer_output_dir, final_output, input_audio_paths)
        run_time_taken = time.perf_counter() - run_start_time
        success = True
        logger.success(
            f"Processed {output_metrics['num_tasks_processed']} clips / "
            f"{output_metrics['total_audio_duration_hours']:.4f} hours in {run_time_taken:.2f}s"
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
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
        "tasks": [],
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
