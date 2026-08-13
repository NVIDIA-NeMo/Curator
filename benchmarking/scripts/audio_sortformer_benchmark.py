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

"""Benchmark Streaming Sortformer on unique public AMI SDM meetings."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import ManifestReader
from nemo_curator.stages.audio.inference.speaker_diarization.sortformer import InferenceSortformerStage
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.tasks import AudioTask

AMI_SPLITS = ("validation", "test")
AMI_SPLIT_NUM_ROWS = {"validation": 18, "test": 16}
EXPECTED_AUDIO_FILENAMES = tuple(
    f"ami_sdm_{split}_{index:03d}.wav" for split in AMI_SPLITS for index in range(AMI_SPLIT_NUM_ROWS[split])
)
DEFAULT_CHUNK_LEN = 6
DEFAULT_CHUNK_LEFT_CONTEXT = 1
DEFAULT_CHUNK_RIGHT_CONTEXT = 7
DEFAULT_FIFO_LEN = 188
DEFAULT_SPKCACHE_UPDATE_PERIOD = 144
DEFAULT_SPKCACHE_LEN = 188
SORTFORMER_STAGE_NAME = "Sortformer_inference"


def _finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        msg = f"{label} must be a finite number, got {value!r}"
        raise RuntimeError(msg) from e
    if not math.isfinite(number):
        msg = f"{label} must be a finite number, got {value!r}"
        raise RuntimeError(msg)
    return number


def _load_jsonl_rows(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        msg = f"{label} is missing or empty: {path}"
        raise RuntimeError(msg)

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"{label} has invalid JSON on line {line_number}: {e}"
                raise RuntimeError(msg) from e
            if not isinstance(row, Mapping):
                msg = f"{label} line {line_number} is not a JSON object"
                raise TypeError(msg)
            rows.append(dict(row))
    if not rows:
        msg = f"{label} contains no data rows: {path}"
        raise RuntimeError(msg)
    return rows


def _validate_manifest_contract(rows: list[dict[str, Any]], label: str) -> None:
    if len(rows) != len(EXPECTED_AUDIO_FILENAMES):
        msg = f"{label} must contain exactly {len(EXPECTED_AUDIO_FILENAMES)} rows, found {len(rows)}"
        raise RuntimeError(msg)

    expected_filenames = set(EXPECTED_AUDIO_FILENAMES)
    seen_filenames: set[str] = set()
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        audio_filepath = row.get("audio_filepath")
        if not isinstance(audio_filepath, str) or not audio_filepath:
            msg = f"{label} line {line_number} must contain audio_filepath"
            raise RuntimeError(msg)
        filename = Path(audio_filepath).name
        if filename not in expected_filenames:
            msg = f"{label} line {line_number} references unexpected audio file {filename!r}"
            raise RuntimeError(msg)
        if filename in seen_filenames:
            msg = f"{label} contains duplicate audio file {filename!r}"
            raise RuntimeError(msg)
        seen_filenames.add(filename)

        audio_item_id = row.get("audio_item_id")
        if not isinstance(audio_item_id, str) or not audio_item_id:
            msg = f"{label} line {line_number} must contain a nonempty audio_item_id"
            raise RuntimeError(msg)
        if audio_item_id in seen_ids:
            msg = f"{label} contains duplicate audio_item_id {audio_item_id!r}"
            raise RuntimeError(msg)
        seen_ids.add(audio_item_id)

        duration = _finite_float(row.get("duration"), f"{label} line {line_number} duration")
        if duration <= 0:
            msg = f"{label} line {line_number} duration must be positive"
            raise RuntimeError(msg)


def _locate_prestaged_data(data_dir: Path) -> tuple[Path, Path]:
    manifest_path = data_dir / "manifest.jsonl"
    audio_dir = data_dir / "audio"
    if not manifest_path.is_file() or not audio_dir.is_dir():
        msg = f"Expected pre-staged manifest.jsonl and audio/ under {data_dir}"
        raise FileNotFoundError(msg)
    missing_audio = [
        audio_dir / filename for filename in EXPECTED_AUDIO_FILENAMES if not (audio_dir / filename).is_file()
    ]
    if missing_audio:
        msg = f"Pre-staged audio files are missing: {', '.join(str(path) for path in missing_audio)}"
        raise FileNotFoundError(msg)
    _validate_manifest_contract(_load_jsonl_rows(manifest_path, "Input manifest"), "Input manifest")
    return manifest_path, audio_dir


def _write_staged_manifest(source_manifest: Path, target_manifest: Path, audio_dir: Path) -> int:
    rows = _load_jsonl_rows(source_manifest, "Source manifest")
    _validate_manifest_contract(rows, "Source manifest")
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    with target_manifest.open("w", encoding="utf-8") as target_file:
        for row in rows:
            row["audio_filepath"] = str((audio_dir / Path(row["audio_filepath"]).name).resolve())
            target_file.write(json.dumps(row) + "\n")
    return len(rows)


def _validate_segment(segment: object, label: str) -> None:
    if not isinstance(segment, Mapping):
        msg = f"{label} must be a mapping"
        raise TypeError(msg)
    start = _finite_float(segment.get("start"), f"{label} start")
    end = _finite_float(segment.get("end"), f"{label} end")
    if start < 0 or end <= start:
        msg = f"{label} has invalid timestamps"
        raise RuntimeError(msg)
    if not isinstance(segment.get("speaker"), str) or not segment["speaker"]:
        msg = f"{label} must contain a nonempty speaker"
        raise RuntimeError(msg)


def _validate_outputs(tasks: Sequence[AudioTask], num_input_rows: int) -> dict[str, int | float | bool]:
    if len(tasks) != num_input_rows:
        msg = f"Sortformer returned {len(tasks)} rows for {num_input_rows} input rows"
        raise RuntimeError(msg)

    total_duration_s = 0.0
    num_tasks_with_segments = 0
    num_segments = 0
    stage_items = 0
    for task_index, task in enumerate(tasks):
        duration = _finite_float(task.data.get("duration"), f"task {task_index} duration")
        if duration <= 0:
            msg = f"task {task_index} duration must be positive"
            raise RuntimeError(msg)
        total_duration_s += duration

        segments = task.data.get("diar_segments")
        if not isinstance(segments, list):
            msg = f"task {task_index} must contain a diar_segments list"
            raise TypeError(msg)
        for segment_index, segment in enumerate(segments):
            _validate_segment(segment, f"task {task_index} segment {segment_index}")
        num_segments += len(segments)
        num_tasks_with_segments += bool(segments)

        for perf in task._stage_perf:
            if perf.stage_name == SORTFORMER_STAGE_NAME:
                stage_items += perf.num_items_processed

    if num_segments == 0:
        msg = "Sortformer produced no diarization segments"
        raise RuntimeError(msg)
    if stage_items <= 0:
        msg = f"{SORTFORMER_STAGE_NAME} processed no data"
        raise RuntimeError(msg)

    return {
        "num_input_rows": num_input_rows,
        "num_output_rows": len(tasks),
        "input_output_row_count_match": True,
        "num_tasks_processed": len(tasks),
        "num_tasks_with_segments": num_tasks_with_segments,
        "num_segments_processed": num_segments,
        "stage_execution_coverage_ratio": 1.0,
        "total_audio_duration_hours": total_duration_s / 3600,
    }


def run_audio_sortformer_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    scratch_output_path: str,
    raw_data_dir: str,
    model_path: str,
    gpu_stage_num_workers: int = 1,
    chunk_len: int = DEFAULT_CHUNK_LEN,
    chunk_left_context: int = DEFAULT_CHUNK_LEFT_CONTEXT,
    chunk_right_context: int = DEFAULT_CHUNK_RIGHT_CONTEXT,
    fifo_len: int = DEFAULT_FIFO_LEN,
    spkcache_update_period: int = DEFAULT_SPKCACHE_UPDATE_PERIOD,
    spkcache_len: int = DEFAULT_SPKCACHE_LEN,
    rttm_out_dir: str | None = None,
    executor: str = "xenna",
) -> dict[str, Any]:
    """Run Sortformer on pre-staged audio and collect structural and throughput metrics."""
    if gpu_stage_num_workers < 1:
        msg = "gpu_stage_num_workers must be at least 1"
        raise ValueError(msg)

    source_manifest, audio_dir = _locate_prestaged_data(Path(raw_data_dir))
    input_manifest = Path(scratch_output_path) / "audio_sortformer_ami_sdm" / "manifest.jsonl"
    num_input_rows = _write_staged_manifest(source_manifest, input_manifest, audio_dir)
    logger.info(f"Benchmark results path: {benchmark_results_path}")
    local_model = Path(model_path)
    if not local_model.is_file():
        msg = f"Pre-staged Sortformer model not found: {local_model}"
        raise FileNotFoundError(msg)

    exc = setup_executor(executor)
    run_start_time = time.perf_counter()
    pipeline = Pipeline(
        name="audio_sortformer_diarization",
        description="Unique AMI SDM meetings -> Streaming Sortformer diarization",
    )
    pipeline.add_stage(ManifestReader(manifest_path=str(input_manifest)))
    pipeline.add_stage(
        InferenceSortformerStage(
            model_path=str(local_model),
            rttm_out_dir=rttm_out_dir,
            chunk_len=chunk_len,
            chunk_left_context=chunk_left_context,
            chunk_right_context=chunk_right_context,
            fifo_len=fifo_len,
            spkcache_update_period=spkcache_update_period,
            spkcache_len=spkcache_len,
        ).with_(resources=Resources(gpus=1), num_workers=gpu_stage_num_workers)
    )
    logger.info(pipeline.describe())
    results = pipeline.run(exc)
    run_time_taken = time.perf_counter() - run_start_time
    output_metrics = _validate_outputs(results, num_input_rows)
    total_audio_hours = output_metrics["total_audio_duration_hours"]

    logger.success(f"Processed all {num_input_rows} unique AMI meetings")
    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            **output_metrics,
            "real_time_factor": run_time_taken / (total_audio_hours * 3600) if total_audio_hours > 0 else 0,
            "throughput_files_per_sec": num_input_rows / run_time_taken if run_time_taken > 0 else 0,
            "throughput_audio_hours_per_hour": (
                total_audio_hours * 3600 / run_time_taken if run_time_taken > 0 else 0
            ),
        },
        "tasks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio Sortformer benchmark on pre-staged meeting audio")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to write benchmark results")
    parser.add_argument("--scratch-output-path", required=True, help="Path for the rewritten input manifest")
    parser.add_argument("--raw-data-dir", required=True, help="Directory containing manifest.jsonl and audio/")
    parser.add_argument("--model-path", required=True, help="Pre-staged local Sortformer .nemo checkpoint")
    parser.add_argument("--gpu-stage-num-workers", type=int, default=1)
    parser.add_argument("--chunk-len", type=int, default=DEFAULT_CHUNK_LEN)
    parser.add_argument("--chunk-left-context", type=int, default=DEFAULT_CHUNK_LEFT_CONTEXT)
    parser.add_argument("--chunk-right-context", type=int, default=DEFAULT_CHUNK_RIGHT_CONTEXT)
    parser.add_argument("--fifo-len", type=int, default=DEFAULT_FIFO_LEN)
    parser.add_argument("--spkcache-update-period", type=int, default=DEFAULT_SPKCACHE_UPDATE_PERIOD)
    parser.add_argument("--spkcache-len", type=int, default=DEFAULT_SPKCACHE_LEN)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data", "ray_actors"])
    parser.add_argument("--rttm-out-dir", default=None)
    args = parser.parse_args()

    params = vars(args)
    logger.info(f"Audio Sortformer benchmark arguments: {params}")
    result_dict: dict[str, Any] = {"params": params, "metrics": {"is_success": False}, "tasks": []}
    success_code = 1
    try:
        result_dict.update(run_audio_sortformer_benchmark(**params))
        success_code = 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        result_dict["metrics"]["error_message"] = str(e)
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
