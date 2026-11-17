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

"""Benchmark the Granary metadata tutorial on a staged audio manifest."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import soundfile as sf
from loguru import logger
from omegaconf import OmegaConf
from utils import write_benchmark_results

from nemo_curator.config.run import create_executor_from_yaml, create_pipeline_from_yaml, create_ray_client_from_yaml

_PIPELINE_CONFIG = Path(__file__).resolve().parents[2] / "tutorials/audio/granary_v2/metadata/pipeline.yaml"
_LID_REJECTIONS = {
    "skipped due to disagreement between langID models.",
    "skipped due to missing langID predictions.",
    "skipped due to missing or empty whisper langID prediction.",
}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                msg = f"Invalid JSON at {path}:{line_number}"
                raise ValueError(msg) from error
            if not isinstance(row, dict):
                msg = f"Expected a JSON object at {path}:{line_number}"
                raise ValueError(msg)  # noqa: TRY004 - malformed JSONL is a value error
            rows.append(row)
    if not rows:
        msg = f"Manifest contains no rows: {path}"
        raise ValueError(msg)
    return rows


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _read_input_durations(path: Path) -> dict[str, float]:
    durations = {}
    item_ids = set()
    for row in _read_rows(path):
        if "text" in row or "transcript" in row:
            msg = "Remove text/transcript from the input: recording transcripts are not aligned to VAD segments"
            raise ValueError(msg)
        audio_path = row.get("audio_filepath")
        if not isinstance(audio_path, str) or not Path(audio_path).is_absolute():
            msg = f"Input audio_filepath must be an absolute path: {audio_path!r}"
            raise ValueError(msg)
        if audio_path in durations:
            msg = f"Duplicate input audio_filepath: {audio_path}"
            raise ValueError(msg)
        item_id = row.get("audio_item_id")
        if "audio_item_id" in row and (
            not isinstance(item_id, str)
            or not item_id.strip()
            or item_id in (".", "..")
            or any(character in item_id for character in ("/", "\\", "\0"))
        ):
            msg = f"audio_item_id must be a plain filename: {item_id!r}"
            raise ValueError(msg)
        if item_id is not None:
            if item_id in item_ids:
                msg = f"Duplicate audio_item_id: {item_id}"
                raise ValueError(msg)
            item_ids.add(item_id)
        duration = row.get("duration")
        actual_duration = sf.info(audio_path).duration
        if (
            not _finite_number(duration)
            or duration <= 0
            or actual_duration <= 0
            or not math.isclose(duration, actual_duration, rel_tol=0, abs_tol=0.001)
        ):
            msg = f"Input duration does not match decoded audio: {audio_path}"
            raise ValueError(msg)
        durations[audio_path] = actual_duration
    return durations


def _validate_segment(row: dict[str, Any], input_duration: float, output_dir: Path) -> tuple[str, float, float]:
    original = row["original_file"]
    start, end = row.get("start_ms"), row.get("end_ms")
    if (
        not _finite_number(start)
        or not _finite_number(end)
        or not 0 <= start < end <= input_duration * 1000 + 1
        or not math.isclose(row["duration"], (end - start) / 1000, rel_tol=0, abs_tol=0.001)
    ):
        msg = f"Invalid speech interval for {original}: {start!r}, {end!r}"
        raise ValueError(msg)
    if type(row.get("segment_num")) is not int or row["segment_num"] < 0:
        msg = f"Invalid segment_num for {original}"
        raise ValueError(msg)
    audio_path = row.get("audio_filepath")
    if not isinstance(audio_path, str):
        msg = f"Missing segment audio_filepath for {original}"
        raise ValueError(msg)  # noqa: TRY004 - invalid serialized output
    audio_file = Path(audio_path)
    if (
        not audio_file.is_absolute()
        or audio_file.suffix != ".opus"
        or not audio_file.is_file()
        or audio_file.stat().st_size == 0
        or not audio_file.resolve().is_relative_to(output_dir / "audio")
    ):
        msg = f"Missing or misplaced Opus segment: {audio_path}"
        raise ValueError(msg)
    try:
        audio_info = sf.info(audio_file)
    except RuntimeError as error:
        msg = f"Unreadable Opus segment: {audio_path}"
        raise ValueError(msg) from error
    if (
        audio_info.format != "OGG"
        or audio_info.subtype != "OPUS"
        or audio_info.channels != 1
        or audio_info.samplerate not in (16_000, 48_000)
        or audio_info.frames <= 0
        or not math.isclose(audio_info.duration, row["duration"], rel_tol=0, abs_tol=0.02)
    ):
        msg = f"Opus format, channels, or duration do not match segment metadata: {audio_path}"
        raise ValueError(msg)
    rttm_path = row.get("rttm_filepath")
    rttm_root = output_dir / "rttm"
    if (
        not isinstance(rttm_path, str)
        or Path(rttm_path).is_absolute()
        or not (rttm_root / rttm_path).is_file()
        or not (rttm_root / rttm_path).resolve().is_relative_to(rttm_root)
    ):
        msg = f"Missing or misplaced RTTM file for {original}"
        raise ValueError(msg)
    if (
        not isinstance(row.get("source_lang"), str)
        or (not row.get("_skipme") and row["source_lang"].strip() in ("", "skipped"))
        or not isinstance(row.get("additional_notes"), dict)
        or not isinstance(row.get("sed_events"), list)
        or type(row.get("num_speakers")) is not int
        or row["num_speakers"] < 0
    ):
        msg = f"Invalid language, SED, or speaker metadata for {original}"
        raise ValueError(msg)
    return original, start, end


def _collect_output_metrics(output_path: Path, input_durations: dict[str, float]) -> dict[str, float | int]:
    covered = set()
    empty_parents = set()
    segment_parents = set()
    intervals = set()
    audio_paths = set()
    segment_duration_s = 0.0
    skipped = 0
    language_rejected = 0
    rows = _read_rows(output_path)
    for row in rows:
        original = row.get("original_file", row.get("audio_filepath"))
        if not isinstance(original, str) or original not in input_durations:
            msg = f"Unknown output source: {original!r}"
            raise ValueError(msg)
        covered.add(original)
        if row.get("read_error"):
            msg = f"Audio read error in output for {original}"
            raise ValueError(msg)
        skip_reason = row.get("_skipme", "")
        allowed_skips = {"", "vad_empty"} if row.get("vad_empty") is True else {"", *_LID_REJECTIONS}
        if not isinstance(skip_reason, str) or skip_reason not in allowed_skips:
            msg = f"Unexpected skip reason for {original}: {skip_reason!r}"
            raise ValueError(msg)
        skipped += bool(skip_reason)
        duration = row.get("duration")
        if not _finite_number(duration) or duration <= 0:
            msg = f"Invalid output duration for {original}: {duration!r}"
            raise ValueError(msg)
        if row.get("vad_empty") is True:
            if (
                original in empty_parents
                or row.get("audio_filepath") != original
                or not math.isclose(duration, input_durations[original], rel_tol=0, abs_tol=0.001)
            ):
                msg = f"Invalid or duplicate VAD-empty audit row for {original}"
                raise ValueError(msg)
            empty_parents.add(original)
            continue

        identity = _validate_segment(row, input_durations[original], output_path.parent)
        if identity in intervals or row["audio_filepath"] in audio_paths:
            msg = f"Duplicate speech interval or audio path: {identity}"
            raise ValueError(msg)
        intervals.add(identity)
        audio_paths.add(row["audio_filepath"])
        segment_parents.add(original)
        language_rejected += bool(skip_reason)
        segment_duration_s += duration

    if covered != set(input_durations) or empty_parents & segment_parents:
        msg = "Output parent coverage is incomplete or mixes speech with a VAD-empty audit row"
        raise ValueError(msg)
    return {
        "num_input_recordings": len(input_durations),
        "num_tasks_processed": len(rows),
        "num_speech_segments": len(intervals),
        "num_vad_empty_recordings": len(empty_parents),
        "num_skipped_tasks": skipped,
        "num_language_rejected_segments": language_rejected,
        "input_coverage_ratio": len(covered) / len(input_durations),
        "total_audio_duration_hours": sum(input_durations.values()) / 3600,
        "total_segment_duration_hours": segment_duration_s / 3600,
    }


def _check_fresh_outputs(results_path: Path) -> None:
    # The runner may already have created params.json and an empty scratch directory.
    for name in ("results", "scratch", "metrics.json", "tasks.pkl"):
        path = results_path / name
        if name == "scratch" and path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            continue
        if path.exists() or path.is_symlink():
            msg = f"Benchmark output already exists: {path}; choose a fresh results directory"
            raise FileExistsError(msg)


def run_audio_granary_metadata_benchmark(
    benchmark_results_path: str,
    input_manifest: str,
    model_dir: str,
    executor: str | None = None,
    execution_mode: str | None = None,
) -> dict[str, Any]:
    """Run the tutorial unchanged and validate fan-out coverage and output artifacts."""
    results_path = Path(benchmark_results_path).resolve()
    _check_fresh_outputs(results_path)
    input_durations = _read_input_durations(Path(input_manifest))
    if not Path(model_dir).is_dir():
        msg = f"Model directory does not exist: {model_dir}"
        raise FileNotFoundError(msg)
    cfg = OmegaConf.load(_PIPELINE_CONFIG)
    cfg.manifest_path = str(Path(input_manifest).resolve())
    cfg.output_dir = str(results_path / "results")
    cfg.model_dir = str(Path(model_dir).resolve())
    if executor is not None:
        cfg.backend = executor
    if execution_mode is not None:
        cfg.execution_mode = execution_mode
    pipeline = create_pipeline_from_yaml(cfg)
    executor_obj = create_executor_from_yaml(cfg)
    with create_ray_client_from_yaml(cfg):
        start = time.perf_counter()
        pipeline.run(executor=executor_obj)
        run_time_s = time.perf_counter() - start
    metrics = _collect_output_metrics(Path(cfg.output_dir) / "metadata.jsonl", input_durations)
    return {
        "params": {
            "executor": cfg.backend,
            "execution_mode": cfg.execution_mode if cfg.backend == "xenna" else None,
            "input_manifest": cfg.manifest_path,
            "model_dir": cfg.model_dir,
            "benchmark_results_path": str(results_path),
            "pipeline_config_path": str(_PIPELINE_CONFIG),
            "pipeline_config": OmegaConf.to_container(cfg, resolve=True),
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_s,
            **metrics,
            "throughput_tasks_per_sec": metrics["num_tasks_processed"] / run_time_s if run_time_s else 0.0,
            "throughput_audio_hours_per_hour": (
                metrics["total_audio_duration_hours"] * 3600 / run_time_s if run_time_s else 0.0
            ),
        },
        "tasks": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--executor", choices=["xenna", "ray_data"], default=None)
    parser.add_argument("--execution-mode", choices=["streaming", "batch"], default=None)
    parser.add_argument("--save-tasks", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    save_tasks = vars(args).pop("save_tasks")
    try:
        _check_fresh_outputs(Path(args.benchmark_results_path))
    except FileExistsError as error:
        logger.error(str(error))
        return 1
    results: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        results.update(run_audio_granary_metadata_benchmark(**vars(args)))
    except Exception as error:
        logger.exception("Granary metadata benchmark failed")
        results["metrics"]["error_message"] = str(error)
    finally:
        write_benchmark_results(results, args.benchmark_results_path, save_tasks=save_tasks)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
