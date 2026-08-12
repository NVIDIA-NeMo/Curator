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

"""Benchmark low-latency Streaming Sortformer on unique public AMI SDM meetings."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import soundfile as sf
from audio_sortformer_contract import (
    AUDIO_SAMPLE_RATE,
    DATASET_AUDIO_SHA256,
    DATASET_NUM_ROWS,
    EXPECTED_AUDIO_FILENAMES,
    MANIFEST_FILENAME,
    MAX_DATASET_DURATION_S,
    MIN_DATASET_DURATION_S,
    MONO_CHANNELS,
    REFERENCE_ANNOTATIONS_SHA256,
    SOURCE_METADATA_FILENAME,
    TIMESTAMP_TOLERANCE_S,
    reference_annotations_sha256,
    sha256,
    source_metadata,
    validate_model,
    validate_reference_annotations,
)
from loguru import logger
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import ManifestReader
from nemo_curator.stages.audio.inference.speaker_diarization.sortformer import InferenceSortformerStage
from nemo_curator.stages.resources import Resources

DEFAULT_CHUNK_LEN = 6
DEFAULT_CHUNK_LEFT_CONTEXT = 1
DEFAULT_CHUNK_RIGHT_CONTEXT = 7
DEFAULT_FIFO_LEN = 188
DEFAULT_SPKCACHE_UPDATE_PERIOD = 144
DEFAULT_SPKCACHE_LEN = 188


def _load_and_rewrite_manifest(  # noqa: C901, PLR0912, PLR0915
    raw_data_dir: Path, target_manifest: Path
) -> list[dict[str, Any]]:
    source_manifest = raw_data_dir / MANIFEST_FILENAME
    audio_dir = raw_data_dir / "audio"
    metadata_path = raw_data_dir / SOURCE_METADATA_FILENAME
    if not source_manifest.is_file():
        msg = f"Pre-staged Sortformer manifest not found: {source_manifest}"
        raise FileNotFoundError(msg)
    if not audio_dir.is_dir():
        msg = f"Pre-staged Sortformer audio directory not found: {audio_dir}"
        raise FileNotFoundError(msg)
    if not metadata_path.is_file():
        msg = f"Pre-staged Sortformer source metadata not found: {metadata_path}"
        raise FileNotFoundError(msg)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != source_metadata():
        msg = f"Sortformer source metadata does not match the pinned public inputs: {metadata_path}"
        raise RuntimeError(msg)

    rows: list[dict[str, Any]] = []
    seen_audio_files: set[str] = set()
    seen_audio_item_ids: set[str] = set()
    seen_session_names: set[str] = set()
    total_duration_s = 0.0

    with source_manifest.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"Sortformer manifest has invalid JSON on line {line_number}: {e}"
                raise RuntimeError(msg) from e
            if not isinstance(row, dict):
                msg = f"Sortformer manifest line {line_number} is not a JSON object"
                raise TypeError(msg)

            audio_filepath = row.get("audio_filepath")
            if not isinstance(audio_filepath, str) or not audio_filepath:
                msg = f"Sortformer manifest line {line_number} must contain audio_filepath"
                raise RuntimeError(msg)
            audio_basename = Path(audio_filepath).name
            if audio_basename not in EXPECTED_AUDIO_FILENAMES:
                msg = f"Sortformer manifest references unexpected audio file {audio_basename!r}"
                raise RuntimeError(msg)
            if audio_basename in seen_audio_files:
                msg = f"Sortformer manifest contains duplicate audio file {audio_basename!r}"
                raise RuntimeError(msg)
            resolved_audio_path = audio_dir / audio_basename
            if not resolved_audio_path.is_file():
                msg = f"Sortformer manifest audio file not found: {resolved_audio_path}"
                raise FileNotFoundError(msg)
            seen_audio_files.add(audio_basename)

            audio_item_id = row.get("audio_item_id")
            if not isinstance(audio_item_id, str) or not audio_item_id:
                msg = f"Sortformer manifest line {line_number} must contain audio_item_id"
                raise RuntimeError(msg)
            if audio_item_id in seen_audio_item_ids:
                msg = f"Sortformer manifest contains duplicate audio_item_id {audio_item_id!r}"
                raise RuntimeError(msg)
            seen_audio_item_ids.add(audio_item_id)

            session_name = row.get("session_name")
            if not isinstance(session_name, str) or not session_name:
                msg = f"Sortformer manifest line {line_number} must contain session_name"
                raise RuntimeError(msg)
            if session_name in seen_session_names:
                msg = f"Sortformer manifest contains duplicate session_name {session_name!r}"
                raise RuntimeError(msg)
            seen_session_names.add(session_name)

            duration = row.get("duration")
            if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                msg = f"Sortformer manifest line {line_number} must contain a finite positive duration"
                raise RuntimeError(msg)
            total_duration_s += float(duration)
            validate_reference_annotations(row, float(duration), f"Sortformer manifest line {line_number}")

            audio_info = sf.info(resolved_audio_path)
            if audio_info.samplerate != AUDIO_SAMPLE_RATE or audio_info.channels != MONO_CHANNELS:
                msg = (
                    f"Expected mono 16 kHz WAV, found {audio_info.channels} channels at "
                    f"{audio_info.samplerate} Hz: {resolved_audio_path}"
                )
                raise RuntimeError(msg)
            measured_duration_s = float(audio_info.frames) / audio_info.samplerate
            if not math.isclose(measured_duration_s, float(duration), abs_tol=TIMESTAMP_TOLERANCE_S):
                msg = f"Sortformer manifest duration does not match staged WAV: {resolved_audio_path}"
                raise RuntimeError(msg)
            expected_sha256 = DATASET_AUDIO_SHA256[audio_basename]
            actual_sha256 = sha256(resolved_audio_path)
            if not isinstance(expected_sha256, str) or actual_sha256 != expected_sha256:
                msg = f"Sortformer staged audio SHA-256 mismatch: {resolved_audio_path}"
                raise RuntimeError(msg)

            rewritten_row = dict(row)
            rewritten_row["audio_filepath"] = str(resolved_audio_path.resolve())
            rows.append(rewritten_row)

    if len(rows) != DATASET_NUM_ROWS:
        msg = f"Sortformer manifest must contain exactly {DATASET_NUM_ROWS} rows, found {len(rows)}"
        raise RuntimeError(msg)
    if seen_audio_files != set(EXPECTED_AUDIO_FILENAMES):
        msg = "Sortformer manifest does not contain the complete pinned AMI SDM validation/test workload"
        raise RuntimeError(msg)
    if not MIN_DATASET_DURATION_S <= total_duration_s <= MAX_DATASET_DURATION_S:
        msg = (
            f"Sortformer source duration {total_duration_s:.3f}s is outside the expected public AMI SDM "
            f"range [{MIN_DATASET_DURATION_S:.0f}, {MAX_DATASET_DURATION_S:.0f}]s"
        )
        raise RuntimeError(msg)
    actual_reference_sha256 = reference_annotations_sha256(rows)
    if actual_reference_sha256 != REFERENCE_ANNOTATIONS_SHA256:
        msg = (
            "Sortformer reference annotation SHA-256 mismatch: "
            f"expected {REFERENCE_ANNOTATIONS_SHA256}, found {actual_reference_sha256}"
        )
        raise RuntimeError(msg)

    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    with target_manifest.open("w", encoding="utf-8") as target_file:
        for row in rows:
            target_file.write(json.dumps(row) + "\n")
    return rows


def _is_valid_segment(segment: object, duration_s: float) -> bool:
    if not isinstance(segment, dict):
        return False
    start = segment.get("start")
    end = segment.get("end")
    speaker = segment.get("speaker")
    return (
        isinstance(start, (int, float))
        and math.isfinite(start)
        and isinstance(end, (int, float))
        and math.isfinite(end)
        and 0 <= start < end <= duration_s + TIMESTAMP_TOLERANCE_S
        and isinstance(speaker, str)
        and bool(speaker)
    )


def _add_segments_to_annotation(
    annotation: Annotation,
    starts: list,
    ends: list,
    speakers: list,
) -> None:
    for track_index, (start, end, speaker) in enumerate(zip(starts, ends, speakers, strict=True)):
        annotation[Segment(float(start), float(end)), track_index] = str(speaker)


def _full_duration_uem(duration_s: float) -> Segment:
    return Segment(0.0, duration_s)


def _collect_diarization_metrics(
    tasks: list,
    elapsed_s: float,
    num_input_files: int,
    source_num_files: int,
    source_audio_duration_s: float,
) -> dict[str, Any]:
    num_output_files = len(tasks) if tasks else 0
    processed_audio_duration_s = 0.0
    total_segments = 0
    malformed_segments = 0
    num_files_with_segments = 0
    output_session_names: list[str] = []
    scored_audio_item_ids: set[str] = set()
    der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

    for task in tasks or []:
        data = task.data if hasattr(task, "data") else {}
        duration = data.get("duration")
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0:
            processed_audio_duration_s += float(duration)

        session_name = data.get("session_name")
        if isinstance(session_name, str) and session_name:
            output_session_names.append(session_name)
        audio_item_id = data.get("audio_item_id")

        segments = data.get("diar_segments", [])
        if not isinstance(segments, list):
            malformed_segments += 1
            continue
        valid_segments = sum(_is_valid_segment(segment, float(duration or 0)) for segment in segments)
        malformed_segments += len(segments) - valid_segments
        total_segments += valid_segments
        if valid_segments > 0:
            num_files_with_segments += 1
        if (
            valid_segments == len(segments)
            and valid_segments > 0
            and isinstance(session_name, str)
            and isinstance(audio_item_id, str)
            and audio_item_id
            and audio_item_id not in scored_audio_item_ids
        ):
            reference = Annotation(uri=session_name)
            hypothesis = Annotation(uri=session_name)
            _add_segments_to_annotation(
                reference,
                data.get("timestamps_start", []),
                data.get("timestamps_end", []),
                data.get("speakers", []),
            )
            _add_segments_to_annotation(
                hypothesis,
                [segment["start"] for segment in segments],
                [segment["end"] for segment in segments],
                [segment["speaker"] for segment in segments],
            )
            der_metric(reference, hypothesis, uem=_full_duration_uem(float(duration)), uri=session_name)
            scored_audio_item_ids.add(audio_item_id)

    expected_audio_duration_s = source_audio_duration_s
    row_count_match = num_output_files == num_input_files
    duration_match = math.isclose(processed_audio_duration_s, expected_audio_duration_s, rel_tol=1e-7, abs_tol=1e-3)
    session_names_unique = len(output_session_names) == num_output_files == len(set(output_session_names))
    segment_coverage_ratio = num_files_with_segments / num_output_files if num_output_files else 0.0
    throughput_files_per_sec = num_output_files / elapsed_s if elapsed_s > 0 else 0.0
    real_time_factor = elapsed_s / processed_audio_duration_s if processed_audio_duration_s > 0 else 0.0
    throughput_audio_hours_per_hour = processed_audio_duration_s / elapsed_s if elapsed_s > 0 else 0.0
    diarization_error_rate_percent = 100.0 * abs(der_metric) if num_files_with_segments else math.inf
    der_components = der_metric[:]
    is_success = (
        row_count_match
        and duration_match
        and session_names_unique
        and malformed_segments == 0
        and num_files_with_segments == num_output_files
        and len(scored_audio_item_ids) == source_num_files
        and num_output_files > 0
    )

    return {
        "is_success": is_success,
        "source_num_files": source_num_files,
        "num_input_files": num_input_files,
        "num_files_processed": num_output_files,
        "input_output_row_count_match": row_count_match,
        "output_session_names_unique": session_names_unique,
        "num_files_with_segments": num_files_with_segments,
        "num_source_files_scored_for_der": len(scored_audio_item_ids),
        "segment_output_coverage_ratio": round(segment_coverage_ratio, 6),
        "total_segments_detected": total_segments,
        "malformed_segments": malformed_segments,
        "diarization_error_rate_percent": round(diarization_error_rate_percent, 4),
        "diarization_confusion_s": round(float(der_components["confusion"]), 4),
        "diarization_missed_detection_s": round(float(der_components["missed detection"]), 4),
        "diarization_false_alarm_s": round(float(der_components["false alarm"]), 4),
        "exec_time_s": round(elapsed_s, 2),
        "source_audio_duration_hours": round(source_audio_duration_s / 3600, 6),
        "total_audio_duration_hours": round(expected_audio_duration_s / 3600, 6),
        "processed_audio_duration_hours": round(processed_audio_duration_s / 3600, 6),
        "input_output_duration_match": duration_match,
        "real_time_factor": round(real_time_factor, 6),
        "throughput_files_per_sec": round(throughput_files_per_sec, 4),
        "throughput_audio_hours_per_hour": round(throughput_audio_hours_per_hour, 3),
    }


def run_audio_sortformer_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
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
    """Run Sortformer once per pinned AMI meeting and collect correctness and throughput metrics."""
    if gpu_stage_num_workers < 1:
        msg = "gpu_stage_num_workers must be at least 1"
        raise ValueError(msg)
    streaming_profile = {
        "chunk_len": chunk_len,
        "chunk_left_context": chunk_left_context,
        "chunk_right_context": chunk_right_context,
        "fifo_len": fifo_len,
        "spkcache_update_period": spkcache_update_period,
        "spkcache_len": spkcache_len,
    }
    for name, value in streaming_profile.items():
        minimum = 0 if name == "chunk_left_context" else 1
        if value < minimum:
            msg = f"{name} must be at least {minimum}"
            raise ValueError(msg)

    raw_data_path = Path(raw_data_dir).resolve()
    resolved_model_path = Path(model_path).resolve()
    validate_model(resolved_model_path)

    results_path = Path(benchmark_results_path).resolve()
    input_manifest_path = results_path / "sortformer_input_manifest.jsonl"
    source_rows = _load_and_rewrite_manifest(raw_data_path, input_manifest_path)
    source_audio_duration_s = sum(float(row["duration"]) for row in source_rows)
    num_input_files = len(source_rows)

    logger.info("Starting audio Sortformer diarization benchmark")
    logger.info(f"Executor: {executor}")
    logger.info(f"Unique public source rows: {len(source_rows)}")
    logger.info(f"Source audio hours: {source_audio_duration_s / 3600:.3f}")
    logger.info(f"GPU workers: {gpu_stage_num_workers}")
    logger.info(f"Streaming profile: {streaming_profile}")
    logger.info(f"Model: {resolved_model_path}")
    logger.info(f"Manifest: {input_manifest_path}")

    pipeline = _build_pipeline(
        manifest_path=input_manifest_path,
        model_path=resolved_model_path,
        gpu_stage_num_workers=gpu_stage_num_workers,
        chunk_len=chunk_len,
        chunk_left_context=chunk_left_context,
        chunk_right_context=chunk_right_context,
        fifo_len=fifo_len,
        spkcache_update_period=spkcache_update_period,
        spkcache_len=spkcache_len,
        rttm_out_dir=rttm_out_dir,
    )

    executor_obj = setup_executor(executor)
    start_time = time.perf_counter()
    results = pipeline.run(executor_obj)
    elapsed_s = time.perf_counter() - start_time
    metrics = _collect_diarization_metrics(
        results,
        elapsed_s,
        num_input_files=num_input_files,
        source_num_files=len(source_rows),
        source_audio_duration_s=source_audio_duration_s,
    )

    logger.success(
        f"Benchmark completed: {metrics['num_files_processed']} files in {elapsed_s:.1f}s "
        f"({metrics['throughput_audio_hours_per_hour']:.1f} audio-hours/hour)"
    )
    return {
        "params": {
            "benchmark_results_path": str(results_path),
            "raw_data_dir": str(raw_data_path),
            "model_path": str(resolved_model_path),
            "gpu_stage_num_workers": gpu_stage_num_workers,
            **streaming_profile,
            "rttm_out_dir": rttm_out_dir,
            "executor": executor,
        },
        "metrics": metrics,
        "tasks": results,
    }


def _build_pipeline(  # noqa: PLR0913
    manifest_path: Path,
    model_path: Path,
    gpu_stage_num_workers: int,
    chunk_len: int,
    chunk_left_context: int,
    chunk_right_context: int,
    fifo_len: int,
    spkcache_update_period: int,
    spkcache_len: int,
    rttm_out_dir: str | None,
) -> Pipeline:
    pipeline = Pipeline(
        name="audio_sortformer_diarization",
        description="Unique public AMI SDM meetings -> low-latency Streaming Sortformer diarization.",
    )
    pipeline.add_stage(ManifestReader(manifest_path=str(manifest_path)))
    pipeline.add_stage(
        InferenceSortformerStage(
            model_path=str(model_path),
            rttm_out_dir=rttm_out_dir,
            chunk_len=chunk_len,
            chunk_left_context=chunk_left_context,
            chunk_right_context=chunk_right_context,
            fifo_len=fifo_len,
            spkcache_update_period=spkcache_update_period,
            spkcache_len=spkcache_len,
        ).with_(resources=Resources(gpus=1), num_workers=gpu_stage_num_workers)
    )
    return pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audio Sortformer benchmark on unique public, pre-staged AMI SDM audio"
    )
    parser.add_argument("--benchmark-results-path", required=True, help="Path for benchmark output artifacts")
    parser.add_argument("--raw-data-dir", required=True, help="Directory containing manifest.jsonl and audio/")
    parser.add_argument("--model-path", required=True, help="Path to the pre-staged Sortformer .nemo checkpoint")
    parser.add_argument(
        "--gpu-stage-num-workers",
        type=int,
        default=1,
        help="Exact number of one-GPU Sortformer workers",
    )
    parser.add_argument("--chunk-len", type=int, default=DEFAULT_CHUNK_LEN)
    parser.add_argument("--chunk-left-context", type=int, default=DEFAULT_CHUNK_LEFT_CONTEXT)
    parser.add_argument("--chunk-right-context", type=int, default=DEFAULT_CHUNK_RIGHT_CONTEXT)
    parser.add_argument("--fifo-len", type=int, default=DEFAULT_FIFO_LEN)
    parser.add_argument("--spkcache-update-period", type=int, default=DEFAULT_SPKCACHE_UPDATE_PERIOD)
    parser.add_argument("--spkcache-len", type=int, default=DEFAULT_SPKCACHE_LEN)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"], help="Executor to use")
    parser.add_argument("--rttm-out-dir", default=None, help="Optional directory for one RTTM file per input")
    args = parser.parse_args()

    logger.info("=== Audio Sortformer Diarization Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    success_code = 1
    result_dict: dict[str, Any] = {
        "params": vars(args),
        "metrics": {"is_success": False},
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
