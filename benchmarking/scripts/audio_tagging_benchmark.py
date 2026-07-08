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

"""Audio tagging pipeline benchmarking script.

Runs the full TTS audio tagging pipeline end-to-end:
    Manifest or FLEURS input -> Resample -> Diarize -> Split -> ASR Align ->
    Join -> Merge -> Bandwidth -> Squim -> PrepareModuleSegments ->
    Second-pass ASR -> Compute WER -> Write

Exercises the tagging pipeline stages for regression tracking.
"""

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from utils import RepeatEntriesStage, setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestReader, ManifestWriterStage
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.speaker_diarization.pyannote import PyAnnoteDiarizationStage
from nemo_curator.stages.audio.metrics.bandwidth import BandwidthEstimationStage
from nemo_curator.stages.audio.metrics.squim import TorchSquimQualityMetricsStage
from nemo_curator.stages.audio.metrics.wer import ComputeWERStage
from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import NeMoASRAlignerStage
from nemo_curator.stages.audio.tagging.merge_alignment_diarization import MergeAlignmentDiarizationStage
from nemo_curator.stages.audio.tagging.prepare_module_segments import PrepareModuleSegmentsStage
from nemo_curator.stages.audio.tagging.resample_audio import ResampleAudioStage
from nemo_curator.stages.audio.tagging.split import JoinSplitAudioMetadataStage, SplitLongAudioStage
from nemo_curator.stages.audio.tagging.utils import validate_tagging_outputs
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_REDACTED_VALUE = "<redacted>"
_SENSITIVE_KEY_PARTS = ("token", "secret", "password", "api_key", "apikey")
_REQUIRED_STAGE_NAMES = (
    "ResampleAudio",
    "PyAnnoteDiarization",
    "SplitLongAudio",
    "ASRAlignment",
    "JoinSplitMetadata",
    "MergeAlignmentDiar",
    "BandwidthEstimation",
    "SquimMetrics",
    "PrepareModuleSegments",
    "ASRAlignment2",
    "ComputeWER",
    "ManifestWriter",
)


def _sanitize_sensitive_params(value: object, key: str = "") -> object:
    """Redact secret-like values before logging or persisting benchmark params."""
    normalized_key = key.lower().replace("-", "_")
    if normalized_key and any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
        return _REDACTED_VALUE
    if isinstance(value, dict):
        return {
            item_key: _sanitize_sensitive_params(item_value, str(item_key)) for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_sensitive_params(item) for item in value]
    return value


def _resolve_hf_token() -> str:
    """Read the PyAnnote credential from the process environment."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_SECRET_KEY")
    if not token:
        msg = "Set HF_TOKEN to a Hugging Face token with access to the required PyAnnote models"
        raise RuntimeError(msg)
    return token


def _redact_known_secrets(message: str, secrets: list[str]) -> str:
    """Remove known runtime credentials from an error message."""
    sanitized = message
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, _REDACTED_VALUE)
    return sanitized


def _add_input_stage(  # noqa: PLR0913
    pipeline: Pipeline,
    input_manifest: str | None,
    raw_data_dir: str | None,
    lang: str,
    split: str,
    auto_download: bool,
    cache_dir: str | None,
) -> None:
    """Add exactly one validated manifest or FLEURS input source."""
    if bool(input_manifest) == bool(raw_data_dir):
        msg = "Specify exactly one of input_manifest or raw_data_dir"
        raise ValueError(msg)

    if input_manifest:
        manifest_path = Path(input_manifest)
        if not manifest_path.is_file():
            msg = f"Input manifest does not exist: {manifest_path}"
            raise FileNotFoundError(msg)
        if manifest_path.stat().st_size == 0:
            msg = f"Input manifest is empty: {manifest_path}"
            raise ValueError(msg)
        logger.info(f"Input manifest: {manifest_path}")
        pipeline.add_stage(ManifestReader(manifest_path=str(manifest_path)))
        return

    fleurs_stage = CreateInitialManifestFleursStage(
        lang=lang,
        split=split,
        raw_data_dir=str(raw_data_dir),
        cache_dir=cache_dir,
        auto_download=auto_download,
        batch_size=4,
    )
    if not auto_download:
        fleurs_stage.locate_prestaged_files(fleurs_stage.language_data_dir())
    logger.info(f"FLEURS input: lang={lang}, split={split}, raw_data_dir={raw_data_dir}")
    pipeline.add_stage(fleurs_stage)


def _validate_written_manifest(final_manifest: str, expected_rows: int) -> None:
    """Ensure the writer persisted every output task as a JSON object."""
    manifest_path = Path(final_manifest)
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        msg = f"Audio tagging output manifest is missing or empty: {manifest_path}"
        raise RuntimeError(msg)

    written_rows = 0
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"Audio tagging output manifest has invalid JSON on line {line_number}: {e}"
                raise RuntimeError(msg) from e
            if not isinstance(row, dict):
                msg = f"Audio tagging output manifest line {line_number} is not a JSON object"
                raise TypeError(msg)
            written_rows += 1
    if written_rows != expected_rows:
        msg = f"Audio tagging output manifest contains {written_rows} rows; expected {expected_rows}"
        raise RuntimeError(msg)


def _validate_second_pass_outputs(tasks: Sequence[AudioTask]) -> dict[str, int | float]:
    """Require second-pass ASR and WER output on every prepared segment."""
    total_segments = 0
    second_pass_segments = 0
    wer_segments = 0

    for task in tasks:
        for segment in task.data["segments"]:
            total_segments += 1
            if isinstance(segment.get("text_2"), str):
                second_pass_segments += 1

            metrics = segment["metrics"]
            wer = metrics.get("wer") if isinstance(metrics, Mapping) else None
            if isinstance(wer, Mapping):
                try:
                    wer_value = float(wer["wer"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(wer_value):
                    wer_segments += 1

    if total_segments == 0:
        msg = "Second-pass output validation received no prepared segments"
        raise RuntimeError(msg)
    if second_pass_segments != total_segments:
        msg = (
            f"Second-pass ASR populated {second_pass_segments} of {total_segments} prepared segments; "
            "ASRAlignment2 may have been skipped"
        )
        raise RuntimeError(msg)
    if wer_segments != total_segments:
        msg = (
            f"ComputeWER populated {wer_segments} of {total_segments} prepared segments; "
            "the WER stage may have been skipped"
        )
        raise RuntimeError(msg)

    return {
        "num_segments_with_second_pass_asr": second_pass_segments,
        "second_pass_asr_segment_coverage_ratio": second_pass_segments / total_segments,
        "num_segments_with_wer": wer_segments,
        "wer_segment_coverage_ratio": wer_segments / total_segments,
    }


def _validate_required_stages_executed(tasks: Sequence[AudioTask]) -> dict[str, int | float]:
    """Require every benchmark processing stage to report non-zero work."""
    processed_by_stage = dict.fromkeys(_REQUIRED_STAGE_NAMES, 0)
    for task in tasks:
        for stage_perf in task._stage_perf:
            if stage_perf.stage_name in processed_by_stage:
                processed_by_stage[stage_perf.stage_name] += stage_perf.num_items_processed

    skipped_stages = [stage_name for stage_name, num_items in processed_by_stage.items() if num_items <= 0]
    if skipped_stages:
        msg = f"Required audio tagging stages processed no data: {', '.join(skipped_stages)}"
        raise RuntimeError(msg)

    num_required_stages = len(_REQUIRED_STAGE_NAMES)
    return {
        "num_required_stages": num_required_stages,
        "num_executed_required_stages": num_required_stages - len(skipped_stages),
        "stage_execution_coverage_ratio": 1.0,
    }


def run_audio_tagging_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    repeat_factor: int,
    hf_token: str,
    max_segment_length: float,
    asr_batch_size: int,
    executor: str,
    input_manifest: str | None = None,
    raw_data_dir: str | None = None,
    lang: str = "en_us",
    split: str = "dev",
    auto_download: bool = True,
    cache_dir: str | None = None,
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the full audio tagging pipeline benchmark."""
    benchmark_results_path = Path(benchmark_results_path)
    results_dir = benchmark_results_path / "results"

    resampled_audio_dir = str(benchmark_results_path / "audio_resampled")
    final_manifest = str(results_dir / "tagging_output.jsonl")

    logger.info("Starting audio tagging pipeline benchmark")
    logger.info(f"Max segment length: {max_segment_length}s")
    if repeat_factor < 1:
        msg = "repeat_factor must be at least 1"
        raise ValueError(msg)

    pipeline = Pipeline(
        name="audio_tagging_benchmark",
        description="Full TTS tagging benchmark with second-pass ASR and WER",
    )

    _add_input_stage(pipeline, input_manifest, raw_data_dir, lang, split, auto_download, cache_dir)
    if repeat_factor > 1:
        pipeline.add_stage(RepeatEntriesStage(repeat_factor=repeat_factor, unique_id_key="audio_item_id"))
        logger.info(f"Repeat factor: {repeat_factor}x (entries multiplied after reading from manifest)")

    exc = setup_executor(executor, config={"execution_mode": "streaming"})
    run_start_time = time.perf_counter()

    # Resample audio to 16 kHz mono WAV
    pipeline.add_stage(
        ResampleAudioStage(
            resampled_audio_dir=resampled_audio_dir,
            input_format="wav",
            target_sample_rate=16000,
            target_format="wav",
            target_nchannels=1,
        ).with_(resources=Resources(cpus=1))
    )

    # Speaker diarization and overlap detection (PyAnnote)
    # NOTE: Fractional GPU values below are benchmark-specific empirical settings
    # tuned for a single-GPU setup. They are hardware/workload dependent and should
    # not be copied as production defaults into other pipeline configs.
    pipeline.add_stage(
        PyAnnoteDiarizationStage(
            name="PyAnnoteDiarization",
            hf_token=hf_token,
            max_length=max_segment_length,
        ).with_(resources=Resources(cpus=1, gpus=0.4))
    )

    # Split long audio segments
    pipeline.add_stage(
        SplitLongAudioStage(
            name="SplitLongAudio",
            suggested_max_len=max_segment_length,
            min_len=1.0,
        ).with_(resources=Resources(cpus=1))
    )

    # ASR forced alignment (NeMo FastConformer) — ~19 GB VRAM with CUDA graphs
    pipeline.add_stage(
        NeMoASRAlignerStage(
            name="ASRAlignment",
            is_fastconformer=True,
            decoder_type="rnnt",
            batch_size=asr_batch_size,
        ).with_(resources=Resources(cpus=1, gpus=0.45))
    )

    # Rejoin split audio metadata
    pipeline.add_stage(JoinSplitAudioMetadataStage(name="JoinSplitMetadata").with_(resources=Resources(cpus=1)))

    # Merge alignment with diarization
    pipeline.add_stage(
        MergeAlignmentDiarizationStage(
            name="MergeAlignmentDiar",
            text_key="text",
            words_key="words",
        ).with_(resources=Resources(cpus=1))
    )

    # Bandwidth estimation per segment
    pipeline.add_stage(BandwidthEstimationStage(name="BandwidthEstimation").with_(resources=Resources(cpus=1)))

    # Audio quality metrics (PESQ, STOI, SI-SDR)
    # NOTE: gpus=0.05 is a benchmark-specific empirical value for this single-GPU
    # setup and should not be used as a production default.
    pipeline.add_stage(TorchSquimQualityMetricsStage(name="SquimMetrics").with_(resources=Resources(gpus=0.05)))

    # Prepare TTS segments
    pipeline.add_stage(
        PrepareModuleSegmentsStage(
            name="PrepareModuleSegments",
            module="tts",
            min_duration=5,
            max_duration=20,
            full_utterance_ratio=1.0,
        ).with_(resources=Resources(cpus=1))
    )

    # Second-pass segment ASR. A 0.1 GPU reservation keeps all streaming
    # benchmark stage reservations within the single-GPU nightly allocation.
    pipeline.add_stage(
        NeMoASRAlignerStage(
            name="ASRAlignment2",
            model_name="nvidia/stt_en_conformer_ctc_large",
            is_fastconformer=False,
            decoder_type="ctc",
            batch_size=64,
            split_batch_size=100,
            text_key="text_2",
            infer_segment_only=True,
            compute_timestamps=False,
        ).with_(resources=Resources(cpus=1, gpus=0.1))
    )

    # Cross-check the first-pass text against the second-pass transcript.
    pipeline.add_stage(
        ComputeWERStage(
            name="ComputeWER",
            language="en",
            hypothesis_text_key="text_2",
            reference_text_key="text",
            pnc_chars=".?,",
            compute_pnc_wer=False,
        ).with_(resources=Resources(cpus=1))
    )

    # Write output manifest
    pipeline.add_stage(
        ManifestWriterStage(name="ManifestWriter", output_path=final_manifest).with_(resources=Resources(cpus=1))
    )

    logger.info(pipeline.describe())

    results = pipeline.run(exc)

    run_time_taken = time.perf_counter() - run_start_time
    output_metrics = validate_tagging_outputs(results)
    output_metrics.update(_validate_second_pass_outputs(results))
    output_metrics.update(_validate_required_stages_executed(results))
    _validate_written_manifest(final_manifest, int(output_metrics["num_tasks_processed"]))
    audio_hours_per_wall_hour = (
        output_metrics["total_audio_duration_hours"] * 3600 / run_time_taken if run_time_taken > 0 else 0
    )

    logger.success("Audio tagging benchmark completed successfully")
    logger.success(f"Processed {output_metrics['num_tasks_processed']} tasks")
    logger.success(f"Produced {output_metrics['num_segments_processed']} tagged segments")
    logger.success(f"Segment task coverage: {output_metrics['segment_task_coverage_ratio']:.1%}")
    logger.success(f"Total audio duration processed: {output_metrics['total_audio_duration_hours']:.2f} hours")
    logger.success(f"Tagged audio duration: {output_metrics['tagged_audio_duration_hours']:.2f} hours")
    logger.success(f"Throughput: {audio_hours_per_wall_hour:.2f} audio hours per wall hour")
    logger.success(f"Total time taken: {run_time_taken / 60:.2f} minutes")

    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            **output_metrics,
            "throughput_tasks_per_sec": (
                output_metrics["num_tasks_processed"] / run_time_taken if run_time_taken > 0 else 0
            ),
            "throughput_audio_hours_per_hour": audio_hours_per_wall_hour,
        },
        "tasks": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the audio-tagging benchmark CLI parser."""
    parser = argparse.ArgumentParser(
        description="Audio tagging pipeline end-to-end benchmark",
        epilog="Set HF_TOKEN in the environment for PyAnnote model access.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-manifest", help="Path to a local input manifest")
    input_group.add_argument(
        "--raw-data-dir",
        help="Parent directory containing a staged FLEURS <lang>/<split>.tsv and <lang>/<split>/ layout",
    )
    parser.add_argument("--lang", default="en_us", help="FLEURS language code when --raw-data-dir is used")
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev", "test"],
        help="FLEURS split when --raw-data-dir is used",
    )
    parser.add_argument(
        "--no-auto-download",
        dest="auto_download",
        action="store_false",
        help="Require FLEURS to be pre-staged instead of downloading it during the benchmark",
    )
    parser.set_defaults(auto_download=True)
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache for FLEURS downloads")
    parser.add_argument("--repeat-factor", type=int, default=1, help="Repeat factor for the input manifest entries")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to write benchmark results")
    parser.add_argument(
        "--max-segment-length", type=float, default=40.0, help="Maximum segment duration (seconds) to infer ASR"
    )
    parser.add_argument("--asr-batch-size", type=int, default=100, help="Batch size for first-pass ASR alignment")
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data", "ray_actors"], help="Executor")
    return parser


def main() -> int:
    parser = build_arg_parser()

    args = parser.parse_args()

    params = _sanitize_sensitive_params(vars(args))
    logger.info("=== Audio Tagging Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {params}")

    success_code = 1

    result_dict: dict[str, Any] = {
        "params": params,
        "metrics": {
            "is_success": False,
            "time_taken_s": 0.0,
            "num_tasks_processed": 0,
            "num_tasks_with_segments": 0,
            "num_segments_processed": 0,
            "segment_task_coverage_ratio": 0.0,
            "num_segments_with_second_pass_asr": 0,
            "second_pass_asr_segment_coverage_ratio": 0.0,
            "num_segments_with_wer": 0,
            "wer_segment_coverage_ratio": 0.0,
            "num_required_stages": len(_REQUIRED_STAGE_NAMES),
            "num_executed_required_stages": 0,
            "stage_execution_coverage_ratio": 0.0,
            "total_audio_duration_hours": 0.0,
            "tagged_audio_duration_hours": 0.0,
            "throughput_tasks_per_sec": 0.0,
            "throughput_audio_hours_per_hour": 0.0,
        },
        "tasks": [],
    }
    hf_token = ""
    try:
        hf_token = _resolve_hf_token()
        result_dict.update(run_audio_tagging_benchmark(**vars(args), hf_token=hf_token))
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    except Exception as e:
        error_message = _redact_known_secrets(str(e), [hf_token])
        logger.error(f"Benchmark failed: {error_message}")
        result_dict["metrics"]["error_message"] = error_message
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
