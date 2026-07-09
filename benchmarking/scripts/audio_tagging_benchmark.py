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

"""Benchmark the complete audio-tagging pipeline on pre-staged meeting audio."""

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from utils import RepeatEntriesStage, setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestReader, ManifestWriterStage
from nemo_curator.stages.audio.inference.speaker_diarization.pyannote import PyAnnoteDiarizationStage
from nemo_curator.stages.audio.metrics.bandwidth import BandwidthEstimationStage
from nemo_curator.stages.audio.metrics.squim import TorchSquimQualityMetricsStage
from nemo_curator.stages.audio.metrics.wer import ComputeWERStage
from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import NeMoASRAlignerStage
from nemo_curator.stages.audio.tagging.merge_alignment_diarization import MergeAlignmentDiarizationStage
from nemo_curator.stages.audio.tagging.prepare_module_segments import PrepareModuleSegmentsStage
from nemo_curator.stages.audio.tagging.resample_audio import ResampleAudioStage
from nemo_curator.stages.audio.tagging.split import JoinSplitAudioMetadataStage, SplitLongAudioStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

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


def _count_jsonl_rows(path: Path, label: str) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        msg = f"{label} is missing or empty: {path}"
        raise RuntimeError(msg)

    rows = 0
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
            rows += 1

    if rows == 0:
        msg = f"{label} contains no data rows: {path}"
        raise RuntimeError(msg)
    return rows


def _validate_segment(segment: object, label: str) -> tuple[float, bool, bool]:
    if not isinstance(segment, Mapping):
        msg = f"{label} must be a mapping"
        raise TypeError(msg)
    if not isinstance(segment.get("text"), str) or not segment["text"].strip():
        msg = f"{label} must contain first-pass text"
        raise RuntimeError(msg)
    words = segment.get("words")
    if not isinstance(words, list) or not words or not all(isinstance(word, Mapping) for word in words):
        msg = f"{label} must contain word alignments"
        raise RuntimeError(msg)

    start = _finite_float(segment.get("start"), f"{label} start")
    end = _finite_float(segment.get("end"), f"{label} end")
    if start < 0 or end <= start:
        msg = f"{label} has invalid timestamps"
        raise RuntimeError(msg)

    second_pass_text = segment.get("text_2")
    if second_pass_text is not None and not isinstance(second_pass_text, str):
        msg = f"{label} second-pass text must be a string"
        raise TypeError(msg)
    has_second_pass_text = isinstance(second_pass_text, str) and bool(second_pass_text.strip())

    metrics = segment.get("metrics")
    wer = metrics.get("wer") if isinstance(metrics, Mapping) else None
    if wer is not None and not isinstance(wer, Mapping):
        msg = f"{label} has invalid WER output"
        raise RuntimeError(msg)
    has_wer = isinstance(wer, Mapping)
    if has_wer:
        _finite_float(wer.get("wer"), f"{label} WER")
    return end - start, has_second_pass_text, has_wer


def _validate_outputs(  # noqa: C901
    tasks: Sequence[AudioTask], final_manifest: Path, num_input_rows: int
) -> dict[str, int | float | bool]:
    """Reject row loss, skipped stages, and zero or malformed tagging output."""
    if len(tasks) != num_input_rows:
        msg = f"Audio tagging returned {len(tasks)} rows for {num_input_rows} input rows"
        raise RuntimeError(msg)

    total_duration = 0.0
    tagged_duration = 0.0
    num_tasks_with_segments = 0
    num_segments = 0
    num_segments_emitted = 0
    num_segments_with_second_pass_asr = 0
    num_segments_with_wer = 0
    stage_items = dict.fromkeys(_REQUIRED_STAGE_NAMES, 0)

    for task_index, task in enumerate(tasks):
        duration = _finite_float(task.data.get("duration"), f"task {task_index} duration")
        if duration <= 0:
            msg = f"task {task_index} duration must be positive"
            raise RuntimeError(msg)
        total_duration += duration

        segments = task.data.get("segments")
        if not isinstance(segments, list):
            msg = f"task {task_index} must contain a segments list"
            raise TypeError(msg)
        task_has_processed_segment = False
        for segment_index, segment in enumerate(segments):
            segment_duration, has_second_pass_text, has_wer = _validate_segment(
                segment, f"task {task_index} segment {segment_index}"
            )
            num_segments_emitted += 1
            num_segments_with_second_pass_asr += has_second_pass_text
            num_segments_with_wer += has_wer
            if not (has_second_pass_text and has_wer):
                continue
            tagged_duration += segment_duration
            num_segments += 1
            task_has_processed_segment = True
        num_tasks_with_segments += task_has_processed_segment

        for perf in task._stage_perf:
            if perf.stage_name in stage_items:
                stage_items[perf.stage_name] += perf.num_items_processed

    if num_segments == 0:
        msg = "Audio tagging pipeline produced no complete tagged segments"
        raise RuntimeError(msg)
    skipped_stages = [name for name, count in stage_items.items() if count <= 0]
    if skipped_stages:
        msg = f"Required stages processed no data: {', '.join(skipped_stages)}"
        raise RuntimeError(msg)

    manifest_rows = _count_jsonl_rows(final_manifest, "Output manifest")
    if manifest_rows != num_input_rows:
        msg = f"Output manifest contains {manifest_rows} rows for {num_input_rows} input rows"
        raise RuntimeError(msg)

    return {
        "num_input_rows": num_input_rows,
        "num_output_rows": len(tasks),
        "num_manifest_rows": manifest_rows,
        "input_output_row_count_match": True,
        "num_tasks_processed": len(tasks),
        "num_tasks_with_segments": num_tasks_with_segments,
        "num_segments_processed": num_segments,
        "num_segments_emitted": num_segments_emitted,
        "num_segments_skipped": num_segments_emitted - num_segments,
        "segment_task_coverage_ratio": num_tasks_with_segments / len(tasks),
        "segment_output_coverage_ratio": num_segments / num_segments_emitted,
        "num_segments_with_second_pass_asr": num_segments_with_second_pass_asr,
        "num_segments_with_wer": num_segments_with_wer,
        "stage_execution_coverage_ratio": 1.0,
        "total_audio_duration_hours": total_duration / 3600,
        "tagged_audio_duration_hours": tagged_duration / 3600,
    }


def run_audio_tagging_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    input_manifest: str,
    diarization_model_path: str,
    repeat_factor: int,
    max_segment_length: float,
    asr_batch_size: int,
    executor: str,
    asr_transcribe_batch_size: int = 32,
    squim_compute_batch_size: int = 32,
    use_cuda_graphs: bool = True,
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the full audio-tagging pipeline on pre-staged audio and models."""
    benchmark_results_path = Path(benchmark_results_path)
    input_manifest_path = Path(input_manifest)
    diarization_model = Path(diarization_model_path)
    if repeat_factor < 1:
        msg = "repeat_factor must be at least 1"
        raise ValueError(msg)
    if not diarization_model.exists():
        msg = f"Pre-staged PyAnnote model not found: {diarization_model}"
        raise FileNotFoundError(msg)

    num_input_rows = _count_jsonl_rows(input_manifest_path, "Input manifest") * repeat_factor
    results_dir = benchmark_results_path / "results"
    final_manifest = results_dir / "tagging_output.jsonl"

    exc = setup_executor(executor, config={"execution_mode": "streaming"})
    run_start_time = time.perf_counter()
    pipeline = Pipeline(name="audio_tagging_benchmark", description="AMI meetings -> full audio tagging")

    pipeline.add_stage(ManifestReader(manifest_path=input_manifest))
    if repeat_factor > 1:
        pipeline.add_stage(RepeatEntriesStage(repeat_factor=repeat_factor, unique_id_key="audio_item_id"))

    pipeline.add_stage(
        ResampleAudioStage(
            resampled_audio_dir=str(benchmark_results_path / "audio_resampled"),
            input_format="wav",
            target_sample_rate=16000,
            target_format="wav",
            target_nchannels=1,
        ).with_(resources=Resources(cpus=1))
    )
    pipeline.add_stage(
        PyAnnoteDiarizationStage(
            name="PyAnnoteDiarization",
            model_name=str(diarization_model),
            max_length=max_segment_length,
        ).with_(resources=Resources(cpus=1, gpus=0.4))
    )
    pipeline.add_stage(
        SplitLongAudioStage(name="SplitLongAudio", suggested_max_len=max_segment_length, min_len=1.0).with_(
            resources=Resources(cpus=1)
        )
    )
    pipeline.add_stage(
        NeMoASRAlignerStage(
            name="ASRAlignment",
            is_fastconformer=True,
            decoder_type="rnnt",
            batch_size=asr_batch_size,
            transcribe_batch_size=asr_transcribe_batch_size,
            use_cuda_graphs=use_cuda_graphs,
        ).with_(resources=Resources(cpus=1, gpus=0.45))
    )
    pipeline.add_stage(JoinSplitAudioMetadataStage(name="JoinSplitMetadata").with_(resources=Resources(cpus=1)))
    pipeline.add_stage(
        MergeAlignmentDiarizationStage(name="MergeAlignmentDiar", text_key="text", words_key="words").with_(
            resources=Resources(cpus=1)
        )
    )
    pipeline.add_stage(BandwidthEstimationStage(name="BandwidthEstimation").with_(resources=Resources(cpus=1)))
    pipeline.add_stage(
        TorchSquimQualityMetricsStage(name="SquimMetrics", compute_batch_size=squim_compute_batch_size).with_(
            resources=Resources(gpus=0.05)
        )
    )
    pipeline.add_stage(
        PrepareModuleSegmentsStage(
            name="PrepareModuleSegments",
            module="tts",
            min_duration=5,
            max_duration=20,
            full_utterance_ratio=1.0,
        ).with_(resources=Resources(cpus=1))
    )
    pipeline.add_stage(
        NeMoASRAlignerStage(
            name="ASRAlignment2",
            model_name="nvidia/stt_en_conformer_ctc_large",
            is_fastconformer=False,
            decoder_type="ctc",
            batch_size=64,
            transcribe_batch_size=asr_transcribe_batch_size,
            split_batch_size=100,
            text_key="text_2",
            infer_segment_only=True,
            compute_timestamps=False,
            use_cuda_graphs=use_cuda_graphs,
        ).with_(resources=Resources(cpus=1, gpus=0.1))
    )
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
    pipeline.add_stage(
        ManifestWriterStage(name="ManifestWriter", output_path=str(final_manifest)).with_(resources=Resources(cpus=1))
    )

    logger.info(pipeline.describe())
    results = pipeline.run(exc)
    run_time_taken = time.perf_counter() - run_start_time
    output_metrics = _validate_outputs(results, final_manifest, num_input_rows)

    logger.success(
        f"Processed all {num_input_rows} input rows into "
        f"{output_metrics['num_segments_processed']} complete tagged segments"
    )
    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            **output_metrics,
            "throughput_tasks_per_sec": num_input_rows / run_time_taken if run_time_taken > 0 else 0,
            "throughput_audio_hours_per_hour": (
                output_metrics["total_audio_duration_hours"] * 3600 / run_time_taken if run_time_taken > 0 else 0
            ),
        },
        "tasks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio tagging benchmark on pre-staged meeting audio")
    parser.add_argument("--input-manifest", required=True, help="Pre-staged input JSONL manifest")
    parser.add_argument("--diarization-model-path", required=True, help="Pre-staged local PyAnnote pipeline directory")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to write benchmark results")
    parser.add_argument("--repeat-factor", type=int, default=1, help="Repeat each input row this many times")
    parser.add_argument("--max-segment-length", type=float, default=40.0, help="Maximum segment duration in seconds")
    parser.add_argument("--asr-batch-size", type=int, default=100, help="First-pass ASR batch size")
    parser.add_argument("--asr-transcribe-batch-size", type=int, default=32, help="ASR model batch size")
    parser.add_argument("--squim-compute-batch-size", type=int, default=32, help="SQUIM model batch size")
    parser.add_argument(
        "--disable-cuda-graphs",
        dest="use_cuda_graphs",
        action="store_false",
        help="Disable CUDA graph decoding for constrained local GPUs",
    )
    parser.set_defaults(use_cuda_graphs=True)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data", "ray_actors"])

    args = parser.parse_args()
    params = vars(args)
    logger.info(f"Audio tagging benchmark arguments: {params}")
    result_dict: dict[str, Any] = {"params": params, "metrics": {"is_success": False}, "tasks": []}
    success_code = 1
    try:
        result_dict.update(run_audio_tagging_benchmark(**params))
        success_code = 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        result_dict["metrics"]["error_message"] = str(e)
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
