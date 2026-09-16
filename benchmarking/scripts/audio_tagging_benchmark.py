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

"""Benchmark the complete audio-tagging pipeline on AMI meeting audio."""

import argparse
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from loguru import logger
from utils import setup_executor, write_benchmark_results

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

DEFAULT_AUDIO_TAGGING_CACHE_DIR = "/tmp/curator/audio_tagging_cache"  # noqa: S108
DATASET_DIR_NAME = "audio_tagging_ami_sdm"
AUDIO_TAGGING_HF_FILENAMES = (
    "manifest.jsonl",
    "audio/EN2002b.Array1-01.wav",
    "audio/ES2004c.Array1-01.wav",
    "audio/TS3003a.Array1-01.wav",
)

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


def _count_jsonl_rows(path: Path, label: str) -> int:
    return len(_load_jsonl_rows(path, label))


def _prestaged_paths(data_dir: Path) -> tuple[Path, Path]:
    return data_dir / "manifest.jsonl", data_dir / "audio"


def _is_prestaged(data_dir: Path) -> bool:
    try:
        _locate_prestaged_data(data_dir)
    except FileNotFoundError:
        return False
    return True


def _locate_prestaged_data(data_dir: Path) -> tuple[Path, Path]:
    manifest_path, audio_dir = _prestaged_paths(data_dir)
    if not manifest_path.is_file():
        msg = (
            f"Pre-staged audio-tagging manifest not found at {manifest_path}. "
            "Stage the dataset under <raw-data-dir>/manifest.jsonl and <raw-data-dir>/audio/, "
            "or run standalone with auto_download=True and an HF dataset repo."
        )
        raise FileNotFoundError(msg)
    if not audio_dir.is_dir():
        msg = (
            f"Pre-staged audio-tagging audio directory not found at {audio_dir}. "
            "Stage the dataset under <raw-data-dir>/manifest.jsonl and <raw-data-dir>/audio/, "
            "or run standalone with auto_download=True and an HF dataset repo."
        )
        raise FileNotFoundError(msg)
    return manifest_path, audio_dir


def _write_staged_manifest(source_manifest: Path, target_manifest: Path, target_audio_dir: Path) -> int:
    rows = _load_jsonl_rows(source_manifest, "Source audio-tagging manifest")
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    with target_manifest.open("w", encoding="utf-8") as target_file:
        for line_number, row in enumerate(rows, start=1):
            staged_audio_path = target_audio_dir / Path(row["audio_filepath"]).name
            if not staged_audio_path.is_file():
                msg = f"Staged audio file missing for source manifest line {line_number}: {staged_audio_path}"
                raise FileNotFoundError(msg)
            row["audio_filepath"] = str(staged_audio_path)
            target_file.write(json.dumps(row) + "\n")
    return len(rows)


def _download_stage_data(hf_repo_id: str | None, cache_dir: Path, data_dir: Path) -> tuple[Path, Path, int]:
    """Download a HF-hosted AMI benchmark payload into scratch for standalone runs."""
    if not hf_repo_id:
        msg = (
            "Standalone audio-tagging auto-download requires --hf-repo-id or "
            "CURATOR_AUDIO_TAGGING_HF_REPO_ID. Nightly should pass --raw-data-dir "
            "and --no-auto-download instead."
        )
        raise ValueError(msg)
    data_dir.mkdir(parents=True, exist_ok=True)
    target_manifest, target_audio_dir = _prestaged_paths(data_dir)
    target_audio_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = {
        filename: Path(
            hf_hub_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                filename=filename,
                cache_dir=str(cache_dir),
            )
        )
        for filename in AUDIO_TAGGING_HF_FILENAMES
    }

    source_manifest = downloaded_files["manifest.jsonl"]
    for filename in AUDIO_TAGGING_HF_FILENAMES[1:]:
        downloaded_audio_path = downloaded_files[filename]
        shutil.copy2(downloaded_audio_path, target_audio_dir / downloaded_audio_path.name)
    num_rows = _write_staged_manifest(source_manifest, target_manifest, target_audio_dir)
    manifest_path, audio_dir = _locate_prestaged_data(data_dir)
    return manifest_path, audio_dir, num_rows


def _resolve_data_dir(  # noqa: PLR0913
    scratch_output_path: Path,
    input_manifest: str | None,
    raw_data_dir: str | None,
    auto_download: bool,
    cache_dir: str | None,
    hf_repo_id: str | None,
) -> tuple[Path, Path, Path | None, int]:
    if input_manifest:
        manifest_path = Path(input_manifest)
        return (
            manifest_path.parent,
            manifest_path,
            None,
            _count_jsonl_rows(manifest_path, "Input manifest"),
        )

    if raw_data_dir:
        data_dir = Path(raw_data_dir)
        source_manifest, audio_dir = _locate_prestaged_data(data_dir)
        run_manifest = scratch_output_path / DATASET_DIR_NAME / "manifest.jsonl"
        num_rows = _write_staged_manifest(source_manifest, run_manifest, audio_dir)
        return data_dir, run_manifest, None, num_rows

    data_dir = scratch_output_path / DATASET_DIR_NAME
    cache_path = Path(
        cache_dir or os.environ.get("CURATOR_AUDIO_TAGGING_CACHE_DIR") or DEFAULT_AUDIO_TAGGING_CACHE_DIR
    )
    if _is_prestaged(data_dir):
        manifest_path, _audio_dir = _locate_prestaged_data(data_dir)
        return data_dir, manifest_path, cache_path, _count_jsonl_rows(manifest_path, "Input manifest")
    if auto_download:
        repo_id = hf_repo_id or os.environ.get("CURATOR_AUDIO_TAGGING_HF_REPO_ID")
        manifest_path, _audio_dir, num_rows = _download_stage_data(repo_id, cache_path, data_dir)
        return data_dir, manifest_path, cache_path, num_rows
    manifest_path, _audio_dir = _locate_prestaged_data(data_dir)
    return data_dir, manifest_path, cache_path, _count_jsonl_rows(manifest_path, "Input manifest")


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

    start = segment["start"]
    end = segment["end"]
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
        duration = task.data["duration"]
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

    output_manifests = sorted(final_manifest.parent.glob("*.jsonl"))
    if output_manifests != [final_manifest]:
        msg = f"Expected only {final_manifest}, found {output_manifests}"
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
        "total_audio_duration_hours": total_duration / 3600,
        "tagged_audio_duration_hours": tagged_duration / 3600,
    }


def run_audio_tagging_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    scratch_output_path: str,
    diarization_model_path: str,
    max_segment_length: float,
    asr_batch_size: int,
    executor: str,
    input_manifest: str | None = None,
    raw_data_dir: str | None = None,
    auto_download: bool = True,
    cache_dir: str | None = None,
    hf_repo_id: str | None = None,
    asr_transcribe_batch_size: int = 32,
    squim_compute_batch_size: int = 32,
    diarization_segmentation_batch_size: int = 128,
    diarization_embedding_batch_size: int = 128,
    use_cuda_graphs: bool = True,
    execution_mode: str | None = None,
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the full audio-tagging pipeline on pre-staged audio and models."""
    benchmark_results_path = Path(benchmark_results_path)
    scratch_output_path = Path(scratch_output_path)
    data_dir, input_manifest_path, data_cache_dir, num_input_rows = _resolve_data_dir(
        scratch_output_path=scratch_output_path,
        input_manifest=input_manifest,
        raw_data_dir=raw_data_dir,
        auto_download=auto_download,
        cache_dir=cache_dir,
        hf_repo_id=hf_repo_id,
    )
    diarization_model = Path(diarization_model_path)
    if not diarization_model.exists():
        msg = f"Pre-staged PyAnnote model not found: {diarization_model}"
        raise FileNotFoundError(msg)

    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Auto download: {auto_download}")
    logger.info(f"HF cache dir: {data_cache_dir}")
    logger.info(f"Input manifest: {input_manifest_path}")
    logger.info(f"Diarization model: {diarization_model}")

    logger.info(f"Selected source workload: {num_input_rows} unique meetings")
    results_dir = benchmark_results_path / "results"
    final_manifest = results_dir / "tagging_output.jsonl"

    executor_config = {"execution_mode": execution_mode} if execution_mode else None
    exc = setup_executor(executor, config=executor_config)
    run_start_time = time.perf_counter()
    pipeline = Pipeline(name="audio_tagging_benchmark", description="AMI meetings -> full audio tagging")

    pipeline.add_stage(ManifestReader(manifest_path=str(input_manifest_path)))

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
            segmentation_batch_size=diarization_segmentation_batch_size,
            embedding_batch_size=diarization_embedding_batch_size,
            max_length=max_segment_length,
        )
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
        )
    )
    pipeline.add_stage(JoinSplitAudioMetadataStage(name="JoinSplitMetadata").with_(resources=Resources(cpus=1)))
    pipeline.add_stage(
        MergeAlignmentDiarizationStage(name="MergeAlignmentDiar", text_key="text", words_key="words").with_(
            resources=Resources(cpus=1)
        )
    )
    pipeline.add_stage(BandwidthEstimationStage(name="BandwidthEstimation").with_(resources=Resources(cpus=1)))
    pipeline.add_stage(TorchSquimQualityMetricsStage(name="SquimMetrics", compute_batch_size=squim_compute_batch_size))
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
        )
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
    parser.add_argument("--diarization-model-path", required=True, help="Pre-staged local PyAnnote pipeline directory")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to write benchmark results")
    parser.add_argument(
        "--scratch-output-path",
        required=True,
        help="Path to scratch output directory (standalone data staging + temp files)",
    )
    parser.add_argument(
        "--input-manifest",
        default=None,
        help="Path to one pre-staged JSONL manifest. Nightly Xenna and Ray Data use this form.",
    )
    parser.add_argument(
        "--raw-data-dir",
        default=None,
        help=(
            "Parent workspace directory for audio-tagging staging. Pre-staged data lives under "
            "<raw-data-dir>/manifest.jsonl and <raw-data-dir>/audio/. Use with --no-auto-download "
            "to avoid standalone HF downloads."
        ),
    )
    parser.add_argument(
        "--no-auto-download",
        dest="auto_download",
        action="store_false",
        help=(
            "Disable standalone data staging; read pre-staged data from "
            "<raw-data-dir>/ or <scratch-output-path>/audio_tagging_ami_sdm/."
        ),
    )
    parser.set_defaults(auto_download=True)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Hugging Face cache directory used only for standalone auto-download runs so repeated "
            f"runs reuse it. Defaults to $CURATOR_AUDIO_TAGGING_CACHE_DIR or {DEFAULT_AUDIO_TAGGING_CACHE_DIR}."
        ),
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help=(
            "Hugging Face dataset repo containing manifest.jsonl and audio/*.wav for standalone "
            "auto-download. Defaults to $CURATOR_AUDIO_TAGGING_HF_REPO_ID."
        ),
    )
    parser.add_argument("--max-segment-length", type=float, default=40.0, help="Maximum segment duration in seconds")
    parser.add_argument("--asr-batch-size", type=int, default=100, help="First-pass ASR batch size")
    parser.add_argument("--asr-transcribe-batch-size", type=int, default=32, help="ASR model batch size")
    parser.add_argument("--squim-compute-batch-size", type=int, default=32, help="SQUIM model batch size")
    parser.add_argument(
        "--diarization-segmentation-batch-size",
        type=int,
        default=128,
        help="PyAnnote segmentation batch size",
    )
    parser.add_argument(
        "--diarization-embedding-batch-size",
        type=int,
        default=128,
        help="PyAnnote speaker-embedding batch size",
    )
    parser.add_argument(
        "--disable-cuda-graphs",
        dest="use_cuda_graphs",
        action="store_false",
        help="Disable CUDA graph decoding for constrained local GPUs",
    )
    parser.set_defaults(use_cuda_graphs=True)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data", "ray_actors"])
    parser.add_argument(
        "--execution-mode",
        choices=["streaming", "batch"],
        default=None,
        help="Xenna execution mode. Defaults to streaming; ignored by other executors.",
    )

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
