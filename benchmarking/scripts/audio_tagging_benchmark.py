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

Runs the core audio tagging pipeline end-to-end:
  ManifestReader -> Resample -> Diarize -> Split -> ASR Align ->
  Join -> Merge -> Write

Exercises the core stages of the tagging pipeline for regression tracking.
"""

import argparse
import time
from pathlib import Path
from typing import Any

from loguru import logger
from utils import RepeatEntriesStage, setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestReader, ManifestWriterStage
from nemo_curator.stages.audio.inference.speaker_diarization.pyannote import PyAnnoteDiarizationStage
from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import NeMoASRAlignerStage
from nemo_curator.stages.audio.tagging.merge_alignment_diarization import MergeAlignmentDiarizationStage
from nemo_curator.stages.audio.tagging.resample_audio import ResampleAudioStage
from nemo_curator.stages.audio.tagging.split import JoinSplitAudioMetadataStage, SplitLongAudioStage
from nemo_curator.stages.resources import Resources


def run_audio_tagging_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    input_manifest: str,
    repeat_factor: int,
    hf_token: str,
    max_segment_length: float,
    asr_batch_size: int,
    executor: str,
    cpus: int,
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the full audio tagging pipeline benchmark."""
    benchmark_results_path = Path(benchmark_results_path)
    results_dir = benchmark_results_path / "results"

    resampled_audio_dir = str(benchmark_results_path / "audio_resampled")
    final_manifest = str(results_dir / "tagging_output.jsonl")

    logger.info("Starting audio tagging pipeline benchmark")
    logger.info(f"CPUs: {cpus}")
    logger.info(f"Max segment length: {max_segment_length}s")

    exc = setup_executor(executor, config={"execution_mode": "streaming"})
    run_start_time = time.perf_counter()

    pipeline = Pipeline(
        name="audio_tagging_benchmark",
        description="Audio tagging core benchmark: FLEURS -> core tagging pipeline",
    )

    pipeline.add_stage(ManifestReader(manifest_path=input_manifest))
    if repeat_factor > 1:
        pipeline.add_stage(RepeatEntriesStage(repeat_factor=repeat_factor))
        logger.info(f"Repeat factor: {repeat_factor}x (entries multiplied after reading from manifest)")

    # Resample audio to 16 kHz mono WAV
    pipeline.add_stage(
        ResampleAudioStage(
            resampled_audio_dir=resampled_audio_dir,
            input_format="wav",
            target_sample_rate=16000,
            target_format="wav",
            target_nchannels=1,
        ).with_(resources=Resources(cpus=cpus))
    )

    # Speaker diarization and overlap detection (PyAnnote)
    pipeline.add_stage(
        PyAnnoteDiarizationStage(
            name="PyAnnoteDiarization",
            hf_token=hf_token,
            max_length=max_segment_length,
        ).with_(resources=Resources(cpus=cpus, gpus=0.5))
    )

    # Split long audio segments
    pipeline.add_stage(
        SplitLongAudioStage(
            name="SplitLongAudio",
            suggested_max_len=max_segment_length,
            min_len=1.0,
        ).with_(resources=Resources(cpus=cpus))
    )

    # ASR forced alignment (NeMo FastConformer)
    pipeline.add_stage(
        NeMoASRAlignerStage(
            name="ASRAlignment",
            is_fastconformer=True,
            decoder_type="rnnt",
            batch_size=asr_batch_size,
        ).with_(resources=Resources(cpus=cpus, gpus=0.45))
    )

    # Rejoin split audio metadata
    pipeline.add_stage(JoinSplitAudioMetadataStage(name="JoinSplitMetadata").with_(resources=Resources(cpus=cpus)))

    # Merge alignment with diarization
    pipeline.add_stage(
        MergeAlignmentDiarizationStage(
            name="MergeAlignmentDiar",
            text_key="text",
            words_key="words",
        ).with_(resources=Resources(cpus=cpus))
    )

    # Write output manifest
    pipeline.add_stage(ManifestWriterStage(output_path=final_manifest).with_(resources=Resources(cpus=cpus)))

    results = pipeline.run(exc)

    run_time_taken = time.perf_counter() - run_start_time

    total_duration = sum(task.data["duration"] for task in results) / 3600

    logger.success("Audio tagging benchmark completed successfully!!")
    logger.success(f"Processed {len(results)} tasks")
    logger.success(f"Total audio duration processed: {total_duration:.2f} hours")
    logger.success(f"Throughput: {len(results) / run_time_taken:.2f} tasks per second")
    logger.success(f"Total time taken: {run_time_taken / 60:.2f} minutes")

    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "num_tasks_processed": len(results),
            "throughput_tasks_per_sec": len(results) / run_time_taken if run_time_taken > 0 else 0,
            "total_audio_duration_hours": total_duration,
        },
        "tasks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audio tagging pipeline e2e benchmark (FLEURS -> full tagging pipeline)"
    )
    parser.add_argument("--input-manifest", required=True, help="Path to input manifest")
    parser.add_argument("--repeat-factor", type=int, default=1, help="Repeat factor for the input manifest entries")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to write benchmark results")
    parser.add_argument("--hf-token", default="", help="HuggingFace token for PyAnnote")
    parser.add_argument(
        "--max-segment-length", type=float, default=40.0, help="Maximum segment duration (seconds) to infer ASR"
    )
    parser.add_argument("--asr-batch-size", type=int, default=100, help="Batch size for ASR alignment")
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data", "ray_actors"], help="Executor")
    parser.add_argument("--cpus", type=int, default=10, help="Number of CPUs to use for the pipeline")

    args = parser.parse_args()

    logger.info("=== Audio Tagging Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    success_code = 1

    result_dict: dict[str, Any] = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    try:
        result_dict.update(run_audio_tagging_benchmark(**vars(args)))
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
