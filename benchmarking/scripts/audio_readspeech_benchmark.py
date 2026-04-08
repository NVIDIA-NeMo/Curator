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

"""Audio ReadSpeech benchmarking script.

This script benchmarks the AudioDataFilterStage pipeline on the DNS Challenge
Read Speech dataset. It measures end-to-end throughput including mono conversion,
VAD, quality filtering (UTMOS, SIGMOS, BandFilter), speaker separation,
and timestamp mapping.

The pipeline auto-downloads the dataset (~4.88 GB) on first run.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.advanced_pipelines.audio_data_filter.audio_data_filter import (
    AudioDataFilterStage,
)
from nemo_curator.stages.audio.datasets.readspeech import CreateInitialManifestReadSpeechStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter


def run_audio_readspeech_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    scratch_output_path: str,
    raw_data_dir: str | None,
    auto_download: bool,
    max_samples: int,
    enable_vad: bool,
    enable_band_filter: bool,
    enable_utmos: bool,
    enable_sigmos: bool,
    enable_speaker_separation: bool,
    utmos_mos_threshold: float,
    sigmos_noise_threshold: float,
    sigmos_ovrl_threshold: float,
    sample_rate: int,
    executor: str = "xenna",
    gpus: int = 1,
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the audio ReadSpeech benchmark and collect comprehensive metrics."""

    benchmark_results_path = Path(benchmark_results_path)
    scratch_output_path = Path(scratch_output_path)
    data_dir = raw_data_dir or str(scratch_output_path / "readspeech")
    results_dir = benchmark_results_path / "results"

    run_start_time = time.perf_counter()

    try:
        if results_dir.exists():
            msg = f"Result directory {results_dir} already exists."
            raise ValueError(msg)  # noqa: TRY301

        logger.info("Starting audio ReadSpeech benchmark")
        logger.info(f"Executor: {executor}")
        logger.info(f"Max samples: {max_samples}")
        logger.info(f"GPUs: {gpus}")
        logger.info(f"Data dir: {data_dir}")
        logger.info(f"Auto download: {auto_download}")
        logger.info(
            f"Filters: VAD={enable_vad}, Band={enable_band_filter}, "
            f"UTMOS={enable_utmos}, SIGMOS={enable_sigmos}, "
            f"SpeakerSep={enable_speaker_separation}"
        )

        executor_obj = setup_executor(executor)

        pipeline = Pipeline(
            name="readspeech_benchmark",
            description="DNS Challenge Read Speech dataset curation benchmark",
        )

        pipeline.add_stage(
            CreateInitialManifestReadSpeechStage(
                raw_data_dir=data_dir,
                max_samples=max_samples,
                auto_download=auto_download,
                batch_size=1,
            )
        )

        pipeline.add_stage(
            AudioDataFilterStage(
                config={
                    "mono_conversion": {"output_sample_rate": sample_rate},
                    "vad": {"enable": enable_vad},
                    "band_filter": {"enable": enable_band_filter},
                    "utmos": {
                        "enable": enable_utmos,
                        "mos_threshold": utmos_mos_threshold,
                    },
                    "sigmos": {
                        "enable": enable_sigmos,
                        "noise_threshold": sigmos_noise_threshold,
                        "ovrl_threshold": sigmos_ovrl_threshold,
                    },
                    "speaker_separation": {"enable": enable_speaker_separation},
                }
            )
        )

        pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
        pipeline.add_stage(
            JsonlWriter(
                path=results_dir,
                write_kwargs={"force_ascii": False},
            )
        )

        logger.info("Running ReadSpeech pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(executor_obj)
        run_time_taken = time.perf_counter() - run_start_time

        num_tasks_processed = len(output_tasks) if output_tasks else 0
        num_input_files = max_samples if max_samples > 0 else 14279

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_tasks_processed} output tasks from {num_input_files} input files")
        success = True

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_tasks_processed = 0
        num_input_files = 0
        success = False

    filter_pass_rate = num_tasks_processed / num_input_files if num_input_files > 0 else 0

    return {
        "params": {
            "executor": executor,
            "raw_data_dir": data_dir,
            "auto_download": auto_download,
            "max_samples": max_samples,
            "sample_rate": sample_rate,
            "gpus": gpus,
            "enable_vad": enable_vad,
            "enable_band_filter": enable_band_filter,
            "enable_utmos": enable_utmos,
            "enable_sigmos": enable_sigmos,
            "enable_speaker_separation": enable_speaker_separation,
            "utmos_mos_threshold": utmos_mos_threshold,
            "sigmos_noise_threshold": sigmos_noise_threshold,
            "sigmos_ovrl_threshold": sigmos_ovrl_threshold,
            "benchmark_results_path": str(benchmark_results_path),
            "scratch_output_path": str(scratch_output_path),
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_tasks_processed": num_tasks_processed,
            "num_input_files": num_input_files,
            "throughput_tasks_per_sec": num_tasks_processed / run_time_taken if run_time_taken > 0 else 0,
            "filter_pass_rate": filter_pass_rate,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio ReadSpeech benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--scratch-output-path", required=True, help="Path to scratch output directory")
    parser.add_argument(
        "--raw-data-dir", default=None,
        help="Path to pre-staged ReadSpeech WAV files. If not set, downloads to scratch-output-path.",
    )
    parser.add_argument("--auto-download", action="store_true", default=False, help="Auto-download dataset (~4.88 GB)")
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"], help="Executor to use")
    parser.add_argument("--max-samples", type=int, default=5000, help="Maximum samples to process (-1 for all)")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Target sample rate")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--enable-vad", action="store_true", default=True, help="Enable VAD segmentation")
    parser.add_argument("--no-vad", dest="enable_vad", action="store_false", help="Disable VAD")
    parser.add_argument("--enable-band-filter", action="store_true", default=True, help="Enable band filter")
    parser.add_argument("--no-band-filter", dest="enable_band_filter", action="store_false", help="Disable band filter")
    parser.add_argument("--enable-utmos", action="store_true", default=True, help="Enable UTMOS filter")
    parser.add_argument("--no-utmos", dest="enable_utmos", action="store_false", help="Disable UTMOS")
    parser.add_argument("--enable-sigmos", action="store_true", default=True, help="Enable SIGMOS filter")
    parser.add_argument("--no-sigmos", dest="enable_sigmos", action="store_false", help="Disable SIGMOS")
    parser.add_argument(
        "--enable-speaker-separation", action="store_true", default=True, help="Enable speaker separation"
    )
    parser.add_argument(
        "--no-speaker-separation", dest="enable_speaker_separation", action="store_false", help="Disable speaker sep"
    )
    parser.add_argument("--utmos-mos-threshold", type=float, default=3.4, help="UTMOS MOS threshold (0-5)")
    parser.add_argument("--sigmos-noise-threshold", type=float, default=4.0, help="SIGMOS noise threshold (0-5)")
    parser.add_argument("--sigmos-ovrl-threshold", type=float, default=3.5, help="SIGMOS overall threshold (0-5)")

    args = parser.parse_args()

    logger.info("=== Audio ReadSpeech Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results: dict[str, Any] = {
        "params": vars(args),
        "metrics": {
            "is_success": False,
        },
        "tasks": [],
    }
    try:
        results.update(run_audio_readspeech_benchmark(**vars(args)))
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
