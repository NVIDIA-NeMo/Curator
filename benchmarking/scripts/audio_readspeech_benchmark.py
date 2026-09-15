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

This script benchmarks the DNS Challenge Read Speech audio curation pipeline,
which processes WAV files through quality filtering (VAD, band filter, UTMOS,
SIGMOS, speaker separation) and outputs a filtered JSONL manifest.

Can be invoked standalone or through the benchmarking framework.
"""

import argparse
import inspect
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import AudioDataFilterStage
from nemo_curator.stages.audio.datasets.readspeech import CreateInitialManifestReadSpeechStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter


def _count_jsonl_lines(results_dir: Path) -> int:
    """Count total lines across all JSONL files in a directory."""
    total = 0
    for jsonl_file in results_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            total += sum(1 for _ in f)
    return total


def run_readspeech_benchmark(  # noqa: PLR0913, PLR0915
    benchmark_results_path: str,
    scratch_output_path: str,
    raw_data_dir: str | None = None,
    executor: str = "xenna",
    max_samples: int = 5000,
    batch_size: int = 1,
    sample_rate: int = 48000,
    auto_download: bool = True,
    enable_vad: bool = False,
    vad_min_duration: float = 2.0,
    vad_max_duration: float = 60.0,
    vad_threshold: float = 0.5,
    vad_min_interval_ms: int = 500,
    vad_speech_pad_ms: int = 300,
    enable_band_filter: bool = False,
    band_value: str = "full_band",
    enable_utmos: bool = False,
    utmos_mos_threshold: float = 3.4,
    enable_sigmos: bool = False,
    sigmos_noise_threshold: float = 4.0,
    sigmos_ovrl_threshold: float = 3.5,
    enable_speaker_separation: bool = False,
    speaker_exclude_overlaps: bool = True,
    speaker_min_duration: float = 0.8,
) -> dict[str, Any]:
    """Run the ReadSpeech audio curation benchmark and collect metrics."""

    benchmark_results_path = Path(benchmark_results_path)
    scratch_output_path = Path(scratch_output_path)
    results_dir = benchmark_results_path / "results"

    if results_dir.exists():
        msg = f"Result directory {results_dir} already exists."
        raise ValueError(msg)

    logger.info("Starting ReadSpeech audio curation benchmark")
    logger.info(f"Executor: {executor}")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Sample rate: {sample_rate}")
    logger.info(f"Auto download: {auto_download}")

    enabled_filters = []
    if enable_vad:
        enabled_filters.append("VAD")
    if enable_band_filter:
        enabled_filters.append(f"Band({band_value})")
    if enable_utmos:
        enabled_filters.append(f"UTMOS(threshold={utmos_mos_threshold})")
    if enable_sigmos:
        enabled_filters.append("SIGMOS")
    if enable_speaker_separation:
        enabled_filters.append("SpeakerSep")
    logger.info(f"Enabled filters: {enabled_filters or ['none']}")

    if raw_data_dir:
        data_dir = Path(raw_data_dir)
        logger.info(f"Using pre-downloaded data at: {data_dir}")
    else:
        data_dir = scratch_output_path / "read_speech"
        logger.info(f"Data directory (auto-download to scratch): {data_dir}")

    executor_obj = setup_executor(executor)
    pipeline = Pipeline(
        name="readspeech_benchmark",
        description="DNS Challenge Read Speech audio curation benchmark",
    )

    pipeline.add_stage(
        CreateInitialManifestReadSpeechStage(
            raw_data_dir=data_dir,
            max_samples=max_samples,
            auto_download=auto_download,
            batch_size=batch_size,
        )
    )

    pipeline.add_stage(AudioDataFilterStage(config={
        "mono_conversion": {
            "output_sample_rate": sample_rate,
        },
        "vad": {
            "enable": enable_vad,
            "min_duration_sec": vad_min_duration,
            "max_duration_sec": vad_max_duration,
            "threshold": vad_threshold,
            "min_interval_ms": vad_min_interval_ms,
            "speech_pad_ms": vad_speech_pad_ms,
        },
        "band_filter": {
            "enable": enable_band_filter,
            "band_value": band_value,
        },
        "utmos": {
            "enable": enable_utmos,
            "mos_threshold": utmos_mos_threshold,
        },
        "sigmos": {
            "enable": enable_sigmos,
            "noise_threshold": sigmos_noise_threshold,
            "ovrl_threshold": sigmos_ovrl_threshold,
        },
        "speaker_separation": {
            "enable": enable_speaker_separation,
            "exclude_overlaps": speaker_exclude_overlaps,
            "min_duration": speaker_min_duration,
        },
        "timestamp_mapper": {
            "passthrough_keys": [
                "band_prediction", "utmos_mos",
                "sigmos_noise", "sigmos_ovrl",
                "speaker_id", "num_speakers",
            ],
        },
    }))

    pipeline.add_stage(AudioToDocumentStage())
    pipeline.add_stage(JsonlWriter(
        path=results_dir,
        write_kwargs={"force_ascii": False},
    ))

    logger.info(f"Pipeline description:\n{pipeline.describe()}")

    run_start = time.perf_counter()

    try:
        logger.info("Running ReadSpeech pipeline...")
        output_tasks = pipeline.run(executor_obj)
        run_time_s = time.perf_counter() - run_start

        num_output_segments = _count_jsonl_lines(results_dir)

        logger.success(f"Benchmark completed in {run_time_s:.2f}s")
        logger.success(f"Output segments: {num_output_segments}")

        throughput = num_output_segments / run_time_s if run_time_s > 0 else 0
        success = True

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_s = time.perf_counter() - run_start
        num_output_segments = 0
        throughput = 0.0
        success = False

    return {
        "params": {
            "executor": executor,
            "raw_data_dir": str(raw_data_dir) if raw_data_dir else None,
            "max_samples": max_samples,
            "sample_rate": sample_rate,
            "auto_download": auto_download,
            "enable_vad": enable_vad,
            "enable_band_filter": enable_band_filter,
            "enable_utmos": enable_utmos,
            "enable_sigmos": enable_sigmos,
            "enable_speaker_separation": enable_speaker_separation,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_s,
            "num_output_segments": num_output_segments,
            "max_samples_input": max_samples,
            "throughput_segments_per_sec": throughput,
        },
        "tasks": output_tasks or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DNS Challenge Read Speech audio curation benchmark",
    )
    parser.add_argument("--benchmark-results-path", required=True,
                        help="Path to benchmark results directory")
    parser.add_argument("--scratch-output-path", required=True,
                        help="Path to scratch output directory (dataset download + temp files)")
    parser.add_argument("--raw-data-dir", default=None,
                        help="Path to pre-downloaded ReadSpeech WAV files (skips download to scratch)")
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"],
                        help="Executor to use (default: xenna)")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Maximum samples to process (-1 for all, default: 5000)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for manifest creation (default: 1)")
    parser.add_argument("--sample-rate", type=int, default=48000,
                        help="Target sample rate (default: 48000)")
    parser.add_argument("--no-auto-download", dest="auto_download", action="store_false",
                        help="Disable automatic dataset download (default: enabled)")
    parser.set_defaults(auto_download=True)

    parser.add_argument("--enable-vad", action="store_true",
                        help="Enable VAD segmentation")
    parser.add_argument("--vad-min-duration", type=float, default=2.0,
                        help="Min VAD segment duration in seconds (default: 2.0)")
    parser.add_argument("--vad-max-duration", type=float, default=60.0,
                        help="Max VAD segment duration in seconds (default: 60.0)")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="VAD detection threshold 0-1 (default: 0.5)")
    parser.add_argument("--vad-min-interval-ms", type=int, default=500,
                        help="Min silence interval to split in ms (default: 500)")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=300,
                        help="Padding before/after speech in ms (default: 300)")

    parser.add_argument("--enable-band-filter", action="store_true",
                        help="Enable band filter")
    parser.add_argument("--band-value", choices=["full_band", "narrow_band"],
                        default="full_band", help="Band filter target (default: full_band)")

    parser.add_argument("--enable-utmos", action="store_true",
                        help="Enable UTMOS quality filter")
    parser.add_argument("--utmos-mos-threshold", type=float, default=3.4,
                        help="Min UTMOS MOS score 1-5 (default: 3.4)")

    parser.add_argument("--enable-sigmos", action="store_true",
                        help="Enable SIGMOS quality filter")
    parser.add_argument("--sigmos-noise-threshold", type=float, default=4.0,
                        help="Min SIGMOS noise score (default: 4.0)")
    parser.add_argument("--sigmos-ovrl-threshold", type=float, default=3.5,
                        help="Min SIGMOS overall score (default: 3.5)")

    parser.add_argument("--enable-speaker-separation", action="store_true",
                        help="Enable speaker separation")
    parser.add_argument("--no-speaker-exclude-overlaps", dest="speaker_exclude_overlaps",
                        action="store_false",
                        help="Allow overlapping speaker segments (default: excluded)")
    parser.set_defaults(speaker_exclude_overlaps=True)
    parser.add_argument("--speaker-min-duration", type=float, default=0.8,
                        help="Min speaker segment duration in seconds (default: 0.8)")

    args = parser.parse_args()

    logger.info("=== ReadSpeech Audio Curation Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    success_code = 1

    result_dict: dict[str, Any] = {
        "params": vars(args),
        "metrics": {
            "is_success": False,
        },
        "tasks": [],
    }
    try:
        # Pass only the argparse fields that match run_readspeech_benchmark's
        # signature. Combined with removing the function's **kwargs sink, this
        # means: (a) any future argparse-only flag (e.g. a script-local --debug)
        # is intentionally filtered out here, and (b) any function parameter
        # that drifts away from its argparse counterpart now raises TypeError
        # loudly instead of being silently swallowed.
        sig = inspect.signature(run_readspeech_benchmark)
        known_params = {k: v for k, v in vars(args).items() if k in sig.parameters}
        result_dict.update(run_readspeech_benchmark(**known_params))
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
