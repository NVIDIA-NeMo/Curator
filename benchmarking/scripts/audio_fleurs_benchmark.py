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

"""Audio Fleurs benchmarking script.

This script runs audio Fleurs benchmarks with comprehensive metrics collection
and logs results to configured sinks.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.writer import JsonlWriter


def run_audio_fleurs_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    scratch_output_path: str,
    model_name: str,
    lang: str,
    split: str,
    wer_threshold: float,
    gpus: int,
    executor: str = "xenna",
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the audio fleurs benchmark and collect comprehensive metrics."""

    benchmark_results_path = Path(benchmark_results_path)
    scratch_output_path = Path(scratch_output_path)
    results_dir = benchmark_results_path / "results"

    run_start_time = time.perf_counter()

    try:
        if results_dir.exists():
            msg = f"Result directory {results_dir} already exists."
            raise ValueError(msg)  # noqa: TRY301

        logger.info("Starting audio fleurs benchmark")
        logger.info(f"Executor: {executor}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Language: {lang}")
        logger.info(f"Split: {split}")
        logger.info(f"WER threshold: {wer_threshold}")
        logger.info(f"GPUs: {gpus}")

        executor_obj = setup_executor(executor)
        pipeline = Pipeline(name="audio_inference", description="Inference audio and filter by WER threshold.")

        pipeline.add_stage(
            CreateInitialManifestFleursStage(
                lang=lang,
                split=split,
                raw_data_dir=scratch_output_path / "armenian/fleurs",
            ).with_(batch_size=4)
        )
        pipeline.add_stage(InferenceAsrNemoStage(model_name=model_name).with_(resources=Resources(gpus=gpus)))
        pipeline.add_stage(
            GetPairwiseWerStage(
                text_key="text",
                pred_text_key="pred_text",
                wer_key="wer",
            )
        )
        pipeline.add_stage(
            GetAudioDurationStage(
                audio_filepath_key="audio_filepath",
                duration_key="duration",
            )
        )
        pipeline.add_stage(
            PreserveByValueStage(
                input_value_key="wer",
                target_value=wer_threshold,
                operator="le",
            )
        )
        pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
        pipeline.add_stage(
            JsonlWriter(
                path=results_dir,
                write_kwargs={"force_ascii": False},
            )
        )

        logger.info("Running audio fleurs pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(executor_obj)
        run_time_taken = time.perf_counter() - run_start_time

        num_tasks_processed = len(output_tasks) if output_tasks else 0

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_tasks_processed} tasks")
        success = True

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_tasks_processed = 0
        success = False

    return {
        "params": {
            "executor": executor,
            "model_name": model_name,
            "lang": lang,
            "split": split,
            "wer_threshold": wer_threshold,
            "gpus": gpus,
            "benchmark_results_path": str(benchmark_results_path),
            "scratch_output_path": str(scratch_output_path),
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_tasks_processed": num_tasks_processed,
            "throughput_tasks_per_sec": num_tasks_processed / run_time_taken if run_time_taken > 0 else 0,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio Fleurs benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--scratch-output-path", required=True, help="Path to scratch output directory")
    parser.add_argument("--model-name", default="nvidia/stt_hy_fastconformer_hybrid_large_pc", help="ASR model name")
    parser.add_argument("--lang", default="hy_am", help="Language code")
    parser.add_argument("--split", default="dev", help="Dataset split to use")
    parser.add_argument("--wer-threshold", type=float, default=5.5, help="WER threshold for filtering")
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"], help="Executor to use")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()

    logger.info("=== Audio Fleurs Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results = {
        "params": vars(args),
        "metrics": {
            "is_success": False,
        },
        "tasks": [],
    }
    try:
        results.update(run_audio_fleurs_benchmark(**vars(args)))
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
