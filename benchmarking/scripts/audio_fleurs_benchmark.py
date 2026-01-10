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

# ruff: noqa: S101  # Allow asserts in this script

import argparse
import sys
import traceback
from operator import le
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

# Import benchmarking utils which are currently only available directly from the Curator source tree.
# __file__ is expected to be <curator repo>/benchmarking/scripts/audio_fleurs_benchmark.py
_repo_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_dir))
from benchmarking.runner.utils import write_benchmark_results  # noqa: E402

_expected_num_results = 50


def read_jsonl(file_paths: Path | list[Path], executor: XennaExecutor) -> Pipeline:
    """Read jsonl files from one or more directories."""
    p = Pipeline(name="read", description="Read jsonl")
    # Build a list of directory path strings from Path object(s), as required by JsonlReader
    file_path_strs = [str(d) for d in (file_paths if isinstance(file_paths, list) else [file_paths])]
    p.add_stage(JsonlReader(file_paths=file_path_strs))
    return p.run(executor)


def assert_valid_pipeline(pipeline: Pipeline, expected_values: argparse.Namespace) -> None:
    """Assert that the pipeline is valid."""
    assert isinstance(pipeline, Pipeline)
    assert pipeline.name == "audio_inference"
    assert "Inference audio" in pipeline.description
    assert len(pipeline.stages) == 7  # noqa: PLR2004

    assert isinstance(pipeline.stages[0], CreateInitialManifestFleursStage)
    assert pipeline.stages[0].lang == expected_values.lang
    assert pipeline.stages[0].split == expected_values.split
    assert pipeline.stages[0].raw_data_dir.name == "fleurs"

    assert isinstance(pipeline.stages[1], InferenceAsrNemoStage)
    assert pipeline.stages[1].model_name == expected_values.model_name

    assert isinstance(pipeline.stages[2], GetPairwiseWerStage)
    assert pipeline.stages[2].wer_key == "wer"

    assert isinstance(pipeline.stages[3], GetAudioDurationStage)
    assert pipeline.stages[3].audio_filepath_key == "audio_filepath"

    assert isinstance(pipeline.stages[4], PreserveByValueStage)
    assert pipeline.stages[4].target_value == expected_values.wer_threshold
    assert pipeline.stages[4].operator == le


def run_audio_fleurs_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    success = False
    results_dir = args.benchmark_results_path / "results"
    try:
        # Ensure the results dir does not exist so that it will be created.
        # This ensures no preexisting files are present which would otherwise be treated as additional results.
        if results_dir.exists():
            msg = f"Result directory {results_dir} already exists."
            raise ValueError(msg)  # noqa: TRY301

        executor = XennaExecutor()
        pipeline = Pipeline(name="audio_inference", description="Inference audio and filter by WER threshold.")

        # Add stages
        # Add the composite stage that combines reading and downloading
        pipeline.add_stage(
            CreateInitialManifestFleursStage(
                lang=args.lang,
                split=args.split,
                raw_data_dir=args.scratch_output_path / "armenian/fleurs",
            ).with_(batch_size=4)
        )
        pipeline.add_stage(
            InferenceAsrNemoStage(model_name=args.model_name).with_(resources=Resources(gpus=args.gpus))
        )
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
                target_value=args.wer_threshold,
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

        assert_valid_pipeline(pipeline, args)

        results = pipeline.run(executor)
        assert len(results) == _expected_num_results
        predict = read_jsonl(results_dir, executor)
        assert len(predict) == _expected_num_results
        success = True

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error running audio fleurs benchmark: {e}\n{traceback.format_exc()}")

    return {
        # Populate the metrics dictionary with metrics to include in the report for this benchmark.
        # This also allows the framework to perform user-defined checks on the metrics to ensure perf requirements are met.
        "metrics": {
            "is_success": success,
        },
        "tasks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio Fleurs benchmark")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--scratch-output-path", required=True, help="Path to scratch output directory")
    parser.add_argument("--model-name", default="nvidia/stt_hy_fastconformer_hybrid_large_pc", help="ASR model name")
    parser.add_argument("--lang", default="hy_am", help="Language code")
    parser.add_argument("--split", default="dev", help="Dataset split to use")
    parser.add_argument("--wer-threshold", type=float, default=5.5, help="WER threshold for filtering")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    logger.info("=== Audio Fleurs Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    # This dictionary will contain benchmark metadata and results, written to files for the benchmark framework to read.
    # The dictionary must contain objects which can be serialized to JSON or pickle files.
    result_dict = {
        "params": vars(args).copy(),
        "metrics": {
            "is_success": False,
        },
        "tasks": [],
    }
    # Now that the args have been saved as JSON-serializable strings for the result_dict, convert paths to Path
    # objects for use in the benchmark script.
    args.benchmark_results_path = Path(args.benchmark_results_path)
    args.scratch_output_path = Path(args.scratch_output_path)

    try:
        result_dict.update(run_audio_fleurs_benchmark(args))
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
