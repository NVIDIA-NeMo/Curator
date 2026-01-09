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

# ruff: noqa: S101, PLR2004

import argparse
from operator import le
from pathlib import Path

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
    assert len(pipeline.stages) == 7

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


def run_audio_fleurs_benchmark(args: argparse.Namespace) -> int:
    if args.benchmark_results_path.exists():
        msg = f"Result directory {args.benchmark_results_path} already exists."
        raise ValueError(msg)
    executor = XennaExecutor()

    # Define pipeline
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
    pipeline.add_stage(InferenceAsrNemoStage(model_name=args.model_name).with_(resources=Resources(gpus=args.gpus)))
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
            path=args.benchmark_results_path,
            write_kwargs={"force_ascii": False},
        )
    )

    assert_valid_pipeline(pipeline, args)

    write_result = pipeline.run(executor)
    assert len(write_result) == 50

    predict = read_jsonl(args.benchmark_results_path, executor)
    assert len(predict) == 50

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio Fleurs benchmark")
    parser.add_argument("--benchmark-results-path", required=True, type=Path, help="Path to benchmark results")
    parser.add_argument("--scratch-output-path", required=True, type=Path, help="Path to scratch output directory")
    parser.add_argument("--model-name", default="nvidia/stt_hy_fastconformer_hybrid_large_pc", help="ASR model name")
    parser.add_argument("--lang", default="hy_am", help="Language code")
    parser.add_argument("--split", default="dev", help="Dataset split to use")
    parser.add_argument("--wer-threshold", type=float, default=5.5, help="WER threshold for filtering")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    logger.info("=== Audio Fleurs Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    return run_audio_fleurs_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
