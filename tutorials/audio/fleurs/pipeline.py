# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os
import shutil

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.writer import JsonlWriter


def create_audio_pipeline(args: argparse.Namespace) -> Pipeline:
    # Define pipeline
    pipeline = Pipeline(name="audio_inference", description="Inference audio and filter by WER threshold.")

    # Add stages
    # Add the composite stage that combines reading and downloading
    pipeline.add_stage(
        CreateInitialManifestFleursStage(
            lang=args.lang,
            split=args.split,
            raw_data_dir=args.raw_data_dir,
        ).with_(batch_size=4)
    )
    pipeline.add_stage(InferenceAsrNemoStage(model_name=args.model_name).with_(resources=Resources(gpus=1.0)))
    pipeline.add_stage(GetPairwiseWerStage(text_key="text", pred_text_key="pred_text", wer_key="wer"))
    pipeline.add_stage(GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration"))
    pipeline.add_stage(PreserveByValueStage(input_value_key="wer", target_value=args.wer_threshold, operator="le"))
    pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
    result_dir = os.path.join(args.raw_data_dir, "result")
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)  # clean up resulting folder
    pipeline.add_stage(
        JsonlWriter(
            path=result_dir,
            write_kwargs={"force_ascii": False},
        )
    )
    return pipeline


def main(args: argparse.Namespace) -> None:
    """
    Prepare FLEURS dataset, run ASR inference and filer by WER threshold.
    """
    pipeline = create_audio_pipeline(args)

    # Print pipeline description
    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    logger.info("Starting pipeline execution...")
    pipeline.run(executor)

    # Print results
    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to store processed data")
    parser.add_argument(
        "--model_name", type=str, default="nvidia/stt_hy_fastconformer_hybrid_large_pc", help="NeMo model name"
    )
    parser.add_argument("--lang", type=str, default="hy_am", help="Language name ")
    parser.add_argument("--split", type=str, default="dev", help="Split name, usually {train, dev, test}")
    parser.add_argument("--wer_threshold", type=float, default=75.0, help="Limit the number of videos to read")
    args = parser.parse_args()
    main(args)
