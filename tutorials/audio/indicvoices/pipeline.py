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

"""IndicVoices ASR stage-1 pipeline: extract -> split-aware manifest writing.

Example:
    python tutorials/audio/indicvoices/pipeline.py \
        --raw_data_dir /data/asr/gu/indic_voices \
        --output_dir   /data/asr/gu/indic_voices_curated \
        --langs gu --split_dir_pattern "{split}" --clean
"""

import argparse
import shutil
import sys

from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.asr.datasets.indicvoices import IndicVoicesHandler
from nemo_curator.stages.audio.asr.io.split_manifest_writer import SplitAwareManifestWriter


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    pipeline = Pipeline(name="indicvoices_asr", description="IndicVoices extract + split-aware manifests")
    pipeline.add_stage(
        IndicVoicesHandler(
            raw_data_dir=args.raw_data_dir,
            output_dir=args.output_dir,
            langs=args.langs,
            native_splits=args.native_splits,
            split_dir_pattern=args.split_dir_pattern,
            dev_fraction=args.dev_fraction,
            extraction_workers=args.extraction_workers,
            skip_untar=args.skip_untar,
        )
    )
    pipeline.add_stage(
        SplitAwareManifestWriter(
            output_dir=args.output_dir,
            langs=args.langs,
            splits=["train", "dev", "test"],
        )
    )
    return pipeline


def main(args: argparse.Namespace) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    if args.clean:
        for lang in args.langs:
            shutil.rmtree(f"{args.output_dir}/{lang}", ignore_errors=True)

    pipeline = create_pipeline(args)
    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    executor = RayDataExecutor() if args.backend == "ray_data" else XennaExecutor()
    logger.info("Starting pipeline execution...")
    pipeline.run(executor)
    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Root containing the raw arrow split dirs")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination root for audio + manifests")
    parser.add_argument("--langs", type=str, nargs="+", default=["gu"], help="Languages to process")
    parser.add_argument(
        "--native_splits", type=str, nargs="+", default=["train", "valid"], help="Native splits to read"
    )
    parser.add_argument(
        "--split_dir_pattern",
        type=str,
        default="{lang}_{split}",
        help="Per-split arrow dir name pattern under raw_data_dir (e.g. '{split}' or '{lang}_{split}')",
    )
    parser.add_argument("--dev_fraction", type=float, default=0.6, help="Fraction of 'valid' routed to dev")
    parser.add_argument("--extraction_workers", type=int, default=10, help="Internal joblib workers for extraction")
    parser.add_argument("--skip_untar", action="store_true", help="Reuse already-extracted WAVs when present")
    parser.add_argument("--clean", action="store_true", help="Remove existing per-language output before running")
    parser.add_argument("--backend", type=str, choices=["xenna", "ray_data"], default="xenna")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
