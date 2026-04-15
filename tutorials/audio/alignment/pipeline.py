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

"""
MFA Forced Alignment Pipeline
==============================

Reads a JSONL audio manifest, runs Montreal Forced Aligner (MFA) on each
batch of entries, and writes the enriched manifest with TextGrid, RTTM,
and CTM file paths back to JSONL.

Example
-------
::

    python pipeline.py \\
        --input-manifest /data/manifest.jsonl \\
        --output-dir /data/aligned \\
        --acoustic-model english_us_arpa \\
        --dictionary english_us_arpa
"""

import argparse
import os
import shutil
import sys

from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alignment import MFAAlignmentStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    pipeline = Pipeline(
        name="mfa_alignment",
        description="Forced alignment with Montreal Forced Aligner",
    )

    pipeline.add_stage(
        MFAAlignmentStage(
            output_dir=args.output_dir,
            mfa_command=args.mfa_command,
            acoustic_model=args.acoustic_model,
            dictionary=args.dictionary,
            g2p_model=args.g2p_model,
            audio_filepath_key=args.audio_filepath_key,
            text_key=args.text_key,
            speaker_key=args.speaker_key,
            num_jobs=args.num_jobs,
            beam=args.beam,
            retry_beam=args.retry_beam,
            create_rttm=not args.no_rttm,
            create_ctm=not args.no_ctm,
            mfa_root_dir=args.mfa_root_dir,
        ).with_(batch_size=args.batch_size)
    )

    pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))

    result_dir = os.path.join(args.output_dir, "result")
    if args.clean and os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    elif not args.clean and os.path.exists(result_dir):
        msg = f"Result directory {result_dir} already exists. Use --clean to overwrite."
        raise ValueError(msg)

    pipeline.add_stage(
        JsonlWriter(
            path=result_dir,
            write_kwargs={"force_ascii": False},
        )
    )

    return pipeline


def main(args: argparse.Namespace) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    pipeline = create_pipeline(args)
    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    executor = (
        RayDataExecutor() if args.backend == "ray_data" else XennaExecutor()
    )

    logger.info("Starting MFA alignment pipeline...")
    pipeline.run(executor)
    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MFA forced alignment pipeline for audio manifests",
    )

    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
        help="Path to input JSONL manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory for TextGrids, RTTMs, CTMs, and result manifest",
    )
    parser.add_argument(
        "--mfa-command",
        type=str,
        default="mfa",
        help="Path to the mfa binary (default: mfa)",
    )
    parser.add_argument(
        "--mfa-root-dir",
        type=str,
        default="",
        help="MFA root directory with models (default: MFA_ROOT_DIR env or ~/.mfa)",
    )
    parser.add_argument(
        "--acoustic-model",
        type=str,
        default="english_us_arpa",
        help="MFA acoustic model name or path",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="english_us_arpa",
        help="MFA dictionary name or path",
    )
    parser.add_argument(
        "--g2p-model",
        type=str,
        default="english_us_arpa",
        help="MFA G2P model for OOV words (set empty to disable)",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="Key in manifest entries for transcript text",
    )
    parser.add_argument(
        "--audio-filepath-key",
        type=str,
        default="audio_filepath",
        help="Key in manifest entries for audio file path",
    )
    parser.add_argument(
        "--speaker-key",
        type=str,
        default="speaker",
        help="Key in manifest entries for speaker label",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=100,
        help="MFA beam size for alignment search",
    )
    parser.add_argument(
        "--retry-beam",
        type=int,
        default=400,
        help="MFA retry beam size for failed alignments",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=0,
        help="Number of parallel MFA jobs (0 = auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of audio files per MFA alignment batch",
    )
    parser.add_argument(
        "--no-rttm",
        action="store_true",
        help="Skip RTTM generation (only produce TextGrids)",
    )
    parser.add_argument(
        "--no-ctm",
        action="store_true",
        help="Skip CTM generation (only produce TextGrids)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing result directory before writing outputs",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["xenna", "ray_data"],
        default="ray_data",
        help="Execution backend: 'ray_data' (default) or 'xenna'",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()
    main(args)
