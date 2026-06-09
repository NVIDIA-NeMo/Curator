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

"""End-to-end speaker-ID + MOS pipeline over NeMo tarred shards.

Chains the stages added in this branch into a single :class:`Pipeline`:

1. ``SpeakerEmbeddingLhotseStage`` -- stream NeMo tarred shards through a
   NeMo speaker model, writing per-shard ``embeddings_<id>.npz`` files.
2. ``SpeakerClusteringStage`` -- globally cluster those embeddings (AHC) and
   write annotated manifests with ``speaker_label`` and ``confidence_score``.
3. (optional, ``--utmos``) ``GetUtmosv2ScoreStage`` -- read the annotated
   manifests back and add a ``utmosv2_score`` MOS rating per utterance.

The first two stages are ``_EmptyTask -> _EmptyTask`` and communicate via
disk: each emits its output task only *after* ``process()`` has finished
writing, so a downstream stage never observes a partial directory -- the
chain is correct even under streaming execution.

UTMOSv2 needs the actual audio.  For tarred datasets the manifest's
``audio_filepath`` is an in-tar member name, so pass ``--audio-root`` pointing
at a directory where those files are resolvable; otherwise run without
``--utmos`` and the speaker-ID phase still completes on its own.

Example::

    python pipeline.py \\
        --input-manifest "/data/manifest__OP_0..9_CL_.json" \\
        --input-tar      "/data/audio__OP_0..9_CL_.tar" \\
        --output-dir     /data/speaker_id_out \\
        --threshold      0.292 \\
        --gpus           1.0
"""

import argparse
import os
import shutil
import sys

from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.metrics.utmosv2_score import GetUtmosv2ScoreStage
from nemo_curator.stages.audio.speaker_id import SpeakerClusteringStage, SpeakerEmbeddingLhotseStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter


def create_speaker_id_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the speaker-ID (+ optional UTMOSv2) pipeline."""
    embeddings_dir = os.path.join(args.output_dir, "embeddings")
    manifests_dir = os.path.join(args.output_dir, "annotated_manifests")

    pipeline = Pipeline(
        name="speaker_id",
        description="Speaker embedding extraction, clustering, and optional UTMOSv2 scoring.",
    )

    # 1. Extract speaker embeddings from the tarred shards (GPU).
    pipeline.add_stage(
        SpeakerEmbeddingLhotseStage(
            model_name=args.model_name,
            lhotse_mode="nemo_tarred",
            input_manifest=args.input_manifest,
            input_tar=args.input_tar,
            output_path=embeddings_dir,
            output_format="npz",
            max_cuts=args.max_cuts,
        ).with_(batch_size=args.batch_size, resources=Resources(cpus=1.0, gpus=args.gpus))
    )

    # 2. Cluster the embeddings and annotate manifests (CPU-only).
    pipeline.add_stage(
        SpeakerClusteringStage(
            input_manifest=args.input_manifest,
            embedding_dir=embeddings_dir,
            output_manifest_dir=manifests_dir,
            threshold=args.threshold,
            linkage_method=args.linkage_method,
            shard_level_clustering=args.shard_level_clustering,
        )
    )

    # 3. (optional) Read the annotated manifests back and add a MOS score.
    if args.utmos:
        pipeline.add_stage(JsonlReader(file_paths=manifests_dir))
        pipeline.add_stage(
            GetUtmosv2ScoreStage(
                audio_root=args.audio_root,
                score_key="utmosv2_score",
            ).with_(resources=Resources(gpus=args.gpus))
        )
        result_dir = os.path.join(args.output_dir, "scored_manifests")
        if args.clean and os.path.isdir(result_dir):
            shutil.rmtree(result_dir)
        pipeline.add_stage(JsonlWriter(path=result_dir, write_kwargs={"force_ascii": False}))

    return pipeline


def main(args: argparse.Namespace) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    if args.utmos and not args.audio_root:
        logger.warning(
            "--utmos is set without --audio-root; UTMOSv2 can only score utterances "
            "whose 'audio_filepath' resolves to a readable file."
        )

    pipeline = create_speaker_id_pipeline(args)

    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    executor = RayDataExecutor() if args.backend == "ray_data" else XennaExecutor()

    logger.info("Starting pipeline execution...")
    pipeline.run(executor)

    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Input (NeMo tarred shards) -- supports brace-expand patterns (_OP_0..9_CL_).
    parser.add_argument("--input-manifest", type=str, required=True, help="NeMo manifest path / brace pattern")
    parser.add_argument("--input-tar", type=str, required=True, help="NeMo tarred-audio path / brace pattern")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory for all outputs")
    # Embedding stage
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvidia/speakerverification_en_titanet_large",
        help="NeMo EncDecSpeakerLabelModel name",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding inference batch size")
    parser.add_argument("--gpus", type=float, default=1.0, help="GPUs to request for GPU stages (0 for CPU)")
    parser.add_argument("--max-cuts", type=int, default=None, help="Cap total utterances processed (debug)")
    # Clustering stage
    parser.add_argument("--threshold", type=float, default=0.292, help="Cosine-similarity cutoff for AHC")
    parser.add_argument("--linkage-method", type=str, choices=["average", "complete", "single"], default="average")
    parser.add_argument(
        "--shard-level-clustering",
        action="store_true",
        help="Cluster each shard independently instead of globally",
    )
    # UTMOSv2 scoring (optional)
    parser.add_argument("--utmos", action="store_true", help="Append a UTMOSv2 MOS-scoring phase")
    parser.add_argument(
        "--audio-root",
        type=str,
        default="",
        help="Root prepended to relative 'audio_filepath' so UTMOSv2 can read the audio",
    )
    # General
    parser.add_argument("--clean", action="store_true", help="Delete existing scored output before writing")
    parser.add_argument("--backend", type=str, choices=["xenna", "ray_data"], default="xenna")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()
    main(args)
