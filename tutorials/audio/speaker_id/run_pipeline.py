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

"""Extract speaker embeddings from NeMo-tarred or JSONL audio data.

Counterpart of ``tutorials/audio/qwen_omni/run_pipeline.py`` but for local
speaker-model inference instead of remote LLM calls.

Usage (direct, multi-GPU friendly -- recommended)::

    CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \\
        --direct \\
        --input_manifest /data/manifest__OP_0..24_CL_.json \\
        --input_tar /data/audio__OP_0..24_CL_.tar \\
        --lhotse_mode nemo_tarred \\
        --output_dir /output/embeddings \\
        --batch_size 64

Merge per-shard files afterwards (optional)::

    python run_pipeline.py --merge --output_dir /output/embeddings

Cluster embeddings and annotate manifests (CPU-only, run after extraction)::

    python run_pipeline.py --cluster \\
        --input_manifest /data/manifest__OP_0..49_CL_.json \\
        --embedding_dir /output/embeddings \\
        --output_manifest_dir /output/output_manifests \\
        --threshold 0.292
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract speaker embeddings.")
    parser.add_argument(
        "--input_manifest",
        type=str,
        default="",
        help="NeMo JSONL manifest path (brace-expand pattern).",
    )
    parser.add_argument("--input_tar", type=str, default="", help="NeMo tarred audio path(s)")
    parser.add_argument("--output_dir", type=str, default="embeddings", help="Output directory for per-shard files")
    parser.add_argument(
        "--output_format",
        type=str,
        default="npz",
        choices=["npz", "pt"],
        help="Embedding output format",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/speakerverification_en_titanet_large",
        help="NeMo speaker model name",
    )
    parser.add_argument("--lhotse_mode", type=str, default="nemo_tarred", help="Lhotse mode: nemo_tarred, lhotse_shar, nemo_row")
    parser.add_argument("--shar_in_dir", type=str, default="", help="Lhotse Shar directory (for lhotse_shar mode)")
    parser.add_argument("--max_cuts", type=int, default=None, help="Max number of utterances to process (debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size per GPU (higher = more GPU memory)")
    parser.add_argument("--direct", action="store_true", help="Run directly without Ray (recommended for multi-GPU)")
    parser.add_argument("--merge", action="store_true", help="Merge per-shard files in output_dir into a single file")
    parser.add_argument("--cluster", action="store_true", help="Run AHC clustering + annotate manifests (CPU-only)")
    parser.add_argument("--embedding_dir", type=str, default="", help="Directory with embeddings_*.npz files (for --cluster)")
    parser.add_argument("--output_manifest_dir", type=str, default="output_manifests", help="Output directory for annotated manifests (for --cluster)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.292,
        help="Cosine-similarity threshold for AHC (default: TitaNet + mean-subtract @ EER from local benchmark; "
        "use 0.35–0.40 to merge less often)",
    )
    parser.add_argument("--linkage_method", type=str, default="average", help="Linkage method: average, complete, single")
    parser.add_argument("--shard_level_clustering", action="store_true", help="Cluster each shard independently instead of globally")
    parser.add_argument(
        "--embedding_normalization",
        type=str,
        default="center_global",
        choices=["none", "center_global", "external"],
        help="Pre-cosine embedding prep: center_global=subtract batch mean (default); "
        "external=cohort_mean.npy (+ optional std); none=legacy raw",
    )
    parser.add_argument(
        "--external_norm_mean_npy",
        type=str,
        default="",
        help="Path to (D,) cohort_mean.npy when --embedding_normalization external",
    )
    parser.add_argument(
        "--external_norm_std_npy",
        type=str,
        default="",
        help="Optional (D,) cohort_std.npy for external normalization",
    )
    parser.add_argument(
        "--norm_eps",
        type=float,
        default=1e-8,
        help="Epsilon added to std before division (external mode)",
    )
    parser.add_argument("--no-ray-local", action="store_true", help="Skip ray.init(local); use existing cluster")
    parser.add_argument(
        "--s3cfg",
        type=str,
        default="",
        help="Path to AIS/S3 config file and section, e.g. ~/.s3cfg[default].",
    )
    return parser.parse_args()


def _run_direct(args: argparse.Namespace) -> None:
    """Run embedding extraction directly on the current GPU, no Ray."""
    from nemo_curator.stages.audio.speaker_id.speaker_embedding_lhotse import SpeakerEmbeddingLhotseStage
    from nemo_curator.tasks import EmptyTask

    stage = SpeakerEmbeddingLhotseStage(
        model_name=args.model_name,
        lhotse_mode=args.lhotse_mode,
        input_manifest=args.input_manifest,
        input_tar=args.input_tar,
        shar_in_dir=getattr(args, "shar_in_dir", ""),
        output_path=args.output_dir,
        output_format=args.output_format,
        max_cuts=args.max_cuts,
        batch_size=args.batch_size,
    )
    stage.setup()
    stage.process(EmptyTask)


def _run_cluster(args: argparse.Namespace) -> None:
    """Run AHC clustering + annotate manifests with speaker labels."""
    from nemo_curator.stages.audio.speaker_id.speaker_clustering_and_scoring import SpeakerClusteringStage
    from nemo_curator.tasks import EmptyTask

    emb_dir = args.embedding_dir or args.output_dir
    stage = SpeakerClusteringStage(
        input_manifest=args.input_manifest,
        embedding_dir=emb_dir,
        output_manifest_dir=args.output_manifest_dir,
        threshold=args.threshold,
        linkage_method=args.linkage_method,
        shard_level_clustering=args.shard_level_clustering,
        embedding_normalization=args.embedding_normalization,
        external_norm_mean_npy=args.external_norm_mean_npy,
        external_norm_std_npy=args.external_norm_std_npy,
        norm_eps=args.norm_eps,
    )
    stage.setup()
    stage.process(EmptyTask)


def _run_merge(args: argparse.Namespace) -> None:
    """Merge per-shard embedding files in output_dir."""
    from nemo_curator.stages.audio.speaker_id.speaker_embedding_lhotse import merge_shard_embeddings

    merged = merge_shard_embeddings(args.output_dir, output_format=args.output_format)
    print(f"Merged file: {merged}")


def _run_ray_pipeline(args: argparse.Namespace) -> None:
    """Run via Ray Data pipeline (original path)."""
    import ray

    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.pipeline import Pipeline

    if not args.no_ray_local:
        ray.init(address="local", ignore_reinit_error=True)

    pipeline = Pipeline(name="speaker_embeddings")

    if args.lhotse_mode:
        from nemo_curator.stages.audio.speaker_id.speaker_embedding_lhotse import SpeakerEmbeddingLhotseStage

        pipeline.add_stage(
            SpeakerEmbeddingLhotseStage(
                model_name=args.model_name,
                lhotse_mode=args.lhotse_mode,
                input_manifest=args.input_manifest,
                input_tar=args.input_tar,
                shar_in_dir=args.shar_in_dir,
                output_path=args.output_dir,
                output_format=args.output_format,
                max_cuts=args.max_cuts,
                batch_size=args.batch_size,
            )
        )
    else:
        from nemo_curator.stages.audio.speaker_id.speaker_embedding_request import SpeakerEmbeddingRequestStage
        from nemo_curator.stages.text.io.reader.jsonl import JsonlReader

        pipeline.add_stage(JsonlReader(file_paths=args.input_manifest))
        pipeline.add_stage(
            SpeakerEmbeddingRequestStage(
                model_name=args.model_name,
                output_path=args.output_dir,
                output_format=args.output_format,
                input_tar=args.input_tar,
                s3cfg=args.s3cfg,
            )
        )

    pipeline.run(executor=RayDataExecutor())


def main() -> None:
    args = parse_args()

    if args.cluster:
        _run_cluster(args)
    elif args.merge:
        _run_merge(args)
    elif args.direct:
        _run_direct(args)
    else:
        _run_ray_pipeline(args)

    print(f"Done. Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
