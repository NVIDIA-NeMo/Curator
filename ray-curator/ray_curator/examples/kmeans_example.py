#!/usr/bin/env python3
"""Example pipeline using FilePartitioningStage and KMeansStage."""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from ray_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.deduplication.semantic import IdentifySemanticDuplicatesStage, KMeansStage, PairwiseStage
from ray_curator.stages.io.reader import JsonlReader
from ray_curator.stages.io.writer import JsonlWriter, ParquetWriter
from ray_curator.stages.text.embedders import DistributedEmbeddingModelStage


def main() -> int:
    """Main function to run the KMeans pipeline."""
    logger.info("Starting KMeans clustering pipeline...")
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"ID field: {args.id_field}")
    logger.info(f"Text field: {args.text_field}")
    logger.info(f"Embedding field: {args.embedding_field}")
    logger.info(f"Number of clusters: {args.n_clusters}")
    logger.info(f"Cosine similarity threshold: {args.cosine_sim_threshold}")

    embedding_output_path = str(Path(args.output_path) / "embeddings")
    kmeans_output_path = str(Path(args.output_path) / "kmeans")
    pairwise_input_path = str(Path(args.output_path) / "pairwise")
    duplicates_output_path = str(Path(args.output_path) / "duplicates")

    if args.executor == "xenna":
        from ray_curator.backends.xenna import XennaExecutor

        main_executor = XennaExecutor()
    else:
        from ray_curator.backends.experimental.ray_data import RayDataExecutor

        main_executor = RayDataExecutor()

    time_taken_dict = {}

    # Embedding pipeline
    t0 = time.perf_counter()

    if args.file_type == "jsonl":
        writer = JsonlWriter(output_dir=embedding_output_path)
    else:
        writer = ParquetWriter(output_dir=embedding_output_path)

    embedding_pipeline = Pipeline(
        name="embedding_pipeline",
        description="Pipeline for embedding documents",
        stages=[
            JsonlReader(
                file_paths=args.input_path,
                files_per_partition=1,
            ),
            DistributedEmbeddingModelStage(
                model_identifier=args.model_identifier,
                text_field=args.text_field,
                embedding_field=args.embedding_field,
            ),
            writer,
        ],
    )
    embedding_results = embedding_pipeline.run(main_executor)
    time_taken_dict["embedding"] = time.perf_counter() - t0
    for task_out in embedding_results:
        logger.info(task_out)

    # KMeans pipeline
    kmeans_pipeline = Pipeline(
        name="kmeans_clustering_pipeline",
        description="Pipeline for K-means clustering on document embeddings",
        stages=[
            KMeansStage(
                input_path=embedding_output_path,
                id_field=args.id_field,
                embedding_field=args.embedding_field,
                output_path=kmeans_output_path,
                n_clusters=args.n_clusters,
                input_filetype=args.file_type,
                input_storage_options={},
                output_storage_options={},
                input_file_limit=None,
            )
        ],
    )
    kmeans_executor = RayActorPoolExecutor(
        config={
            "reserved_cpus": 1.0,  # Reserve some CPUs for system overhead
            "reserved_gpus": 0.0,
        }
    )
    t1 = time.perf_counter()
    kmeans_results = kmeans_pipeline.run(kmeans_executor)
    time_taken_dict["kmeans"] = time.perf_counter() - t1
    for task_out in kmeans_results:
        logger.info(task_out)

    # Pairwise pipeline
    pairwise_pipeline = Pipeline(
        name="pairwise_clustering_pipeline",
        description="Pipeline for pairwise clustering on document embeddings",
        stages=[
            PairwiseStage(
                id_field=args.id_field,
                embedding_field=args.embedding_field,
                input_path=kmeans_output_path,
                output_path=pairwise_input_path,
            )
        ],
    )

    if args.cosine_sim_threshold is not None:
        from ray_curator.stages.resources import Resources

        duplicate_identify_stage = IdentifySemanticDuplicatesStage(
            threshold=args.cosine_sim_threshold,
            output_path=duplicates_output_path,
            verbose=True,
        )

        if args.use_gpu_for_duplicates:
            duplicate_identify_stage.with_(resources=Resources(cpus=1.0, gpus=1.0))
        else:
            duplicate_identify_stage.with_(resources=Resources(cpus=1.0, gpus=0.0))
        pairwise_pipeline.add_stage(duplicate_identify_stage)

    t2 = time.perf_counter()
    pairwise_results = pairwise_pipeline.run(main_executor)
    time_taken_dict["pairwise"] = time.perf_counter() - t2
    for task_out in pairwise_results:
        logger.info(task_out)

    logger.success(f"Time taken: {[(k, f'{v:.2f}s') for k, v in time_taken_dict.items()]}")
    logger.success(f"Time taken: {[(k, f'{v:.2f}s') for k, v in time_taken_dict.items()]}")
    for path in [
        args.input_path,
        embedding_output_path,
        kmeans_output_path,
        pairwise_input_path,
        duplicates_output_path,
    ]:
        if path == args.input_path:
            df = pd.concat(
                [
                    pd.read_json(os.path.join(path, f), lines=True, orient="records")
                    for f in os.listdir(path)
                    if f.endswith(".jsonl")
                ]
            )
        else:
            df = pd.read_parquet(path)

        logger.info(f"{path.replace(args.output_path, '')} {list(df.columns)} with {len(df):,} rows")

    return 0


if __name__ == "__main__":
    logger.info("Starting KMeans clustering pipeline...")
    parser = argparse.ArgumentParser(description="Run a pipeline with FilePartitioningStage and KMeansStage")

    # Required arguments
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input parquet files or directory containing parquet files",
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to output directory for clustered results"
    )
    parser.add_argument("--id-field", type=str, default="id", help="Name of the ID field in the input files")
    parser.add_argument("--text-field", type=str, default="text", help="Name of the text field in the input files")
    parser.add_argument(
        "--model-identifier", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Model identifier"
    )
    parser.add_argument(
        "--embedding-field", type=str, default="embeddings", help="Name of the embedding field in the parquet files"
    )
    parser.add_argument("--n-clusters", type=int, default=10, help="Number of clusters to create")

    parser.add_argument("--file-type", type=str, default="parquet", help="File type to include (default: 'parquet')")
    parser.add_argument("--executor", type=str, default="ray", help="Executor to use (default: 'ray_data') or 'xenna'")

    # New argument for identify duplicates stage
    parser.add_argument(
        "--cosine-sim-threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for identifying duplicates (e.g., 0.8). If not provided, identify duplicates stage is skipped.",
    )
    parser.add_argument(
        "--use-gpu-for-duplicates",
        action="store_true",
        help="Use GPU for duplicates stage (default: False)",
    )

    args = parser.parse_args()

    sys.exit(main())
