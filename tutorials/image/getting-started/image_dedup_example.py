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
import time

from helper import download_webdataset

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.deduplication.removal import ImageDuplicatesRemovalStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter


def create_image_embedding_pipeline(args: argparse.Namespace) -> Pipeline:
    """Create image curation pipeline with file partitioning, image reading, embedding, deduplication."""

    # Define pipeline
    pipeline = Pipeline(name="image_curation", description="Curate images with embeddings and quality scoring")

    # Stage 0: Partition tar files for parallel processing
    pipeline.add_stage(FilePartitioningStage(
        file_paths=args.input_wds_dataset_dir,
        files_per_partition=args.tar_files_per_partition,
        file_extensions=[".tar"],
    ))

    # Stage 1: Read images from webdataset tar files (now runs in parallel)
    pipeline.add_stage(ImageReaderStage(
        dali_batch_size=args.batch_size,
        verbose=args.verbose,
        num_threads=16,  # More threads for I/O
        num_gpus_per_worker=0.25,
    ))

    # Stage 2: Generate CLIP embeddings for images
    pipeline.add_stage(ImageEmbeddingStage(
        model_dir=args.model_dir,
        num_gpus_per_worker=args.embedding_gpus_per_worker,
        model_inference_batch_size=args.embedding_batch_size,
        verbose=args.verbose,
    ))

    # Stage 3: Convert embeddings to document batch
    pipeline.add_stage(ConvertImageBatchToDocumentBatchStage(fields=["image_id", "embedding"]))

    # Stage 4: Save embeddings to parquet file
    pipeline.add_stage(ParquetWriter(
        path=args.embeddings_dir,
    ))

    return pipeline

def create_embedding_deduplication_workflow(args: argparse.Namespace) -> Pipeline:
    """Create image deduplication pipeline with embedding deduplication."""
    return SemanticDeduplicationWorkflow(
        input_path=args.embeddings_dir,
        output_path=args.removal_parquets_dir,
        id_field="image_id",
        embedding_field="embedding",
        n_clusters=100,
        eps=0.01,
        read_kwargs={"storage_options": {}},
        write_kwargs={"storage_options": {}},
        verbose=args.verbose,
    )

def create_image_deduplication_pipeline(args: argparse.Namespace) -> Pipeline:
    """Create image deduplication pipeline with image deduplication."""
    # Define pipeline
    pipeline = Pipeline(name="image_deduplication", description="Deduplicate images with image deduplication")

    # Stage 0: Partition tar files for parallel processing
    pipeline.add_stage(FilePartitioningStage(
        file_paths=args.input_wds_dataset_dir,
        files_per_partition=args.tar_files_per_partition,
        file_extensions=[".tar"],
    ))

    # Stage 1: Read images from webdataset tar files (now runs in parallel)
    pipeline.add_stage(ImageReaderStage(
        dali_batch_size=args.batch_size,
        verbose=args.verbose,
        num_threads=16,  # More threads for I/O
        num_gpus_per_worker=0.25,
    ))

    # Stage 2: Read removal list from parquet file and filter images
    pipeline.add_stage(ImageDuplicatesRemovalStage(
        removal_parquets_dir=args.removal_parquets_dir + "/duplicates",
        duplicate_id_field="id",
        verbose=args.verbose,
    ))

    # Stage 3: Write filtered images to disk
    pipeline.add_stage(ImageWriterStage(
        output_dir=args.output_dataset_dir,
        remove_image_data=True,
        verbose=args.verbose,
    ))

    return pipeline


def main(args: argparse.Namespace) -> None:
    """Main execution function for image curation pipeline."""

    ray_client = RayClient()
    ray_client.start()

    print("Starting image curation pipeline...")
    print(f"Input parquet file: {args.input_parquet}")
    print(f"Input webdataset directory: {args.input_wds_dataset_dir}")
    print(f"Output webdataset directory: {args.output_dataset_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Tar files per partition: {args.tar_files_per_partition}")
    print(f"Task batch size: {args.batch_size}")
    print("\n" + "=" * 50 + "\n")

    # Step 1: Download and prepare webdataset from parquet file
    if not args.skip_download:
        print("Step 1: Downloading webdataset from parquet file...")
        download_start = time.time()

        # Create output directory if it doesn't exist
        os.makedirs(args.input_wds_dataset_dir, exist_ok=True)

        # Download webdataset using helper function
        download_webdataset(
            parquet_path=args.input_parquet,
            output_dir=args.input_wds_dataset_dir,
            num_processes=args.download_processes,
            entries_per_tar=args.entries_per_tar,
        )

        download_time = time.time() - download_start
        print(f"✓ Dataset download completed in {download_time:.2f} seconds")
        print(f"✓ Webdataset saved to: {args.input_wds_dataset_dir}")
        print("\n" + "=" * 50 + "\n")
    else:
        print("Step 1: Skipping download (using existing dataset)")
        print(f"Using existing dataset at: {args.input_wds_dataset_dir}")
        print("\n" + "=" * 50 + "\n")

    # Step 2: Create and run curation pipelines
    # Step 2.1: Create image embedding pipeline
    print("Step 2.1: Running image embedding pipeline...")
    start_time = time.time()
    pipeline = create_image_embedding_pipeline(args)
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")
    pipeline.run()

    # Step 2.2: Create image deduplication pipeline (pairwise executor is XennaExecutor by default)
    print("Step 2.2: Running image deduplication pipeline...")
    start_time = time.time()
    pipeline = create_embedding_deduplication_workflow(args)
    print("\n" + "=" * 50 + "\n")
    pipeline.run()

    # Step 2.3: Create image deduplication pipeline
    print("Step 2.3: Running image deduplication pipeline...")
    start_time = time.time()
    pipeline = create_image_deduplication_pipeline(args)
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")
    pipeline.run()

    end_time = time.time()

    # Calculate and print execution time
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\nImage curation pipeline completed!")
    print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"\nProcessed dataset available at: {args.output_dataset_dir}")

    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image curation pipeline with embedding generation and quality scoring"
    )

    # Dataset arguments
    parser.add_argument(
        "--input-parquet",
        type=str,
        required=False,
        default=None,
        help="Path to input parquet file containing image URLs and metadata"
    )
    parser.add_argument(
        "--input-wds-dataset-dir",
        type=str,
        required=True,
        help="Directory to save the downloaded webdataset"
    )
    parser.add_argument(
        "--output-dataset-dir",
        type=str,
        required=True,
        help="Directory to save the resulting webdataset"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Directory to save the embeddings"
    )
    parser.add_argument(
        "--removal-parquets-dir",
        type=str,
        required=True,
        help="Directory to save the remove parquets"
    )
    parser.add_argument(
        "--download-processes",
        type=int,
        default=8,
        help="Number of parallel processes for downloading images"
    )
    parser.add_argument(
        "--entries-per-tar",
        type=int,
        default=1000,
        help="Number of entries per tar shard during download"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        default=False,
        help="Skip dataset download and use existing webdataset"
    )

    # Image reader arguments
    parser.add_argument(
        "--tar-files-per-partition",
        type=int,
        default=1,
        help="Number of tar files to process per partition (controls parallelism) for FilePartitioningStage"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of images per ImageBatch for the reader stage"
    )

    # General arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory containing all model weights"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging for all stages"
    )

    # Embedding stage arguments
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--embedding-gpus-per-worker",
        type=float,
        default=0.25,
        help="GPU allocation per worker for embedding generation"
    )

    args = parser.parse_args()
    main(args)
