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

"""Image pipeline benchmarking script.

This script runs an image curation pipeline benchmark with comprehensive
metrics collection and various executor support.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage


def create_image_curation_pipeline(args: argparse.Namespace) -> Pipeline:
    """Create image curation pipeline with file partitioning, image reading, embedding, and aesthetic scoring stages."""

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
        verbose=args.verbose,  # Force verbose to see debug info
        num_threads=args.reader_num_threads,  # More threads for I/O
        num_gpus_per_worker=args.reader_gpus_per_worker,
    ))

    # Stage 2: Generate CLIP embeddings for images
    pipeline.add_stage(ImageEmbeddingStage(
        model_dir=args.model_dir,
        num_gpus_per_worker=args.embedding_gpus_per_worker,
        model_inference_batch_size=args.embedding_batch_size,
        remove_image_data=False,
        verbose=args.verbose,
    ))

    # Stage 3: Generate aesthetic quality scores and filter
    pipeline.add_stage(ImageAestheticFilterStage(
        model_dir=args.model_dir,
        num_gpus_per_worker=args.aesthetic_gpus_per_worker,
        model_inference_batch_size=args.aesthetic_batch_size,
        score_threshold=args.aesthetic_threshold,
        verbose=args.verbose,
    ))

    # Stage 4: Write down to disk
    pipeline.add_stage(ImageWriterStage(
        output_dir=args.output_dataset_dir,
        images_per_tar=args.images_per_tar,
        remove_image_data=True,
        verbose=args.verbose,
    ))

    return pipeline


def run_image_pipeline_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the image pipeline benchmark and collect comprehensive metrics."""
    executor = setup_executor(args.executor)

    input_wds_dir = Path(args.input_wds_dataset_dir).absolute()
    output_dir = Path(args.output_dataset_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input webdataset directory: {input_wds_dir}")
    logger.info(f"Output dataset directory: {output_dir}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Tar files per partition: {args.tar_files_per_partition}")
    logger.info(f"Task batch size: {args.batch_size}")
    logger.info(f"Embedding batch size: {args.embedding_batch_size}")
    logger.info(f"Aesthetic threshold: {args.aesthetic_threshold}")
    logger.debug(f"Executor: {executor}")

    # Create pipeline
    pipeline = create_image_curation_pipeline(args)

    run_start_time = time.perf_counter()

    try:
        logger.info("Running image curation pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        # Calculate metrics from output tasks
        # Count total images processed (sum of images in each ImageBatch)
        num_images_processed = sum(
            len(task.data) for task in output_tasks if task.data is not None
        )

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_images_processed} images")
        logger.success(f"Output tasks: {len(output_tasks)}")
        success = True

    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_images_processed = 0
        success = False

    return {
        "params": {
            "executor": args.executor,
            "input_wds_dataset_dir": str(input_wds_dir),
            "output_dataset_dir": str(output_dir),
            "benchmark_results_path": str(args.benchmark_results_path),
            "model_dir": args.model_dir,
            "tar_files_per_partition": args.tar_files_per_partition,
            "batch_size": args.batch_size,
            "embedding_batch_size": args.embedding_batch_size,
            "embedding_gpus_per_worker": args.embedding_gpus_per_worker,
            "aesthetic_batch_size": args.aesthetic_batch_size,
            "aesthetic_gpus_per_worker": args.aesthetic_gpus_per_worker,
            "aesthetic_threshold": args.aesthetic_threshold,
            "images_per_tar": args.images_per_tar,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_images_processed": num_images_processed,
            "num_output_tasks": len(output_tasks),
            "throughput_images_per_sec": num_images_processed / run_time_taken if run_time_taken > 0 else 0,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    """Main entry point for image pipeline benchmark."""
    ray_client = RayClient()
    ray_client.start()

    parser = argparse.ArgumentParser(
        description="Image curation pipeline benchmark with embedding generation and quality scoring"
    )

    # Benchmark-specific arguments
    parser.add_argument(
        "--benchmark-results-path",
        type=Path,
        required=True,
        help="Path to write benchmark results",
    )
    parser.add_argument(
        "--executor",
        default="xenna",
        choices=["xenna", "ray_data"],
        help="Executor to use for pipeline execution",
    )

    # Dataset arguments
    parser.add_argument(
        "--input-wds-dataset-dir",
        type=str,
        required=True,
        help="Directory containing the input webdataset"
    )
    parser.add_argument(
        "--output-dataset-dir",
        type=str,
        required=True,
        help="Directory to save the resulting webdataset"
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

    # Image reader arguments
    parser.add_argument(
        "--reader-num-threads",
        type=int,
        default=16,
        help="Number of threads for image reading"
    )
    parser.add_argument(
        "--reader-gpus-per-worker",
        type=float,
        default=0.25,
        help="GPU allocation per worker for image reading"
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

    # Aesthetic scoring arguments
    parser.add_argument(
        "--aesthetic-batch-size",
        type=int,
        default=32,
        help="Batch size for aesthetic scoring"
    )
    parser.add_argument(
        "--aesthetic-gpus-per-worker",
        type=float,
        default=0.25,
        help="GPU allocation per worker for aesthetic scoring"
    )
    parser.add_argument(
        "--aesthetic-threshold",
        type=float,
        default=0.5,
        help="Aesthetic score threshold for filtering (images below this score will be filtered out)"
    )

    # Output dataset arguments
    parser.add_argument(
        "--images-per-tar",
        type=int,
        default=100,
        help="Number of images per tar file in output dataset"
    )

    args = parser.parse_args()

    logger.info("=== Image Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_image_pipeline_benchmark(args)

    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        print(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        results = {
            "params": vars(args),
            "metrics": {
                "is_success": False,
            },
            "tasks": [],
        }
    finally:
        write_benchmark_results(results, args.benchmark_results_path)
        ray_client.stop()

    # Return proper exit code based on success
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
