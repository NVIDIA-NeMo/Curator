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

"""Embedding generation benchmarking script.

This script runs embedding generation benchmarks with comprehensive metrics collection
using various executors and logs results to configured sinks.
"""

import argparse
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from utils import load_dataset_files, parse_partition_size, setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

_max_seq_length_map = {
    "sentence-transformers/all-MiniLM-L6-v2": 256,
    "google/embeddinggemma-300m": 2048,
}


class EmbeddingModelVariation(Enum):
    # SentenceTransformer (default for nightly benchmarks)
    SENTENCE_TRANSFORMER = "sentence_transformer"
    PYTORCH_MODEL = "pytorch_model"

    # VLLM variations
    VLLM_TEXT = "vllm_text"
    VLLM_TEXT_PRETOKENIZED = "vllm_text_pretokenized"


def create_embedding_generation_pipeline(  # noqa: PLR0913
    input_files: list[str],
    output_path: Path,
    model_identifier: str,
    model_inference_batch_size: int,
    model_variation: EmbeddingModelVariation,
    partition_size: str,
    use_id_generator: bool,
) -> Pipeline:
    if model_variation in {EmbeddingModelVariation.SENTENCE_TRANSFORMER, EmbeddingModelVariation.PYTORCH_MODEL}:
        from nemo_curator.stages.text.embedders import EmbeddingCreatorStage

        embedding_stage = EmbeddingCreatorStage(
            model_identifier=model_identifier,
            use_sentence_transformer=model_variation is EmbeddingModelVariation.SENTENCE_TRANSFORMER,
            text_field="text",
            max_seq_length=_max_seq_length_map[model_identifier],
            max_chars=None,
            embedding_pooling="mean_pooling",
            model_inference_batch_size=model_inference_batch_size,
        )
    elif model_variation in {EmbeddingModelVariation.VLLM_TEXT, EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED}:
        from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage

        embedding_stage = VLLMEmbeddingModelStage(
            model_identifier=model_identifier,
            text_field="text",
            pretokenize=model_variation is EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED,
        )
    else:
        msg = f"Unsupported model variation: {model_variation}"
        raise ValueError(msg)

    return Pipeline(
        name="embedding_generation_pipeline",
        stages=[
            ParquetReader(
                file_paths=input_files,
                **parse_partition_size(partition_size),
                fields=["text"],
                _generate_ids=use_id_generator,
            ),
            embedding_stage,
            ParquetWriter(
                path=str(output_path), fields=(["_curator_dedup_id"] if use_id_generator else []) + ["embeddings"]
            ),
        ],
    )


def run_embedding_generation_benchmark(  # noqa: PLR0913
    input_path: str,
    output_path: str,
    executor: str,
    dataset_size_gb: float,
    model_identifier: str,
    model_inference_batch_size: int,
    use_id_generator: bool,
    model_variation: EmbeddingModelVariation,
    partition_size: str,
) -> dict[str, Any]:
    """Run the embedding generation benchmark and collect comprehensive metrics."""

    input_path = Path(input_path)
    output_path = Path(output_path).absolute()

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting embedding generation benchmark")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Dataset size: {dataset_size_gb} GB")
    logger.info(f"Model: {model_identifier}")
    logger.info(f"Batch size: {model_inference_batch_size}")
    logger.info(f"Use ID generator: {use_id_generator}")
    logger.info(f"Model variation: {model_variation.name}")
    logger.info(f"Partition size: {partition_size}")
    logger.info(f"Executor: {executor}")

    run_start_time = time.perf_counter()

    logger.info("Running embedding generation pipeline...")

    input_files = load_dataset_files(input_path, dataset_size_gb)
    executor_obj = setup_executor(executor)

    pipeline = create_embedding_generation_pipeline(
        input_files=input_files,
        output_path=output_path,
        model_identifier=model_identifier,
        model_inference_batch_size=model_inference_batch_size,
        model_variation=model_variation,
        partition_size=partition_size,
        use_id_generator=use_id_generator,
    )

    if use_id_generator:
        from nemo_curator.stages.deduplication.id_generator import create_id_generator_actor, kill_id_generator_actor

        create_id_generator_actor()
        output_tasks = pipeline.run(executor_obj)
        kill_id_generator_actor()
    else:
        output_tasks = pipeline.run(executor_obj)
    num_documents_processed = sum(task._stage_perf[-1].num_items_processed for task in output_tasks)
    run_time_taken = time.perf_counter() - run_start_time
    throughput_docs_per_sec = num_documents_processed / run_time_taken if run_time_taken > 0 else 0

    return {
        "params": {
            "executor": executor,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "dataset_size_gb": dataset_size_gb,
            "model_identifier": model_identifier,
            "model_inference_batch_size": model_inference_batch_size,
            "model_variation": model_variation.value,
            "partition_size": partition_size,
            "use_id_generator": use_id_generator,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "num_documents_processed": num_documents_processed,
            "throughput_docs_per_sec": throughput_docs_per_sec,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding generation benchmark")
    # Paths
    parser.add_argument("--input-path", required=True, type=Path, help="Path to input data")
    parser.add_argument(
        "--benchmark-results-path",
        type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Output directory for embeddings",
    )
    # Executor
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"], help="Executor to use")
    parser.add_argument("--dataset-size-gb", type=float, required=True, help="Size of dataset to process in GB")
    parser.add_argument(
        "--model-identifier",
        required=True,
        help="Model identifier (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument("--model-inference-batch-size", type=int, default=1024, help="Batch size for model inference")
    parser.add_argument("--use-id-generator", action="store_true", help="If set, use the ID generator")
    parser.add_argument(
        "--partition-size",
        type=str,
        default="1fpp",
        help="Partition size: ends with 'fpp' for files_per_partition (e.g., '1fpp') or number for MB blocksize (e.g., '128')",
    )
    parser.add_argument(
        "--model-variation",
        type=str,
        default="sentence_transformer",
        choices=[v.value for v in EmbeddingModelVariation],
        help="Embedding model variation to use (default: sentence_transformer)",
    )

    args = parser.parse_args()

    logger.info("=== Embedding Generation Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")
    if args.model_identifier not in _max_seq_length_map:
        msg = f"Unknown model '{args.model_identifier}'. max_seq_length not set. "
        raise ValueError(msg)

    success_code = 1  # assume failure until benchmark succeeds

    # This dictionary will contain benchmark metadata and results, written to files for the benchmark framework to read.
    result_dict = {
        "params": vars(args),
        "metrics": {
            "is_success": False,
        },
        "tasks": [],
    }
    try:
        result_dict.update(
            run_embedding_generation_benchmark(
                input_path=args.input_path,
                output_path=args.output_path,
                executor=args.executor,
                dataset_size_gb=args.dataset_size_gb,
                model_identifier=args.model_identifier,
                model_inference_batch_size=args.model_inference_batch_size,
                use_id_generator=args.use_id_generator,
                model_variation=EmbeddingModelVariation(args.model_variation),
                partition_size=args.partition_size,
            )
        )
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        print(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
