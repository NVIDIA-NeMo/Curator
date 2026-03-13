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

# ruff: noqa: PLR0913

"""Embedding generation benchmarking script.

Supports multiple embedding model backends:
- sentence_transformer: EmbeddingCreatorStage with SentenceTransformer (default)
- pytorch_model: EmbeddingCreatorStage with raw PyTorch model + custom pooling
- vllm_text: VLLMEmbeddingModelStage with text input
- vllm_text_pretokenized: VLLMEmbeddingModelStage with pretokenization

Usage:
    python embedding_generation_benchmark.py \
        --benchmark-results-path /tmp/results \
        --input-path ./data/embeddings \
        --model-identifier sentence-transformers/all-MiniLM-L6-v2 \
        --dataset-size-gb 1 \
        --model-variation sentence_transformer

    python embedding_generation_benchmark.py \
        --benchmark-results-path /tmp/results \
        --input-path ./data/embeddings \
        --model-identifier google/embeddinggemma-300m \
        --dataset-size-gb 1 \
        --model-variation vllm_text
"""

import argparse
import time
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from utils import load_dataset_files, setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter


class EmbeddingModelVariation(Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    PYTORCH_MODEL = "pytorch_model"
    VLLM_TEXT = "vllm_text"
    VLLM_TEXT_PRETOKENIZED = "vllm_text_pretokenized"


def _resolve_max_seq_length(model_identifier: str, max_seq_length: int | None) -> int:
    """Resolve and clamp max_seq_length to the model's actual limit.

    Different backends handle sequence length limits differently:
    - SentenceTransformers silently truncates to its own max_seq_length (often
      lower than max_position_embeddings, e.g. 256 for MiniLM).
    - HF AutoModel/TokenizerStage uses tokenizer.model_max_length, falling back
      to max_position_embeddings.
    - vLLM reads the sentence-transformers config and strictly enforces it,
      refusing to start if max_model_len exceeds it.

    To ensure apples-to-apples comparison across backends, we resolve the
    model's effective limit once and clamp any user-provided value to it.
    """
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    model_limit = tokenizer.model_max_length

    # Guard against HF bug where some models set model_max_length to max(int)
    if model_limit > 1e5:  # noqa: PLR2004
        config = AutoConfig.from_pretrained(model_identifier)
        model_limit = config.max_position_embeddings

    if max_seq_length is None:
        logger.info(f"Auto-resolved max_seq_length={model_limit} from model config")
        return model_limit

    if max_seq_length > model_limit:
        logger.warning(
            f"Requested max_seq_length={max_seq_length} exceeds model limit ({model_limit}). "
            f"Clamping to {model_limit} for consistent results across backends."
        )
        return model_limit

    return max_seq_length


def _create_embedding_stages(
    model_identifier: str,
    model_variation: EmbeddingModelVariation,
    model_inference_batch_size: int,
    max_seq_length: int | None,
    embedding_pooling: str,
) -> list:
    """Create the embedding stage(s) for the given model variation."""
    if model_variation in {EmbeddingModelVariation.SENTENCE_TRANSFORMER, EmbeddingModelVariation.PYTORCH_MODEL}:
        from nemo_curator.stages.text.embedders import EmbeddingCreatorStage

        use_sentence_transformer = model_variation == EmbeddingModelVariation.SENTENCE_TRANSFORMER
        return [
            EmbeddingCreatorStage(
                model_identifier=model_identifier,
                use_sentence_transformer=use_sentence_transformer,
                text_field="text",
                embedding_field="embeddings",
                model_inference_batch_size=model_inference_batch_size,
                sort_by_length=True,
                max_seq_length=max_seq_length,
                embedding_pooling=embedding_pooling,
            ),
        ]

    if model_variation in {EmbeddingModelVariation.VLLM_TEXT, EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED}:
        from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage

        # vLLM strictly enforces max_model_len from the model config, unlike
        # sentence-transformers which silently truncates.  Pass max_seq_length
        # through so vLLM knows the intended limit and won't error on inputs
        # that exceed the model's default max_position_embeddings.
        vllm_init_kwargs: dict[str, Any] = {}
        if max_seq_length is not None:
            vllm_init_kwargs["max_model_len"] = max_seq_length

        return [
            VLLMEmbeddingModelStage(
                model_identifier=model_identifier,
                text_field="text",
                embedding_field="embeddings",
                pretokenize=model_variation == EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED,
                vllm_init_kwargs=vllm_init_kwargs,
            ),
        ]

    msg = f"Unsupported model variation: {model_variation}"
    raise ValueError(msg)


def run_embedding_generation_benchmark(
    input_path: str,
    output_path: str,
    executor: str,
    dataset_size_gb: float,
    model_identifier: str,
    model_inference_batch_size: int,
    model_variation: str,
    max_seq_length: int | None,
    embedding_pooling: str,
    input_format: str = "parquet",
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> dict[str, Any]:
    """Run the embedding generation benchmark and collect comprehensive metrics."""
    variation = EmbeddingModelVariation(model_variation)
    max_seq_length = _resolve_max_seq_length(model_identifier, max_seq_length)
    input_path = Path(input_path)
    output_path = Path(output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting embedding generation benchmark")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Dataset size: {dataset_size_gb} GB")
    logger.info(f"Model: {model_identifier}")
    logger.info(f"Model variation: {variation.name}")
    logger.info(f"Batch size: {model_inference_batch_size}")
    logger.info(f"Embedding pooling: {embedding_pooling}")
    logger.info(f"Max sequence length: {max_seq_length}")
    logger.info(f"Input format: {input_format}")
    logger.info(f"Executor: {executor}")

    run_start_time = time.perf_counter()

    keep_ext = "jsonl" if input_format == "jsonl" else "parquet"
    input_files = load_dataset_files(input_path, dataset_size_gb, keep_extensions=keep_ext)
    executor_obj = setup_executor(executor)

    embedding_stages = _create_embedding_stages(
        model_identifier=model_identifier,
        model_variation=variation,
        model_inference_batch_size=model_inference_batch_size,
        max_seq_length=max_seq_length,
        embedding_pooling=embedding_pooling,
    )

    if input_format == "jsonl":
        reader = JsonlReader(file_paths=input_files, files_per_partition=1, fields=["text"], _generate_ids=False)
        writer = JsonlWriter(path=str(output_path), fields=["embeddings"])
    else:
        reader = ParquetReader(file_paths=input_files, files_per_partition=1, fields=["text"], _generate_ids=False)
        writer = ParquetWriter(path=str(output_path), fields=["embeddings"])

    pipeline = Pipeline(
        name="embedding_generation_pipeline",
        stages=[reader, *embedding_stages, writer],
    )
    output_tasks = pipeline.run(executor_obj)

    run_time_taken = time.perf_counter() - run_start_time

    num_documents_processed = sum(task._stage_perf[-1].num_items_processed for task in output_tasks)
    throughput_docs_per_sec = num_documents_processed / run_time_taken if run_time_taken > 0 else 0

    logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
    logger.success(f"Processed {num_documents_processed} documents")

    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "num_documents_processed": num_documents_processed,
            "throughput_docs_per_sec": throughput_docs_per_sec,
            "model_variation": variation.value,
            "max_seq_length": max_seq_length,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding generation benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--output-path", default="./embedding_generation_output", help="Output directory for results")
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"], help="Executor to use")
    parser.add_argument("--dataset-size-gb", type=float, required=True, help="Size of dataset to process in GB")
    parser.add_argument(
        "--model-identifier",
        required=True,
        help="Model identifier (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument("--model-inference-batch-size", type=int, default=1024, help="Batch size for model inference")
    parser.add_argument(
        "--model-variation",
        default="sentence_transformer",
        choices=[v.value for v in EmbeddingModelVariation],
        help="Embedding model backend (default: sentence_transformer)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Max sequence length for tokenization (auto-detected from model if not set)",
    )
    parser.add_argument(
        "--embedding-pooling",
        default="mean_pooling",
        choices=["mean_pooling", "last_token"],
        help="Pooling strategy for pytorch_model variation (ignored by sentence_transformer)",
    )
    parser.add_argument(
        "--input-format",
        default="parquet",
        choices=["parquet", "jsonl"],
        help="Input file format (default: parquet)",
    )

    args = parser.parse_args()

    logger.info("=== Embedding Generation Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    success_code = 1
    result_dict: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        result_dict.update(run_embedding_generation_benchmark(**vars(args)))
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
