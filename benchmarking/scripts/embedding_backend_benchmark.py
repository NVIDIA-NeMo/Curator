#!/usr/bin/env python3
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

"""Benchmark script to compare embedding backends (transformers vs vllm).

Usage:
    python embedding_backend_benchmark.py -d /path/to/data --model google/embeddinggemma-300m
    python embedding_backend_benchmark.py -d /path/to/data --model intfloat/e5-small-v2 --backend vllm
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

logger = logging.getLogger(__name__)


def build_file_list(base_path: str, num_files: int) -> list[str]:
    if not os.path.isdir(base_path):
        msg = f"Base path does not exist or is not a directory: {base_path}"
        raise FileNotFoundError(msg)
    names = sorted([f for f in os.listdir(base_path) if f.endswith(".parquet")])
    selected = names if (num_files is None or num_files <= 0) else names[:num_files]
    file_paths = [os.path.join(base_path, n) for n in selected]
    if not file_paths:
        msg = f"No parquet files found under {base_path}"
        raise RuntimeError(msg)
    return file_paths


def get_num_inputs(file_list: list[str]) -> int:
    return sum(len(pd.read_parquet(file)) for file in file_list)


def get_executor(args: argparse.Namespace) -> RayDataExecutor | XennaExecutor:
    name = (args.executor or "xenna").lower()
    if name in ("ray", "raydata", "ray_data"):
        return RayDataExecutor()
    if name == "xenna":
        return XennaExecutor(
            {"autoscale_interval_s": args.autoscale_interval_s, "execution_mode": args.execution_mode}
        )
    msg = "executor must be 'ray_data' or 'xenna'"
    raise ValueError(msg)


def run_embedding_pipeline(  # noqa: PLR0913
    base_path: str,
    executor: RayDataExecutor | XennaExecutor,
    model_identifier: str,
    backend: str = "transformers",
    num_files: int = -1,
    out_path: str = "embedding_output.parquet",
    text_field: str = "text",
    files_per_partition: int = 1,
    # Transformers-specific options
    model_inference_batch_size: int = 256,
    embedding_pooling: str = "mean_pooling",
    # VLLM-specific options
    gpu_memory_utilization: float = 0.9,
) -> dict[str, str | float | int]:
    if not os.path.isdir(base_path):
        msg = f"Base path does not exist or is not a directory: {base_path}"
        raise FileNotFoundError(msg)

    file_paths = build_file_list(base_path, num_files)
    num_selected = len(file_paths)

    exec_name = type(executor).__name__

    logger.info(
        "Config | backend=%s | model=%s | executor=%s | files=%d (requested=%s) | out=%s",
        backend,
        model_identifier,
        exec_name,
        num_selected,
        str(num_files),
        out_path,
    )

    pipeline = Pipeline(name=f"embedding_{backend}")

    # Reader stage
    pipeline.add_stage(
        ParquetReader(
            file_paths=file_paths,
            files_per_partition=files_per_partition,
            fields=[text_field],
            _generate_ids=False,
        )
    )

    # Embedding stage
    pipeline.add_stage(
        EmbeddingCreatorStage(
            model_identifier=model_identifier,
            text_field=text_field,
            embedding_field="embeddings",
            backend=backend,
            # Transformers-specific options
            model_inference_batch_size=model_inference_batch_size,
            embedding_pooling=embedding_pooling,
            autocast=True,
            sort_by_length=True,
            # VLLM-specific options
            gpu_memory_utilization=gpu_memory_utilization,
            num_gpus=1,
            num_cpus=8,
        )
    )

    # Writer stage
    pipeline.add_stage(ParquetWriter(path=out_path, fields=["embeddings"]))

    logger.info("Starting pipeline execution with %s backend...", backend.upper())
    t0 = time.perf_counter()
    pipeline.run(executor)
    elapsed_s = time.perf_counter() - t0
    num_docs = get_num_inputs(file_paths)
    throughput = num_docs / elapsed_s if elapsed_s > 0 else 0

    logger.info(
        "Finished | elapsed=%.2fs | backend=%s | model=%s | executor=%s | files=%d | fpp=%d | docs=%d | throughput=%.2f docs/s | out=%s",
        elapsed_s,
        backend,
        model_identifier,
        exec_name,
        num_selected,
        files_per_partition,
        num_docs,
        throughput,
        out_path,
    )

    return {
        "backend": backend,
        "elapsed_s": elapsed_s,
        "num_docs": num_docs,
        "throughput": throughput,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run Embedding pipeline comparing backends (transformers vs vllm).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data paths
    parser.add_argument("-d", "--data-path", required=True, help="Directory containing input Parquet files")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of files to process; -1 means all")
    parser.add_argument("-o", "--out-path", default="embedding_output.parquet", help="Output Parquet path")

    # Executor options
    parser.add_argument("-e", "--executor", type=str, default="xenna", choices=["xenna", "ray_data"])
    parser.add_argument("--autoscale-interval-s", type=int, default=5)
    parser.add_argument("-em", "--execution-mode", type=str, default="batch", choices=["streaming", "batch"])
    parser.add_argument("-fpp", "--files-per-partition", type=int, default=1, help="Files per partition")

    # Model options
    parser.add_argument("--model", type=str, default="google/embeddinggemma-300m", help="HuggingFace model identifier")
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Embedding backend to use",
    )
    parser.add_argument("--text-field", type=str, default="text", help="Field containing text to embed")

    # Transformers-specific options
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for transformers backend")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean_pooling",
        choices=["mean_pooling", "last_token"],
        help="Pooling strategy for transformers backend",
    )

    # VLLM-specific options
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5, help="GPU memory utilization for VLLM")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Trust remote code")

    args = parser.parse_args()
    client = RayClient()
    client.start()

    executor = get_executor(args)

    result = run_embedding_pipeline(
        base_path=args.data_path,
        executor=executor,
        model_identifier=args.model,
        backend=args.backend,
        num_files=args.num_files,
        out_path=args.out_path,
        text_field=args.text_field,
        model_inference_batch_size=args.batch_size,
        embedding_pooling=args.pooling,
        gpu_memory_utilization=args.gpu_memory_utilization,
        files_per_partition=args.files_per_partition,
    )

    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info("Backend: %s", result["backend"])
    logger.info("Documents processed: %d", result["num_docs"])
    logger.info("Total time: %.2f seconds", result["elapsed_s"])
    logger.info("Throughput: %.2f docs/second", result["throughput"])
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main() or 0)
