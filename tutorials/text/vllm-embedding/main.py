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

"""Embed a text column of Parquet files with vLLM, one model replica per GPU."""

import argparse
import json
import time

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient()
    ray_client.start()

    embedder = VLLMEmbeddingModelStage(
        model_identifier=args.model,
        text_field=args.text_field,
        text_prefix=args.text_prefix,
        metadata_fields=args.keep_fields,
        embedding_output_dtype=args.embedding_dtype,
        cache_dir=args.cache_dir,
        vllm_init_kwargs={
            # Large engine steps keep the GPU busy; vLLM's defaults are sized for chat, not bulk embedding.
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            # Replicas that share a GPU must share its memory too.
            "gpu_memory_utilization": 0.8 / args.replicas_per_gpu,
            **({"max_model_len": args.max_model_len} if args.max_model_len else {}),
            **args.vllm_init_kwargs,
        },
    )
    # One vLLM engine loop rarely keeps a large GPU busy: its CPU-side work leaves the GPU idle between steps.
    # Several replicas per GPU fill those gaps. Leave num_workers unset to use every GPU.
    embedder = embedder.with_(
        resources=Resources(cpus=1, gpus=1 / args.replicas_per_gpu),
        **({"num_workers": args.num_workers} if args.num_workers else {}),
    )

    pipeline = Pipeline(
        name="vllm_embedding",
        stages=[
            # One file per task, so each GPU replica streams whole files.
            ParquetReader(
                file_paths=args.input_path, files_per_partition=1, fields=[*args.keep_fields, args.text_field]
            ),
            embedder,
            ParquetWriter(path=args.output_path),
        ],
    )

    start = time.perf_counter()
    try:
        pipeline.run()
    finally:
        ray_client.stop()
    print(f"Embeddings written to {args.output_path} in {time.perf_counter() - start:.0f}s")


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="Directory of Parquet files")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model id of a vLLM-supported embedding model")
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--keep-fields", nargs="*", default=[], help="Columns to keep next to the embeddings, e.g. an id"
    )
    parser.add_argument("--text-prefix", default=None, help='Prefix the model was trained with, e.g. "passage: "')
    parser.add_argument("--replicas-per-gpu", type=int, default=1, help="Model replicas sharing each GPU")
    parser.add_argument("--num-workers", type=int, default=None, help="Total replicas (default: fill every GPU)")
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=262144)
    parser.add_argument("--max-model-len", type=int, default=None, help="Truncate longer texts to this many tokens")
    parser.add_argument("--embedding-dtype", default="float32", choices=["float16", "float32", "float64"])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--vllm-init-kwargs", type=json.loads, default={}, help="Extra vllm.LLM kwargs as JSON")
    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
