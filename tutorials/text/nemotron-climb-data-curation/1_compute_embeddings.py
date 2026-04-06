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

import argparse
import json

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.embedders.base import EmbeddingCreatorStage
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter
from nemo_curator.stages.text.modules.add_id import AddId

_EMBEDDING_MODEL = "NovaSearch/stella_en_400M_v5"
_EMBEDDING_MODEL_MAX_SEQ_LENGTH = 512


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    ray_client.start()

    if args.input_file_type == "jsonl":
        reader = JsonlReader
    elif args.input_file_type == "parquet":
        reader = ParquetReader
    else:
        msg = f"Invalid input file type: {args.input_file_type}"
        raise ValueError(msg)

    if args.output_file_type == "jsonl":
        writer = JsonlWriter
    elif args.output_file_type == "parquet":
        writer = ParquetWriter
    else:
        msg = f"Invalid output file type: {args.output_file_type}"
        raise ValueError(msg)

    pipeline = Pipeline(name="1_compute_embeddings")

    pipeline.add_stage(reader(file_paths=args.input_path, files_per_partition=1))

    if args.id_field is not None:
        pipeline.add_stage(AddId(id_field=args.id_field))

    if args.embedding_model == _EMBEDDING_MODEL and args.max_seq_length is None:
        args.max_seq_length = _EMBEDDING_MODEL_MAX_SEQ_LENGTH

    if args.embedding_model == _EMBEDDING_MODEL and args.transformers_init_kwargs == {}:
        args.transformers_init_kwargs = {"trust_remote_code": True}

    pipeline.add_stage(
        EmbeddingCreatorStage(
            model_identifier=args.embedding_model,
            use_sentence_transformer=True,
            text_field=args.text_field,
            embedding_field=args.embedding_field,
            cache_dir=args.cache_dir,
            max_chars=args.max_chars,
            max_seq_length=args.max_seq_length,
            padding_side=args.padding_side,
            embedding_pooling=args.embedding_pooling,
            model_inference_batch_size=args.model_inference_batch_size,
            autocast=not args.disable_autocast,
            sort_by_length=not args.disable_sort_by_length,
            hf_token=args.hf_token,
            transformers_init_kwargs=args.transformers_init_kwargs,
        )
    )

    pipeline.add_stage(writer(path=args.output_path))

    pipeline.run()

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Ray cluster args
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)

    # Reader args
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--input-file-type", type=str, required=True, choices=["jsonl", "parquet"])

    # ID args
    parser.add_argument("--id-field", type=str, default=None)

    # Embedding model args
    parser.add_argument("--embedding-model", type=str, default=_EMBEDDING_MODEL)
    parser.add_argument("--use-sentence-transformer", action="store_true")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--embedding-field", type=str, default="embeddings")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--max-seq-length", type=str, default=None)
    parser.add_argument("--padding-side", type=str, default="right")
    parser.add_argument("--embedding-pooling", type=str, default="mean_pooling", choices=["mean_pooling", "last_token"])
    parser.add_argument("--model-inference-batch-size", type=int, default=1024)
    parser.add_argument("--disable-autocast", action="store_true")
    parser.add_argument("--disable-sort-by-length", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--transformers-init-kwargs", type=json.loads, default={})

    # Writer args
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--output-file-type", type=str, required=True, choices=["jsonl", "parquet"])

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
