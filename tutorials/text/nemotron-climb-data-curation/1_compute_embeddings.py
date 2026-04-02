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

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.embedders.base import EmbeddingCreatorStage
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter
from nemo_curator.stages.text.modules.add_id import AddId


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

    pipeline.add_stage(
        EmbeddingCreatorStage(
            model_identifier=args.embedding_model,
            use_sentence_transformer=False,
            text_field=args.text_field,
            embedding_field=args.embedding_field,
            cache_dir=args.cache_dir,
            max_chars=args.max_chars,
            max_seq_length=None,
            padding_side="right",
            embedding_pooling="mean_pooling",  # or "last_token"
            model_inference_batch_size=1024,
            autocast=True,
            sort_by_length=True,
            hf_token=args.hf_token,
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

    # Embedder args
    # TODO: Add all arguments
    parser.add_argument("--embedding-model", type=str, default="NovaSearch/stella_en_400M_v5")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--embedding-field", type=str, default="embeddings")
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)

    # Writer args
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--output-file-type", type=str, required=True, choices=["jsonl", "parquet"])

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
