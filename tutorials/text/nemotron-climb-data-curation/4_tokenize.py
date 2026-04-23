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
import os
import shutil

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer.megatron_tokenizer import MegatronTokenizerWriter
from nemo_curator.utils.merge_file_prefixes import merge_file_prefixes


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    ray_client.start()

    if args.input_filetype == "jsonl":
        reader = JsonlReader
    elif args.input_filetype == "parquet":
        reader = ParquetReader
    else:
        msg = f"Invalid input file type: {args.input_filetype}"
        raise ValueError(msg)

    subdirectories = [os.path.join(args.input_path, d) for d in os.listdir(args.input_path)]

    for centroid_path in subdirectories:
        pipeline = Pipeline(name="4_tokenize")

        pipeline.add_stage(reader(file_paths=centroid_path, files_per_partition=1, fields=[args.text_field]))

        centroid = centroid_path.split("/")[-1].split("=")[1]
        cache_path = os.path.join(args.output_path, "cache")
        os.makedirs(cache_path, exist_ok=True)
        writer_path = os.path.join(cache_path, f"domain_{centroid}")

        # Use Curator pipeline to tokenize the data
        pipeline.add_stage(
            MegatronTokenizerWriter(
                path=writer_path,
                model_identifier=args.tokenizer_model,
                cache_dir=args.cache_dir,
                hf_token=args.hf_token,
                text_field=args.text_field,
                tokenization_batch_size=args.tokenization_batch_size,
                append_eod=args.append_eod,
                transformers_init_kwargs=args.transformers_init_kwargs,
            )
        )

        pipeline.run()

        # Merge the tokenized files into a single bin/idx file
        result_path = os.path.join(args.output_path, f"domain_{centroid}")
        merge_file_prefixes(input_dir=writer_path, output_prefix=result_path)

        # Remove individual tokenized files in favor of the merged bin/idx file
        shutil.rmtree(cache_path)

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Ray cluster args
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=0)

    # Reader args
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--input-filetype", type=str, default="jsonl", choices=["parquet", "jsonl"])

    # Tokenizer args
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--tokenizer-model", type=str, default="meta-llama/Llama-2-7b")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--tokenization-batch-size", type=int, default=1000)
    parser.add_argument("--append-eod", action="store_true")
    parser.add_argument("--transformers-init-kwargs", type=json.loads, default={})

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
