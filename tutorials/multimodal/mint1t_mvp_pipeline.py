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
from nemo_curator.stages.multimodal.io import MultimodalParquetWriter, WebdatasetReader
from nemo_curator.stages.multimodal.stages import MultimodalJpegAspectRatioFilterStage


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    read_kwargs = {}
    write_kwargs = {}
    if args.storage_options_json:
        storage_options = json.loads(args.storage_options_json)
        read_kwargs["storage_options"] = storage_options
        write_kwargs["storage_options"] = storage_options

    pipe = Pipeline(name="mint1t_mvp_multimodal", description="WebDataset MINT1T -> multimodal rows -> parquet")
    pipe.add_stage(
        WebdatasetReader(
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            blocksize=args.input_blocksize,
            max_batch_bytes=args.output_max_batch_bytes,
            read_kwargs=read_kwargs,
            load_binary=False,
        )
    )
    pipe.add_stage(MultimodalJpegAspectRatioFilterStage(drop_invalid_rows=True))
    pipe.add_stage(
        MultimodalParquetWriter(
            path=args.output_path,
            materialize_on_write=args.materialize_on_write,
            write_kwargs=write_kwargs,
            mode=args.mode,
        )
    )
    return pipe


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient()
    ray_client.start()
    pipeline = build_pipeline(args)
    print(pipeline.describe())
    pipeline.run()
    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MINT1T multimodal MVP pipeline")
    parser.add_argument("--input-path", type=str, required=True, help="Input tar shard path or directory")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory for parquet")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--input-blocksize", type=str, default=None)
    parser.add_argument("--output-max-batch-bytes", type=int, default=None)
    parser.add_argument("--materialize-on-write", action="store_true", dest="materialize_on_write")
    parser.add_argument("--no-materialize-on-write", action="store_false", dest="materialize_on_write")
    parser.set_defaults(materialize_on_write=True)
    parser.add_argument("--mode", type=str, default="ignore", choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument(
        "--storage-options-json",
        type=str,
        default=None,
        help="JSON-encoded fsspec storage options for cloud paths",
    )
    main(parser.parse_args())
