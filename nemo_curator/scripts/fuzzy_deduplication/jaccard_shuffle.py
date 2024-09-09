# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import time

from nemo_curator.modules.fuzzy_dedup import _Shuffle
from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.fuzzy_dedup_utils.io_utils import (
    get_text_ddf_from_json_path_with_blocksize,
)
from nemo_curator.utils.script_utils import ArgumentHelper


def func():
    import cudf

    from nemo_curator.modules.fuzzy_dedup import _Shuffle


def main(args):
    input_data_paths = args.input_data_dirs
    input_anchor_docs_with_bk_dir = args.input_bucket_mapping_dir
    OUTPUT_PATH = args.output_dir
    output_shuffled_docs_path = os.path.join(OUTPUT_PATH, "shuffled_docs.parquet")

    client = get_client(**ArgumentHelper.parse_client_args(args))
    client.run(func)
    print(f"Num Workers = {get_num_workers(client)}", flush=True)
    print("Connected to dask cluster", flush=True)
    print("Running jaccard shuffle script", flush=True)
    print(f"Args = {args}")
    st = time.time()
    text_ddf = get_text_ddf_from_json_path_with_blocksize(
        input_data_paths=input_data_paths,
        num_files=args.num_files,
        blocksize=args.text_ddf_blocksize,
        id_column=args.input_json_id_field,
        text_column=args.input_json_text_field,
        input_meta=args.input_meta,
    )
    print(
        "Graph creation for get_text_ddf_from_json_path_with_blocksize" " complete.",
        flush=True,
    )
    print(f"text_ddf.npartitions  = {text_ddf.npartitions}", flush=True)
    shuffle = _Shuffle(
        id_fields=["dataset_id", "doc_id"],
        text_field=args.input_json_text_field,
        profile_dir=args.profile_path,
        int_to_str_id=args.input_json_id_field,
    )
    shuffle.shuffle_docs_on_buckets(
        documents_df=text_ddf,
        bucket_w_anchors_path=input_anchor_docs_with_bk_dir,
        output_shuffled_docs_path=output_shuffled_docs_path,
        bucket_mapping_df_blocksize=args.bucket_mapping_ddf_blocksize,
        parts_per_worker=args.parts_per_worker,
        bucket_parts_per_worker=args.bucket_parts_per_worker,
        partition_on="_output_partition_id",
    )
    et = time.time()
    print(f"Jaccard Shuffle E2E time taken = {et-st} s")


def attach_args(parser=None):
    if not parser:
        description = """Shuffles input text documents based on the given bucket
        map. The output is a partitioned parquet dataset with the documents
        shuffled by buckets
        """
        parser = ArgumentHelper.parse_gpu_dedup_args(description=description)

    argumentHelper = ArgumentHelper(parser)

    argumentHelper.add_arg_input_meta()
    argumentHelper.add_arg_output_dir()
    argumentHelper.add_arg_text_ddf_blocksize()
    parser.add_argument(
        "--bucket-mapping-ddf-blocksize",
        type=int,
        default=256,
        help="The block size for for anchor_docs_with_bk ddf in mb",
    )
    parser.add_argument(
        "--bucket-parts-per-worker",
        default=8,
        type=int,
        help="The number of bucket parts to process per worker per batch",
    )
    parser.add_argument(
        "--input-bucket-mapping-dir",
        type=str,
        help="The directory containing anchor docs with bk files",
    )
    parser.add_argument(
        "--parts-per-worker",
        default=1,
        type=int,
        help="The number of parts to process per worker per batch",
    )

    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
