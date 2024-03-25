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

from functools import partial

import cudf

from nemo_curator.gpu_deduplication.jaccard_utils.io_utils import (
    get_text_ddf_from_json_path,
)
from nemo_curator.gpu_deduplication.utils import parse_nc_args


def merge_text_partition(df, connected_components_path):
    res = cudf.read_parquet(connected_components_path).drop(columns="dataset_id")
    res = res.drop_duplicates("group")
    res = res.drop(columns=["group"])
    df = res.merge(df, on="doc_id", how="left")
    df = df.rename(columns={"doc_id": "adlr_id"})
    return df.drop(columns="dataset_id")


def write_result_text_parquet(original_path, output_dir):
    ddf = get_text_ddf_from_json_path(
        original_path, num_files=-1, files_per_input_partition=10
    )

    connected_components_path = f"{output_dir}/connected_components.parquet"
    print(ddf.head())
    merge_func = partial(
        merge_text_partition, connected_components_path=connected_components_path
    )
    ddf = ddf.map_partitions(merge_func, meta={"adlr_id": "uint32", "text": "O"})

    mask = ddf.text.isnull()
    ddf = ddf[~mask]

    df = ddf.compute()
    df = df.reset_index(drop=True)
    df.to_parquet(f"{output_dir}/dedup_with_text.parquet")


def main(args):
    write_result_text_parquet(
        original_path=[args.original_path], output_dir=args.output_dir
    )


def attach_args(parser=None):
    description = """verify all pairs jaccard"""
    if not parser:
        parser = parse_nc_args(description=description)

    parser.add_argument(
        "--output-dir",
        type=str,
        help="The output directory to write results to",
    )
    parser.add_argument(
        "--original-path",
        type=str,
        help="The path of original jsonl files",
    )
    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    args = attach_args().parse_args()
