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

import argparse
import json

import cudf
from dask import dataframe as dd
from dask.distributed import Client


def main(args):
    # Create the ID mapping
    df = cudf.DataFrame()
    df["base_id"] = [base_id for base_id in args.base_ids.split(",")]
    df["dataset_id"] = df["base_id"].hash_values()
    df_pd = df.to_pandas()

    output_dict = {
        hashed_id: base_id
        for base_id, hashed_id in zip(df_pd["base_id"], df_pd["dataset_id"])
    }

    # Write out the mapping to disk
    with open(args.output_id_mapping, "w") as output_file:
        json.dump(output_dict, output_file)

    # Index the parquet files by group
    client = Client()
    ddf = dd.read_parquet(args.path_to_connected_components)
    ddf = ddf.set_index("group")
    ddf.to_parquet(args.output_indexed_connected_components)


def attach_args(
    parser=argparse.ArgumentParser(
        """
Prepares the output connected components from dedup for
extraction to .txt and .jsonl files
  """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    parser.add_argument(
        "--base-ids",
        type=str,
        default="doc_id",
        help="A comma-delimited list of base-ids that were used for "
        "different datasets during dedup. For example, "
        "if you were deduplicating Wikipedia and Common Crawl, you might "
        "have adlr_ids such has wiki-000001 and cc-000001. "
        "The base-ids in this case would be 'wiki,cc'",
    )
    parser.add_argument(
        "--path-to-connected-components",
        type=str,
        default=None,
        help="Path to the connected components that is created "
        "at the last step of the fuzzy dedup.",
    )
    parser.add_argument(
        "--output-indexed-connected-components",
        type=str,
        default=None,
        help="Path to the output connected components "
        "that have been prepared for "
        "extraction to .txt and .jsonl files",
    )
    parser.add_argument(
        "--output-id-mapping",
        type=str,
        default="mapping.json",
        help="A mapping between each of the strings specified "
        "in '--base-ids' and their respective hashes",
    )
    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())


def console_script():
    main(attach_args().parse_args())
