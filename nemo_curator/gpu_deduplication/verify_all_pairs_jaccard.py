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
from time import time

import cudf
import dask_cudf

from nemo_curator.gpu_deduplication.jaccard_utils.jaccard_similarity_utils import (
    compute_jaccard_partition,
    create_empty_jaccard_result,
)
from nemo_curator.gpu_deduplication.utils import get_client, parse_nc_args


def num_ngram(ds):
    return ds.str.character_ngrams(5, True).list.unique().list.len()


def write_eligible_pairs(dedup_with_text_path, cache_dir):
    df = cudf.read_parquet(dedup_with_text_path)
    df["num_ngram"] = num_ngram(df["text"])
    df.drop(columns="text", inplace=True)
    df["group"] = 0
    B = 8_000
    rm = 0
    for s in range(0, df.shape[0], B):
        e = min(s + B, df.shape[0])
        da = df.iloc[s:e]
        db = da.merge(df, on="group")
        mask = db["adlr_id_x"] < db["adlr_id_y"]
        db = db[mask]
        mask = (db["num_ngram_x"] < db["num_ngram_y"] * 0.8) | (
            db["num_ngram_y"] < db["num_ngram_x"] * 0.8
        )
        print(db.shape, mask.sum())
        rm += mask.sum()
        db = db[~mask]
        db.drop(columns=["group", "num_ngram_x", "num_ngram_y"], inplace=True)
        db.to_parquet(f"{cache_dir}/pair_{s}.parquet")
        del da, db
    print("total pairs removed", rm)


def merge_text(df, dedup_with_text_path):
    dg = cudf.read_parquet(dedup_with_text_path)
    for i in "xy":
        df = df.merge(dg, left_on=f"adlr_id_{i}", right_on="adlr_id")
        df.drop(columns="adlr_id", inplace=True)
    return df


def get_max_num_rows_to_process_once(df):
    nbytes = max(
        df["text_x"].str.byte_count().sum(), df["text_y"].str.byte_count().sum()
    )

    # TODO: fix below
    # to 4x
    exploded_bytes = nbytes * 5 * 4
    max_chars_allowed = 2_147_483_647
    byte_ratio = int(exploded_bytes) // max_chars_allowed
    if byte_ratio > 1:
        nrows_at_once = len(df) // byte_ratio
    else:
        nrows_at_once = len(df)

    nrows_at_once = max(1, nrows_at_once)
    return nrows_at_once


def compute_jaccard_pair(docs_df):
    nrows_at_once = get_max_num_rows_to_process_once(docs_df)
    result_ls = []
    for i in range(0, docs_df.shape[0], nrows_at_once):
        pair_df = docs_df[i : i + nrows_at_once]
        if len(pair_df) == 0:
            result_df = create_empty_jaccard_result()
        else:
            result_df = compute_jaccard_partition(pair_df)
        result_ls.append(result_df)
    if len(result_ls) == 0:
        return create_empty_jaccard_result()
    df_pair = cudf.concat(result_ls)
    return df_pair


def run_verify_all_pairs_jaccard(dedup_with_text_path, cache_dir, output_dir):
    ddf = dask_cudf.read_parquet(f"{cache_dir}/pair_*.parquet")
    ddf = ddf.repartition(npartitions=2048)

    meta_df = cudf.DataFrame(
        {
            "adlr_id_x": [0],
            "adlr_id_y": [0],
            "text_x": ["x"],
            "text_y": ["x"],
        }
    )

    ddf = ddf.map_partitions(
        partial(merge_text, dedup_with_text_path=dedup_with_text_path), meta=meta_df
    )

    meta_df = cudf.DataFrame(
        {
            "adlr_id_x": [0],
            "adlr_id_y": [0],
            "jaccard": [1.0],
        }
    )

    ddf = ddf.map_partitions(compute_jaccard_pair, meta=meta_df)
    mask = ddf["jaccard"] > 0.8
    dup_pairs = ddf[mask].compute()
    print("# of duplicated pairs with jaccard>0.8", dup_pairs.shape[0])
    dup_pairs.to_parquet(f"{output_dir}/duplicated_pairs.parquet")


def main(args):
    start = time()
    description = """Verify correctness of deduped results by calculating all pairs"""
    dedup_with_text_path = f"{args.output_dir}/dedup_with_text.parquet"

    write_eligible_pairs(dedup_with_text_path, args.cache_dir)
    client = get_client(args)

    # Run actual computation
    run_verify_all_pairs_jaccard(
        dedup_with_text_path,
        args.cache_dir,
        args.output_dir,
    )
    print(f"All done in {time()-start:.1f} seconds")


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
        "--cache-dir",
        type=str,
        help="The cache directory to write intermediate results to",
    )
    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
