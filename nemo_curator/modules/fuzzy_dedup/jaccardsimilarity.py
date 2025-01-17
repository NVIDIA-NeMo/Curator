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

from __future__ import annotations

import os

import cudf
import numpy as np
from dask import dataframe as dd


class JaccardSimilarity:
    def __init__(
        self,
        id_field="id",
        anchor_id_fields=["anchor_0_id", "anchor_1_id"],
        text_field="text",
        ngram_width=5,
    ):
        self.id_field = id_field
        self.anchor_id_fields = anchor_id_fields
        self.text_field = text_field
        self.anchor_id = f"anchor_{id_field}"
        self.left_id = f"{self.id_field}_x"
        self.right_id = f"{self.id_field}_y"
        self.ngram_width = ngram_width

    def __call__(DocumentDataset):
        raise NotImplementedError

    def jaccard_compute(self, shuffled_docs_path):
        paths = [
            entry.path
            for entry in os.scandir(shuffled_docs_path)
            if not entry.path.endswith(".txt")
        ]
        meta_df = cudf.DataFrame(
            {
                self.left_id: ["x"],
                self.right_id: ["y"],
                "jaccard": np.float32([0.0]),
            }
        )
        result_df = dd.from_map(
            self._compute_jaccard_on_1_partition, paths, meta=meta_df
        ).reset_index(drop=True)
        return result_df

    def _compute_jaccard_on_1_partition(self, path):
        try:
            df = cudf.read_parquet(path)
            pair_df = self._compute_jaccard_and_create_pair_df(df)
        except OverflowError:
            paths = [entry.path for entry in os.scandir(os.path.join(path))]
            anchor_df_str_size_ls = [
                self._get_anchor_docs_and_string_size(path) for path in paths
            ]
            anchor_df = cudf.concat(
                [anchor_doc for anchor_doc, _ in anchor_df_str_size_ls],
                ignore_index=True,
            ).drop_duplicates()
            df_str_size = [str_size for _, str_size in anchor_df_str_size_ls]
            paths = JaccardSimilarity._create_bins(
                df_str_size, np.iinfo(np.int32).max // 10
            )
            pair_dfs = []
            for path in paths:
                print(path)
                df = cudf.read_parquet(path).reset_index(drop=True)
                df = cudf.concat([df, anchor_df], ignore_index=True)
                pair_df = self._compute_jaccard_and_create_pair_df(df)
                pair_dfs.append(pair_df)
            pair_df = cudf.concat(pair_dfs, ignore_index=True)
        return pair_df

    def _get_anchor_docs_and_string_size(self, path):
        df = cudf.read_parquet(path)
        str_bytes = df[self.text_field].str.byte_count().sum()
        is_anchor_flag = df[self.id_field] == df[self.anchor_id_fields[0]]
        for anchor_id in self.anchor_id_fields[1:]:
            is_anchor_flag = is_anchor_flag | (df[self.id_field] == df[anchor_id])
        anchor_df = df[is_anchor_flag].reset_index(drop=True)
        return anchor_df, {"path": path, "str_bytes": str_bytes}

    @staticmethod
    def _create_bins(path_dicts, max_size):
        path_dicts.sort(key=lambda x: x["str_bytes"], reverse=True)
        bins, bin_sizes = [], []
        for path_d in path_dicts:
            new_path, new_size = path_d["path"], path_d["str_bytes"]
            for i, bin_size in enumerate(bin_sizes):
                if bin_size + new_size <= max_size:
                    bins[i].append(new_path)
                    bin_sizes[i] += new_size
                    new_size = 0
                    break
            if new_size:
                bins.append([new_path])
                bin_sizes.append(new_size)
        return bins

    def _compute_jaccard_and_create_pair_df(self, df):
        df = df.drop_duplicates(
            subset=[self.id_field] + self.anchor_id_fields, ignore_index=True
        )
        anchor_columns = self.anchor_id_fields
        id_field = self.id_field
        result_ls = []
        try:
            for anchor_col in anchor_columns:
                doc_df = df[[id_field, self.text_field, anchor_col]]
                doc_df = doc_df.rename(columns={anchor_col: self.anchor_id})
                doc_df = doc_df[doc_df[id_field] != doc_df[self.anchor_id]]
                anchor_df = self._get_anchor_df(df, anchor_col)
                result_df = self._compute_jaccard_pair(doc_df, anchor_df)
                result_ls.append(result_df)

            return cudf.concat(result_ls)
        except OverflowError as e:
            print(
                "Failed with  OverflowError in compute_jaccard_and_create_pair_df",
                flush=True,
            )
            print(df, flush=True)
            print("--" * 30)
            print("Error")
            print("---" * 30)
            raise e

    def _get_anchor_df(self, df, anchor_col):
        anchor_df = df[df[self.id_field] == df[anchor_col]]
        anchor_df = anchor_df.reset_index(drop=True)
        anchor_df = anchor_df[[anchor_col, self.text_field]]
        anchor_df = anchor_df.rename(columns={anchor_col: self.anchor_id})
        return anchor_df

    def _compute_jaccard_pair(self, docs_df, anchor_df):
        nrows_at_once = JaccardSimilarity._get_max_num_rows_to_process_once(
            df=docs_df, text_field=self.text_field
        )
        result_ls = []
        for i in range(0, docs_df.shape[0], nrows_at_once):
            pair_df = docs_df[i : i + nrows_at_once]
            pair_df = pair_df.merge(anchor_df, on=self.anchor_id)
            pair_df = pair_df.rename(
                columns={self.id_field: self.left_id, self.anchor_id: self.right_id}
            )
            mask = pair_df[self.left_id] != pair_df[self.right_id]
            pair_df = pair_df[mask].reset_index(drop=True)
            if len(pair_df) == 0:
                result_df = self._create_empty_jaccard_result()
            else:
                result_df = self._compute_jaccard_partition(pair_df)
            result_ls.append(result_df)
        if len(result_ls) == 0:
            return self._create_empty_jaccard_result()
        df_pair = cudf.concat(result_ls)
        return df_pair

    def _create_empty_jaccard_result(self):
        df = cudf.DataFrame()
        df[self.left_id] = "x"
        df[self.right_id] = "y"
        df["jaccard"] = np.empty(shape=0, dtype=np.float32)
        return df

    def _compute_jaccard_partition(self, df):
        text_x = f"{self.text_field}_x"
        text_y = f"{self.text_field}_y"
        df["jaccard"] = df[text_x].str.jaccard_index(df[text_y], width=self.ngram_width)
        df.drop(columns=[text_x, text_y], inplace=True)
        return df

    @staticmethod
    def _get_max_num_rows_to_process_once(df, text_field):
        nbytes = df[text_field].str.byte_count().sum()
        # Number of exmploded bytes
        exploded_bytes = nbytes * 5 * 2
        max_chars_allowed = 2_147_483_647
        byte_ratio = int(exploded_bytes) // max_chars_allowed
        if byte_ratio > 1:
            nrows_at_once = len(df) // byte_ratio
        else:
            nrows_at_once = len(df)

        nrows_at_once = max(1, nrows_at_once)
        return nrows_at_once
