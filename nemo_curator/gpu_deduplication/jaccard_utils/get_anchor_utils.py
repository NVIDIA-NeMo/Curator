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


def random_select_anchor(df_bk, n=2):
    """
    Randomly select `n` anchors from each bucket.
    """
    df_bk = df_bk.copy()
    df_bk["hash"] = df_bk[["doc_id", "dataset_id"]].hash_values()
    df_bk = df_bk.sort_values(["bucket", "hash"])
    df_bk["order_in_bucket"] = df_bk.groupby("bucket").cumcount()
    df_bk["is_anchor"] = df_bk["order_in_bucket"] < n
    for i in range(0, n):
        df_bk[f"is_anchor_id_{i}"] = df_bk["order_in_bucket"] == i
    df_bk = df_bk.drop(columns=["hash", "order_in_bucket"], axis=1)
    df_bk = df_bk.reset_index(drop=True)
    df_bk = df_bk[df_bk.is_anchor]
    return df_bk


def add_anchor_docs(df_bk):
    """
    Get anchor documents for each bucket.
    """
    num_anchors = 2
    df_anchor_bk = random_select_anchor(df_bk=df_bk, n=num_anchors)
    df_anchor_bk_0 = df_anchor_bk[df_anchor_bk["is_anchor_id_0"]][
        ["bucket", "dataset_id", "doc_id"]
    ].reset_index(drop=True)
    df_anchor_bk_0 = df_anchor_bk_0.rename(
        columns={"doc_id": "anchor_0_doc_id", "dataset_id": "anchor_0_dataset_id"}
    )

    df_anchor_bk_1 = df_anchor_bk[df_anchor_bk["is_anchor_id_1"]][
        ["bucket", "dataset_id", "doc_id"]
    ].reset_index(drop=True)
    df_anchor_bk_1 = df_anchor_bk_1.rename(
        columns={"doc_id": "anchor_1_doc_id", "dataset_id": "anchor_1_dataset_id"}
    )

    df_anchor_docs = df_anchor_bk_1.merge(df_anchor_bk_0, on=["bucket"], how="inner")
    df_anchor_docs_with_bk = df_bk.merge(df_anchor_docs, on=["bucket"], how="inner")
    return df_anchor_docs_with_bk
