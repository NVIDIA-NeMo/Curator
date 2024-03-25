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


def convert_str_id_to_int(df, id_column="id"):
    """
    Converts the legacy id format "dataset_name-0000034"
    type of ID into 2 int based ID's
    """
    dx = df[id_column].str.rsplit("-", n=1, expand=True)
    df["doc_id"] = dx[1].astype("int64").values
    df["dataset_id"] = dx[0].hash_values()
    df.drop(columns=[id_column], inplace=True)
    return df


def convert_str_pair_adlr_ids_to_int(df):
    assert "adlr_id_x" in df.columns
    assert "adlr_id_y" in df.columns

    for tag in ["x", "y"]:
        dx = df[f"adlr_id_{tag}"].str.rsplit("-", n=1, expand=True)
        df[f"dataset_id_{tag}"] = dx[0].astype("uint32").values
        df[f"doc_id_{tag}"] = dx[1].astype("int64").values
        # See the above convert_adlr_id_to_int function
        df = df.drop(columns=[f"adlr_id_{tag}"])
    return df


def combine_back_adlr_ids(df):
    df["adlr_id"] = df["dataset_id"].astype(str) + "-" + df["doc_id"].astype(str)
    df.drop(columns=["dataset_id", "doc_id"], inplace=True)

    if "anchor_0_dataset_id" in df.columns:
        df["anchor_0_adlr_id"] = (
            df["anchor_0_dataset_id"].astype(str)
            + "-"
            + df["anchor_0_doc_id"].astype(str)
        )
        df.drop(columns=["anchor_0_dataset_id", "anchor_0_doc_id"], inplace=True)

    if "anchor_1_dataset_id" in df.columns:
        df["anchor_1_adlr_id"] = (
            df["anchor_1_dataset_id"].astype(str)
            + "-"
            + df["anchor_1_doc_id"].astype(str)
        )
        df.drop(columns=["anchor_1_dataset_id", "anchor_1_doc_id"], inplace=True)
    return df
