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


def int_ids_to_str(df, id_column="id"):
    """
    Converts int id's generated via `convert_str_id_to_int`
    back to a string ID
    """
    df[id_column] = df["dataset_id"].astype(str) + "-" + df["doc_id"].astype(str)
    df.drop(columns=["dataset_id", "doc_id"], inplace=True)

    if "anchor_0_dataset_id" in df.columns:
        df[f"anchor_0_{id_column}"] = (
            df["anchor_0_dataset_id"].astype(str)
            + "-"
            + df["anchor_0_doc_id"].astype(str)
        )
        df.drop(columns=["anchor_0_dataset_id", "anchor_0_doc_id"], inplace=True)

    if "anchor_1_dataset_id" in df.columns:
        df[f"anchor_1_{id_column}"] = (
            df["anchor_1_dataset_id"].astype(str)
            + "-"
            + df["anchor_1_doc_id"].astype(str)
        )
        df.drop(columns=["anchor_1_dataset_id", "anchor_1_doc_id"], inplace=True)
    return df
