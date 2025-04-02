# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import warnings
from typing import List, Optional, Union

import dask.dataframe as dd


def deduplicate_groups(
    duplicates: dd.DataFrame, group_field: Optional[str], perform_shuffle: bool
) -> dd.DataFrame:
    if group_field is None:
        return duplicates

    if perform_shuffle:
        # Redistribute data across partitions so that all duplicates are in same partition
        duplicates_shuffled = duplicates.shuffle(on=[group_field], ignore_index=True)
    else:
        duplicates_shuffled = duplicates

    duplicates_to_remove = (
        duplicates_shuffled
        # For each partition, keep only the duplicated rows (excluding first occurrence)
        .map_partitions(lambda x: x[x[group_field].duplicated(keep="first")]).drop(
            columns=group_field
        )
    )
    return duplicates_to_remove


def left_anti_join(
    left: dd.DataFrame,
    right: dd.DataFrame,
    left_on: Union[str, List[str]],
    right_on: Union[str, List[str]],
) -> dd.DataFrame:
    assert left_on != right_on, "left_on and right_on cannot be the same"

    merge = left.merge(
        right=right,
        how="left",
        broadcast=True,  # Broadcast smaller DataFrame to all partitions
        left_on=left_on,
        right_on=right_on,
    )

    # This effectively removes all rows that were not in duplicates_to_remove
    removed_result = merge[merge[right_on].isna()].drop(columns=[right_on])
    return removed_result


def remove_duplicates(
    left: dd.DataFrame,
    duplicates: dd.DataFrame,
    id_field: str,
    group_field: Optional[str] = None,
    perform_shuffle: bool = False,
) -> dd.DataFrame:
    left_npartitions = left.optimize().npartitions
    right_npartitions = duplicates.optimize().npartitions
    if left_npartitions < right_npartitions:
        msg = (
            f"The number of partitions in `dataset` ({left_npartitions}) is less than "
            f"the number of partitions in the duplicates ({right_npartitions}). "
            "This may lead to a shuffle join. Repartitioning right dataset to match left partitions."
            "To control this behavior, call identify_duplicates and removal as two separate steps"
        )
        warnings.warn(msg)
        duplicates = duplicates.repartition(npartitions=left_npartitions)

    # Create a new column name for temporary ID storage during merge
    new_id_field = f"{id_field}_new"

    duplicates_to_remove = (
        deduplicate_groups(duplicates, group_field, perform_shuffle)
        # Rename the ID field to avoid conflicts in the upcoming merge
        .rename(columns={id_field: new_id_field})[[new_id_field]]
    )

    return left_anti_join(
        left=left,
        right=duplicates_to_remove,
        left_on=id_field,
        right_on=new_id_field,
    )
