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

from typing import Optional

import dask.dataframe as dd
import numpy as np
from dask import delayed

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.module_utils import count_digits


class AddId:
    def __init__(
        self, id_field, id_prefix: str = "doc_id", start_index: Optional[int] = None
    ) -> None:
        self.id_field = id_field
        self.id_prefix = id_prefix
        self.start_index = start_index

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        if self.start_index is None:
            return self._add_id_fast(dataset)
        else:
            return self._add_id_ordered(dataset)

    def _add_id_fast(self, dataset: DocumentDataset) -> DocumentDataset:
        meta = dataset.df.dtypes.to_dict()
        meta[self.id_field] = "string"

        partition_zero_padding = count_digits(dataset.df.npartitions)
        id_df = dataset.df.map_partitions(
            self._add_id_fast_partition,
            partition_zero_padding,
            meta=meta,
            enforce_metadata=False,
        )

        return DocumentDataset(id_df)

    def _add_id_fast_partition(self, partition, global_padding, partition_info=None):
        local_padding = count_digits(len(partition))
        global_id = partition_info["number"]

        id_column = [
            f"{self.id_prefix}-{local_id:0{local_padding}d}{global_id:0{global_padding}d}"
            for local_id in range(len(partition))
        ]
        partition[self.id_field] = id_column

        return partition

    def _add_id_ordered(self, dataset: DocumentDataset) -> DocumentDataset:
        original_meta = dataset.df.dtypes.to_dict()
        original_meta[self.id_field] = "string"
        delayed_dataset = dataset.df.to_delayed()

        parition_lengths = [0]
        for partition in delayed_dataset[:-1]:
            parition_lengths.append(delayed(len)(partition))

        lower_id_bounds = delayed(np.cumsum)(parition_lengths)
        delayed_id_dataset = []
        for i, partition in enumerate(delayed_dataset):
            delayed_id_dataset.append(
                delayed(self._add_id_ordered_partition)(partition, lower_id_bounds[i])
            )

        id_dataset = DocumentDataset(
            dataset_df=dd.from_delayed(delayed_id_dataset, meta=original_meta)
        )

        return id_dataset

    def _add_id_ordered_partition(self, partition, partition_start_id):
        id_column = [
            f"{self.id_prefix}-{int(i + self.start_index):010d}"
            for i in range(partition_start_id, len(partition) + partition_start_id)
        ]
        partition[self.id_field] = id_column
        partition[self.id_field] = partition[self.id_field].astype("string")

        return partition
