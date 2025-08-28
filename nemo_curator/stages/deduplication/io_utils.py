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

from collections.abc import Callable

# TODO: Should this be a safe import?
import cudf
import numpy as np
import ray

from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR, IdGenerator
from nemo_curator.utils.file_utils import get_fs


class DeduplicationIO:
    def __init__(
        self,
        id_generator: "IdGenerator | None",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id_generator = id_generator

    def read_jsonl(
        self, filepath: str | list[str], columns: list[str] | None = None, assign_id: bool = False, **kwargs
    ) -> "cudf.DataFrame":
        df = cudf.read_json(filepath, lines=True, **kwargs)
        if columns is not None:
            df = df[columns]
        return self.assign_id(filepath, df) if assign_id and self.id_generator else df

    def read_parquet(self, filepath: str | list[str], assign_id: bool = False, **kwargs) -> "cudf.DataFrame":
        read_kwargs = kwargs.copy()
        read_kwargs["allow_mismatched_pq_schemas"] = True
        df = cudf.read_parquet(filepath, **read_kwargs)
        return self.assign_id(filepath, df) if assign_id and self.id_generator else df

    def write_parquet(self, df: "cudf.DataFrame", filepath: str, **kwargs) -> None:
        fs = get_fs(filepath, storage_options=kwargs.get("storage_options", {}))
        # TODO: Add overwrite behavior here
        fs.makedirs(fs._parent(filepath), exist_ok=True)
        df.to_parquet(filepath, **kwargs)

    def custom_read(
        self, filepath: str | list[str], read_func: Callable, assign_id: bool = False, **kwargs
    ) -> "cudf.DataFrame":
        df = read_func(filepath, **kwargs)
        return self.assign_id(filepath, df) if assign_id and self.id_generator else df

    def assign_id(self, filepath: str | list[str], df: "cudf.DataFrame") -> "cudf.DataFrame":
        if CURATOR_DEDUP_ID_STR not in df.columns:
            # Only need the ID generator if _curator_id is missing
            if self.id_generator is None:
                msg = "ID generator is required when _curator_id column is not present in the data"
                raise ValueError(msg)

            num_rows = len(df)
            min_id = ray.get(self.id_generator.register_batch.remote(filepath, num_rows))
            df[CURATOR_DEDUP_ID_STR] = np.arange(min_id, min_id + num_rows)
        return df
