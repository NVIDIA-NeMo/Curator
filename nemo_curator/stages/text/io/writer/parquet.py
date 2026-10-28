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

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import split_protocol

from nemo_curator.tasks import DocumentBatch

from .base import BaseWriter

# TODO(NMCUR-432): Use the same deterministic partitioned-dataset layout for pandas and Arrow inputs.
_PANDAS_ONLY_WRITE_OPTIONS = {"partition_cols"}


@dataclass
class ParquetWriter(BaseWriter):
    """Writer that writes a DocumentBatch to a Parquet file."""

    # Additional kwargs for the selected Parquet writer
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    file_extension: str = "parquet"
    name: str = "parquet_writer"

    def write_data(self, task: DocumentBatch, file_path: str) -> None:
        """Write Arrow batches directly and retain pandas compatibility."""
        if isinstance(task.data, pa.Table) and self._write_arrow(task.data, file_path):
            return

        df = task.to_pandas()  # Convert to pandas DataFrame if needed
        if self.fields is not None:
            df = df[self.fields]
        # Build kwargs for to_parquet with explicit options
        write_kwargs = {
            "index": None,
        }

        # Add any additional kwargs, allowing them to override defaults
        write_kwargs.update(self.write_kwargs)
        df.to_parquet(file_path, **write_kwargs)

    def _write_arrow(self, table: pa.Table, file_path: str) -> bool:
        """Write directly unless the caller requested pandas-only behavior."""
        if self.fields is not None:
            table = table.select(self.fields)

        write_kwargs = self.write_kwargs.copy()
        engine = write_kwargs.pop("engine", None)
        index = write_kwargs.pop("index", None)
        write_kwargs.pop("storage_options", None)
        if engine not in {None, "pyarrow"} or index is True:
            return False

        if _PANDAS_ONLY_WRITE_OPTIONS.intersection(write_kwargs):
            return False

        _, fs_path = split_protocol(file_path)
        pq.write_table(table, fs_path, filesystem=self.fs, **write_kwargs)
        return True
