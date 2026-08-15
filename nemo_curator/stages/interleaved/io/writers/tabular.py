# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .base import BaseInterleavedWriter

_DICTIONARY_COLUMNS = [
    "position",
    "modality",
    "content_type",
    "source_ref.uri",
    "source_ref.content_type",
    "source_frame_index",
]


@dataclass
class InterleavedParquetWriterStage(BaseInterleavedWriter):
    """Write interleaved rows to Parquet with optional binary materialization."""

    file_extension: str = "parquet"
    name: str = "interleaved_parquet_writer"

    def _write_table(self, table: pa.Table, file_path: str, write_kwargs: dict[str, Any]) -> None:
        write_kwargs.setdefault("compression", "snappy")
        write_kwargs.setdefault("row_group_size", 128_000)
        write_kwargs.setdefault("use_dictionary", _DICTIONARY_COLUMNS)
        write_kwargs.pop("index", None)
        with self.fs.open(file_path, "wb") as fobj, self._time_metric("parquet_write_s"):
            pq.write_table(table.replace_schema_metadata(), fobj, **write_kwargs)
