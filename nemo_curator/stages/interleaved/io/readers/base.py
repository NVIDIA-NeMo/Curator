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

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.utils.schema import align_table, reconcile_schema
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA


@dataclass
class BaseInterleavedReader(ProcessingStage[FileGroupTask, InterleavedBatch]):
    """Base contract for interleaved readers.

    By default (``schema=None``) user-added passthrough columns are preserved
    and only reserved-column types are reconciled via ``reconcile_schema``.

    If *schema* is set explicitly, every output table is strictly aligned to it
    (missing columns become typed nulls, extra columns are dropped).

    Use *schema_overrides* to add or override individual field types relative to
    ``INTERLEAVED_SCHEMA`` while keeping strict alignment:

    .. code-block:: python

        reader = InterleavedParquetReader(
            "data.parquet",
            schema_overrides={"url": pa.string(), "timestamp": pa.int64()},
        )
    """

    read_kwargs: dict[str, Any] = field(default_factory=dict)
    schema: pa.Schema | None = None
    schema_overrides: dict[str, pa.DataType] | None = None
    name: str = "base_interleaved_reader"

    def __post_init__(self) -> None:
        if self.schema_overrides is not None:
            self.schema = _resolve_schema(self.schema, self.schema_overrides)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sample_id", "position", "modality"]

    def _align_output(self, table: pa.Table) -> pa.Table:
        """Reconcile or align *table* to the declared schema."""
        if self.schema is not None:
            return align_table(table, self.schema)
        return table.cast(reconcile_schema(table.schema))


def _resolve_schema(
    schema: pa.Schema | None,
    overrides: dict[str, pa.DataType] | None,
) -> pa.Schema:
    """Return the effective schema from user-supplied *schema* or *overrides*.

    Priority: *schema* > *schema_overrides* merged on top of ``INTERLEAVED_SCHEMA``.
    Returns ``None`` only when both inputs are ``None``.
    """
    if schema is not None:
        return schema
    if overrides:
        fields = {f.name: f for f in INTERLEAVED_SCHEMA}
        for name, dtype in overrides.items():
            metadata = fields[name].metadata if name in fields else None
            fields[name] = pa.field(name, dtype, nullable=True, metadata=metadata)
        return pa.schema(list(fields.values()))
    return None
