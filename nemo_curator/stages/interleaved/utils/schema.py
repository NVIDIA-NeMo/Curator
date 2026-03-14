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

"""Centralized schema utilities for interleaved IO readers and writers.

All arrow-based readers/writers share these functions for type reconciliation,
schema alignment (null-fill + reorder), and schema serialization.
"""

from __future__ import annotations

import pyarrow as pa
from pyarrow import ipc

from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

_LARGE_COMPAT: dict[tuple[pa.DataType, pa.DataType], pa.DataType] = {
    (pa.large_string(), pa.string()): pa.large_string(),
    (pa.large_binary(), pa.binary()): pa.large_binary(),
}


def reconcile_schema(inferred: pa.Schema) -> pa.Schema:
    """Build a schema with canonical types for reserved columns and inferred types for passthrough.

    Avoids unsafe downcasts (e.g. large_string -> string) that cause offset
    overflow on large tables read via the pyarrow backend.  Field-level
    metadata from ``INTERLEAVED_SCHEMA`` (e.g. ``lance-encoding:blob`` on
    ``binary_content``) is propagated to the output schema.
    """
    canonical = {f.name: f for f in INTERLEAVED_SCHEMA}
    fields: list[pa.Field] = []
    for f in inferred:
        if f.name not in canonical:
            col_type = f.type.value_type if pa.types.is_dictionary(f.type) else f.type
            fields.append(pa.field(f.name, col_type, nullable=f.nullable))
            continue
        target = canonical[f.name]
        resolved_type = _LARGE_COMPAT.get((f.type, target.type), target.type)
        out_field = pa.field(f.name, resolved_type, nullable=target.nullable)
        if target.metadata:
            out_field = out_field.with_metadata(target.metadata)
        fields.append(out_field)
    return pa.schema(fields)


def align_table(table: pa.Table, target: pa.Schema) -> pa.Table:
    """Pad, reorder, and cast *table* to match *target* exactly.

    - Columns in *target* absent from *table* are added as null arrays.
    - Columns in *table* absent from *target* are dropped.
    - Column order matches *target*.
    - Core column types are reconciled via :func:`reconcile_schema` before casting.
    """
    reconciled_target = reconcile_schema(target)
    existing = set(table.schema.names)
    arrays: list[pa.Array] = []
    for field in reconciled_target:
        if field.name in existing:
            col = table.column(field.name)
            if col.type != field.type:
                safe = not (
                    (pa.types.is_large_string(col.type) and pa.types.is_string(field.type))
                    or (pa.types.is_large_binary(col.type) and pa.types.is_binary(field.type))
                )
                col = col.cast(field.type, safe=safe)
            arrays.append(col)
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.table(arrays, schema=reconciled_target)


def serialize_schema(schema: pa.Schema) -> str:
    """Serialize a pyarrow schema to a hex string via IPC."""
    sink = pa.BufferOutputStream()
    ipc.new_stream(sink, schema).close()
    return sink.getvalue().to_pybytes().hex()


def deserialize_schema(encoded: str) -> pa.Schema:
    """Recover a pyarrow schema serialized by :func:`serialize_schema`."""
    buf = pa.py_buffer(bytes.fromhex(encoded))
    with ipc.open_stream(buf) as reader:
        return reader.schema
