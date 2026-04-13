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

All arrow-based readers/writers share these functions for type reconciliation
and schema alignment (null-fill + reorder).
"""

from __future__ import annotations

import pyarrow as pa
from loguru import logger

from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA, RESERVED_COLUMNS

_LARGE_COMPAT: dict[tuple[pa.DataType, pa.DataType], pa.DataType] = {
    (pa.large_string(), pa.string()): pa.large_string(),
    (pa.large_binary(), pa.binary()): pa.large_binary(),
}


def reconcile_schema(inferred: pa.Schema) -> pa.Schema:
    """Build a schema with canonical types for reserved columns and inferred types for passthrough.

    Avoids unsafe downcasts (e.g. large_string -> string) that cause offset
    overflow on large tables read via the pyarrow backend.
    """
    canonical = {f.name: f for f in INTERLEAVED_SCHEMA}
    fields: list[pa.Field] = []
    for f in inferred:
        if f.name not in canonical:
            # Unwrap dictionary encoding — it's a Parquet storage detail, not part of the logical type.
            col_type = f.type.value_type if pa.types.is_dictionary(f.type) else f.type
            fields.append(pa.field(f.name, col_type, nullable=f.nullable))
            continue
        target = canonical[f.name]
        resolved_type = _LARGE_COMPAT.get((f.type, target.type), target.type)
        fields.append(pa.field(f.name, resolved_type, nullable=target.nullable))
    return pa.schema(fields)


def resolve_schema(
    schema: pa.Schema | None,
    overrides: dict[str, pa.DataType] | None,
) -> pa.Schema | None:
    """Return the effective schema from user-supplied *schema* or *overrides*.

    Priority: *schema* > *overrides* merged on top of ``INTERLEAVED_SCHEMA`` > ``None``.

    If *schema* is provided and *overrides* is also provided, *overrides* are
    ignored and a warning is emitted.  Returns ``None`` if both are ``None``.
    """
    if schema is not None:
        if overrides:
            logger.warning("schema_overrides ignored because schema= is already set; use one or the other, not both")
        return schema
    if overrides:
        fields = {f.name: f for f in INTERLEAVED_SCHEMA}
        for name, dtype in overrides.items():
            orig = fields.get(name)
            nullable = orig.nullable if orig is not None else True
            metadata = orig.metadata if orig is not None else None
            fields[name] = pa.field(name, dtype, nullable=nullable, metadata=metadata)
        return pa.schema(list(fields.values()))
    return None


def align_table(table: pa.Table, target: pa.Schema) -> pa.Table:
    """Pad, reorder, and cast *table* to match *target* exactly.

    - Columns in *target* absent from *table* are added as null arrays.
    - Columns in *table* absent from *target* are dropped.
    - Column order matches *target*.

    Reserved INTERLEAVED_SCHEMA columns allow ``safe=False`` casts so that
    explicit large↔small type overrides work (e.g. ``large_string``→``string``
    for Parquet compat).  Passthrough (user-defined) columns always use
    ``safe=True`` so that overflow errors surface rather than silently corrupt
    data (e.g. ``large_string``→``string`` on a >2 GB column).
    """
    existing = set(table.schema.names)
    arrays: list[pa.Array] = []
    for field in target:
        if field.name in existing:
            col = table.column(field.name)
            if col.type != field.type:
                if field.name in RESERVED_COLUMNS:
                    safe = not (
                        (pa.types.is_large_string(col.type) and pa.types.is_string(field.type))
                        or (pa.types.is_large_binary(col.type) and pa.types.is_binary(field.type))
                    )
                else:
                    safe = True  # passthrough columns: surface overflow rather than corrupt
                col = col.cast(field.type, safe=safe)
            arrays.append(col)
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.table(arrays, schema=target)
