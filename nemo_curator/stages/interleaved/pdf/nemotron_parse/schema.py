# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-independent Arrow contract for page-budget document outputs."""

import pyarrow as pa

from nemo_curator.stages.interleaved.utils.schema import align_interleaved_table
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

PAGE_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field(
            field.name,
            pa.large_string() if pa.types.is_string(field.type) else field.type,
            nullable=field.nullable,
        )
        for field in INTERLEAVED_SCHEMA
    ]
    + [
        pa.field("page_number", pa.int64()),
        pa.field("element_class", pa.large_string()),
        pa.field("url", pa.large_string()),
        pa.field("pdf_name", pa.large_string(), nullable=False),
        pa.field("document_id", pa.large_string(), nullable=False),
    ]
)


def align_page_output(table: pa.Table) -> pa.Table:
    """Align known PDF fields without silently discarding undeclared columns."""
    unexpected = set(table.column_names) - set(PAGE_OUTPUT_SCHEMA.names)
    if unexpected:
        msg = f"Undeclared page-budget output columns: {sorted(unexpected)}"
        raise ValueError(msg)
    return align_interleaved_table(table, PAGE_OUTPUT_SCHEMA)
