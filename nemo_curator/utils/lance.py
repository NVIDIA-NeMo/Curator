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

import lance
import pyarrow as pa
import pyarrow.compute as pc

LANCE_ROWADDR_COLUMN = "__lance_rowaddr"
LANCE_ROWID_COLUMN = "__lance_rowid"
LANCE_FRAGID_COLUMN = "__lance_fragid"


def add_lance_metadata_columns(table: pa.Table) -> pa.Table:
    missing = [name for name in ("_rowid", "_rowaddr") if name not in table.column_names]
    if missing:
        msg = f"Lance scanner did not return {missing}; include_lance_metadata requires row ids and addresses"
        raise ValueError(msg)

    renamed = {
        "_rowid": LANCE_ROWID_COLUMN,
        "_rowaddr": LANCE_ROWADDR_COLUMN,
    }
    table = table.rename_columns([renamed.get(name, name) for name in table.column_names])
    row_addresses = table[LANCE_ROWADDR_COLUMN].combine_chunks().cast(pa.uint64())
    fragment_ids = pc.shift_right(row_addresses, pa.scalar(32, type=pa.uint64())).cast(pa.uint64())
    return table.append_column(LANCE_FRAGID_COLUMN, fragment_ids)


def materialize_lance_blob_columns(dataset: lance.LanceDataset, table: pa.Table) -> pa.Table:
    """Replace scanned Blob v2 descriptors with binary payloads."""
    row_addresses = [int(value) for value in table["_rowaddr"].combine_chunks().to_pylist()]
    for field in dataset.schema:
        column_index = table.schema.get_field_index(field.name)
        if column_index < 0 or getattr(field.type, "extension_name", None) != "lance.blob.v2":
            continue
        # read_blobs may omit nulls, so align returned payloads to scanned rows by address.
        payloads_by_address = dict(dataset.read_blobs(field.name, addresses=row_addresses))
        payloads = pa.array(
            [payloads_by_address.get(row_address) for row_address in row_addresses], type=pa.large_binary()
        )
        output_field = pa.field(field.name, pa.large_binary(), nullable=field.nullable)
        table = table.set_column(column_index, output_field, payloads)
    return table
