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

import pyarrow as pa

LANCE_ROWADDR_COLUMN = "__lance_rowaddr"
LANCE_ROWID_COLUMN = "__lance_rowid"
LANCE_FRAGID_COLUMN = "__lance_fragid"


def lance_fragment_ids_from_row_addresses(rowaddr_column: pa.ChunkedArray) -> pa.Array:
    rowaddrs = rowaddr_column.combine_chunks().cast(pa.uint64())
    return pa.array([int(value) >> 32 for value in rowaddrs.to_pylist()], type=pa.uint64())


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
    return table.append_column(LANCE_FRAGID_COLUMN, lance_fragment_ids_from_row_addresses(table[LANCE_ROWADDR_COLUMN]))
