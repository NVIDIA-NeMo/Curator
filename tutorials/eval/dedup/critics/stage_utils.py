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

"""Column helper shared by the critic stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pyarrow as pa


def replace_columns(table: pa.Table, columns: dict[str, pa.Array], *, drop: Iterable[str] = ()) -> pa.Table:
    """Remove `drop` and any existing columns named in `columns`, then append `columns`."""
    removed = {*drop, *columns}
    table = table.select([name for name in table.column_names if name not in removed])
    for name, values in columns.items():
        table = table.append_column(name, values)
    return table
