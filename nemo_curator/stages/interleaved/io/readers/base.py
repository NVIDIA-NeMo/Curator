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


@dataclass
class BaseInterleavedReader(ProcessingStage[FileGroupTask, InterleavedBatch]):
    """Base contract for interleaved readers.

    If *output_schema* is set, every output table is aligned to it
    (missing columns become nulls, extra columns are dropped, types reconciled).
    Otherwise only core-column types are reconciled.
    """

    read_kwargs: dict[str, Any] = field(default_factory=dict)
    output_schema: pa.Schema | None = None
    name: str = "base_interleaved_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sample_id", "position", "modality"]

    def _align_output(self, table: pa.Table) -> pa.Table:
        """Reconcile or align *table* to the declared output schema."""
        if self.output_schema is not None:
            return align_table(table, self.output_schema)
        return table.cast(reconcile_schema(table.schema))
