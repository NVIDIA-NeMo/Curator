# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomic, backend-neutral task batches planned for one downstream dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

from .tasks import Task

if TYPE_CHECKING:
    from nemo_curator.utils.performance_utils import StagePerfStats


@dataclass
class DispatchBatchTask(Task[list[Task[Any]]]):
    """One planner-owned batch that backends must treat as an atomic row.

    Backends may place several ``DispatchBatchTask`` rows in one stage
    ``process_batch`` invocation, but they never inspect or split ``data``.
    The selected owner stage is responsible for making exactly one downstream
    adapter/model call for each non-empty dispatch batch.
    """

    batch_id: str = ""
    owner_stage: str = ""
    sequence_index: int = 0
    bucket_index: int = 0
    total_cost: float = 0.0
    item_costs: tuple[float, ...] = ()
    cost_unit: str = "items"
    policy_signature: str = ""

    @property
    def items(self) -> list[Task[Any]]:
        return self.data

    @property
    def num_items(self) -> int:
        return sum(item.num_items for item in self.data)

    def validate(self) -> bool:
        return all(
            (
                bool(self.batch_id and self.owner_stage),
                isinstance(self.data, list) and bool(self.data),
                all(isinstance(item, Task) for item in self.data),
                len(self.item_costs) == len(self.data),
                self.sequence_index >= 0 and self.bucket_index >= 0,
                bool(self.cost_unit and self.policy_signature),
                isfinite(float(self.total_cost)) and self.total_cost >= 0,
                all(isfinite(float(cost)) and cost >= 0 for cost in self.item_costs),
            )
        )

    def with_items(self, items: list[Task[Any]]) -> DispatchBatchTask:
        """Return the same dispatch envelope around transformed child tasks."""
        if len(items) != len(self.item_costs):
            msg = f"Dispatch batch {self.batch_id!r} expected {len(self.item_costs)} item(s), got {len(items)}"
            raise ValueError(msg)
        batch = DispatchBatchTask(
            dataset_name=self.dataset_name,
            data=items,
            _stage_perf=list(self._stage_perf),
            _metadata=dict(self._metadata),
            batch_id=self.batch_id,
            owner_stage=self.owner_stage,
            sequence_index=self.sequence_index,
            bucket_index=self.bucket_index,
            total_cost=self.total_cost,
            item_costs=self.item_costs,
            cost_unit=self.cost_unit,
            policy_signature=self.policy_signature,
        )
        batch.task_id = self.task_id
        return batch

    def flattened_items(self) -> list[Task[Any]]:
        """Return child tasks with dispatch-level performance records attached."""
        for item in self.data:
            _copy_missing_stage_perf(self._stage_perf, item)
        return list(self.data)


def _copy_missing_stage_perf(perf_records: list[StagePerfStats], task: Task[Any]) -> None:
    seen = {_perf_key(perf) for perf in task._stage_perf}
    for perf in perf_records:
        key = _perf_key(perf)
        if key in seen:
            continue
        task.add_stage_perf(perf)
        seen.add(key)


def _perf_key(perf: StagePerfStats) -> str:
    return str(getattr(perf, "invocation_id", "") or f"id:{id(perf)}")
