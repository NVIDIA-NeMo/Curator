# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Backend-neutral envelopes for one atomic model dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from .tasks import Task


@dataclass
class DispatchBatchTask(Task[list[Task[Any]]]):
    """A planner-owned batch whose child list must never be backend-split."""

    batch_id: str = ""
    owner_stage: str = ""
    sequence_index: int = 0
    bucket_index: int = 0
    total_cost: float = 0.0
    item_costs: tuple[float, ...] = ()
    cost_unit: str = "seconds"
    policy_signature: str = ""

    @property
    def items(self) -> list[Task[Any]]:
        return self.data

    @property
    def num_items(self) -> int:
        return sum(item.num_items for item in self.items)

    def validate(self) -> bool:
        return all(
            (
                bool(self.batch_id),
                bool(self.owner_stage),
                isinstance(self.data, list) and bool(self.data),
                all(isinstance(item, Task) for item in self.data),
                len(self.item_costs) == len(self.data),
                self.sequence_index >= 0,
                self.bucket_index >= 0,
                bool(self.cost_unit),
                bool(self.policy_signature),
                isfinite(float(self.total_cost)) and self.total_cost >= 0,
                all(isfinite(float(cost)) and cost >= 0 for cost in self.item_costs),
            )
        )

    def with_items(self, items: list[Task[Any]]) -> DispatchBatchTask:
        """Rebuild the envelope without changing its dispatch contract."""
        if len(items) != len(self.item_costs):
            msg = f"Dispatch batch {self.batch_id!r} expected {len(self.item_costs)} items, got {len(items)}"
            raise ValueError(msg)
        result = DispatchBatchTask(
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
        result.task_id = self.task_id
        return result
