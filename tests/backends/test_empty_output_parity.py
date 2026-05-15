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

"""Backend parity: a stage that emits no output must not break the pipeline.

The resumability mock pipeline mixes this with fan-out, filter, batching and
checkpointing — useful as an integration test, but if Ray Data fails it is hard
to tell which axis broke it. This file isolates the single question: when a
user stage emits zero outputs (per-task ``None`` or vectorized empty
``process_batch``), does the pipeline finish cleanly on both ``XennaExecutor``
and ``RayDataExecutor``?

Scenarios covered, parametrized over both executors:

  * ``test_drop_all_via_process_returns_none`` — every ``process()`` call in
    the middle stage returns ``None``. The default ``process_batch`` therefore
    returns ``[]``. Downstream stages must accept the empty input and the
    pipeline must return an empty result list.
  * ``test_drop_all_via_process_batch_returns_empty`` — same shape but the
    middle stage overrides ``process_batch`` and returns ``[]`` directly,
    skipping the per-task ``None`` path.
  * ``test_drop_some_via_process_returns_none`` — partial drop: half the
    inputs return ``None``. Verifies that the empty-output bug fix did not
    regress the normal filter case.

No checkpoint path is set — this is purely about whether empty batches flow
through the backend without crashing or losing the partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task, _EmptyTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import BaseExecutor


NUM_TASKS = 4
ROWS_PER_TASK = 8


@dataclass
class RowTask(Task[pd.DataFrame]):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


class _SourceStage(ProcessingStage[_EmptyTask, RowTask]):
    """Emit ``NUM_TASKS`` partitions, each with ``ROWS_PER_TASK`` rows."""

    name = "source"
    resources = Resources(cpus=0.5)

    def is_source_stage(self) -> bool:
        return True

    def process(self, _: _EmptyTask) -> list[RowTask]:
        return [
            RowTask(
                task_id=f"part_{i}",
                dataset_name="empty_parity",
                data=pd.DataFrame({"row_idx": range(ROWS_PER_TASK), "src": [i] * ROWS_PER_TASK}),
            )
            for i in range(NUM_TASKS)
        ]


class _DropAllStage(ProcessingStage[RowTask, RowTask]):
    """Per-task drop: ``process()`` returns ``None`` for every input.

    The default ``process_batch`` loop therefore yields ``[]``. On Ray Data
    this surfaces in the adapter as ``{"item": []}`` — the case under test.
    """

    name = "drop_all"
    resources = Resources(cpus=0.5)

    def process(self, task: RowTask) -> RowTask | None:
        return None


class _DropAllBatchedStage(ProcessingStage[RowTask, RowTask]):
    """Vectorized variant: overrides ``process_batch`` to return ``[]`` directly."""

    name = "drop_all_batched"
    resources = Resources(cpus=0.5)
    batch_size = 2

    def process(self, task: RowTask) -> RowTask:  # required by ABC; unused
        return task

    def process_batch(self, tasks: list[RowTask]) -> list[RowTask]:
        return []


class _DropEvenStage(ProcessingStage[RowTask, RowTask]):
    """Partial drop: keep odd ``src`` partitions, drop even ones.

    Guards against the empty-batch fix accidentally short-circuiting valid output.
    """

    name = "drop_even"
    resources = Resources(cpus=0.5)

    def process(self, task: RowTask) -> RowTask | None:
        if int(task.data["src"].iloc[0]) % 2 == 0:
            return None
        return task


class _SinkStage(ProcessingStage[RowTask, RowTask]):
    """Pass-through. Used to force a stage AFTER the drop so the empty
    output has to flow through one more backend hop before the executor
    returns it to the driver.
    """

    name = "sink"
    resources = Resources(cpus=0.5)

    def process(self, task: RowTask) -> RowTask:
        return task


def _make_executor(kind: str) -> BaseExecutor:
    if kind == "xenna":
        return XennaExecutor()
    if kind == "ray_data":
        return RayDataExecutor()
    msg = f"unknown executor kind {kind!r}"
    raise ValueError(msg)


EXECUTORS = [
    pytest.param("xenna", id="xenna"),
    pytest.param("ray_data", id="ray_data"),
]


@pytest.mark.parametrize("executor_kind", EXECUTORS)
def test_drop_all_via_process_returns_none(
    shared_ray_client: None,  # noqa: ARG001
    executor_kind: str,
) -> None:
    """Every ``process()`` returns ``None`` -> empty batch must flow through to driver."""
    pipeline = Pipeline(name="drop_all_none")
    pipeline.add_stage(_SourceStage())
    pipeline.add_stage(_DropAllStage())
    pipeline.add_stage(_SinkStage())

    out = pipeline.run(_make_executor(executor_kind))

    assert out is not None, "pipeline.run must not return None"
    assert len(out) == 0, f"all inputs were dropped; expected 0 output tasks, got {len(out)}"


@pytest.mark.parametrize("executor_kind", EXECUTORS)
def test_drop_all_via_process_batch_returns_empty(
    shared_ray_client: None,  # noqa: ARG001
    executor_kind: str,
) -> None:
    """``process_batch`` returns ``[]`` directly -> same expectation."""
    pipeline = Pipeline(name="drop_all_batch")
    pipeline.add_stage(_SourceStage())
    pipeline.add_stage(_DropAllBatchedStage())
    pipeline.add_stage(_SinkStage())

    out = pipeline.run(_make_executor(executor_kind))

    assert out is not None
    assert len(out) == 0


@pytest.mark.parametrize("executor_kind", EXECUTORS)
def test_drop_some_via_process_returns_none(
    shared_ray_client: None,  # noqa: ARG001
    executor_kind: str,
) -> None:
    """Half the inputs drop; the other half must survive end-to-end.

    Anti-regression for the empty-output fix: a backend that handles ``[]``
    by skipping the partition entirely would also drop the survivors,
    silently zeroing out the output.
    """
    pipeline = Pipeline(name="drop_some")
    pipeline.add_stage(_SourceStage())
    pipeline.add_stage(_DropEvenStage())
    pipeline.add_stage(_SinkStage())

    out = pipeline.run(_make_executor(executor_kind))

    assert out is not None
    expected = sum(1 for i in range(NUM_TASKS) if i % 2 == 1)
    assert len(out) == expected, f"expected {expected} surviving partitions, got {len(out)}"
    survivor_srcs = sorted(int(t.data["src"].iloc[0]) for t in out)
    assert survivor_srcs == [i for i in range(NUM_TASKS) if i % 2 == 1]
