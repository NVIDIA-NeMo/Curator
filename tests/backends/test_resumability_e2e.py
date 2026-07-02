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
"""End-to-end resumability against a REAL Ray cluster, executor, and
``pipeline.run(checkpoint_path=...)``.

The other resumability tests mock Ray (or use the undecorated actor class), so
they exercise the counter math + LMDB logic but are blind to the Ray-session
lifecycle: who connects the driver, when the detached actor is created, and
whether it survives the executor's own ``ray.init``/``ray.shutdown``. This test
drives the whole real path twice over one checkpoint dir and would fail if, for
example, the checkpoint actor were never created (the bug where the driver was
never connected before ``create_resumability_actor``). It relies on the
session-scoped ``shared_ray_cluster`` fixture, which sets ``RAY_ADDRESS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, FailedTask, Task

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _IntTask(Task[int]):
    data: int = 0

    @property
    def num_items(self) -> int:
        return 1

    def validate(self) -> bool:
        return True


@dataclass
class _Source(ProcessingStage[EmptyTask, _IntTask]):
    """Emits ``n`` source partitions (ids 0..n-1 by position)."""

    name: str = "source"
    n: int = 0
    is_source_stage: bool = True
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, _: EmptyTask) -> list[_IntTask]:
        return [_IntTask(data=i, dataset_name="d") for i in range(self.n)]


@dataclass
class _Sink(ProcessingStage[_IntTask, _IntTask]):
    """Sink that fails the listed partitions (returns ``FailedTask``)."""

    name: str = "sink"
    is_sink_stage: bool = True
    fail: tuple[int, ...] = ()
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: _IntTask) -> _IntTask | FailedTask:
        return FailedTask() if task.data in self.fail else task


def _run(checkpoint: Path, n: int, fail: tuple[int, ...] = ()) -> list[int]:
    """One real source->sink pass through ``pipeline.run`` with resumability.
    Returns the sorted ``data`` values that reached the sink output."""
    pipe = Pipeline(name="resumability_e2e")
    pipe.add_stage(_Source(n=n))
    pipe.add_stage(_Sink(fail=fail))
    out = pipe.run(executor=RayActorPoolExecutor(), checkpoint_path=str(checkpoint))
    return sorted(t.data for t in (out or []))


@pytest.mark.usefixtures("shared_ray_cluster")
def test_resume_skips_completed_and_reruns_failed(tmp_path: Path) -> None:
    ckpt = tmp_path / "ck"

    # Run 1: sources 0,1,2 emitted; source 1 fails at the sink. 0 and 2 complete
    # (persisted to the checkpoint); 1 stays pending. Only 0,2 reach the sink.
    assert _run(ckpt, n=3, fail=(1,)) == [0, 2]

    # Run 2 (same checkpoint dir): the source stage reads 0,2 as completed and
    # skips them; only source 1 reruns -- and now succeeds. If the actor/Ray
    # lifecycle were broken (e.g. actor never created), run 1 would not have
    # persisted anything and run 2 would reprocess everything ([0, 1, 2]).
    assert _run(ckpt, n=3, fail=()) == [1]


@pytest.mark.usefixtures("shared_ray_cluster")
def test_fully_completed_pipeline_reruns_nothing(tmp_path: Path) -> None:
    ckpt = tmp_path / "ck"

    # Run 1: everything succeeds -> all sources complete.
    assert _run(ckpt, n=3) == [0, 1, 2]

    # Run 2: every source is already complete, so the source stage emits nothing
    # downstream and the executor errors out on the empty stream.
    with pytest.raises(ValueError, match="No tasks to process"):
        _run(ckpt, n=3)
