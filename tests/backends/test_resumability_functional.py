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
"""End-to-end resume loop without a Ray cluster: drives the real
``BaseStageAdapter`` against a real LMDB-backed ``ResumabilityActor`` over two
runs sharing a checkpoint dir. Exercises the full counter → actor → LMDB → skip
loop (and would fail under the old parent-id keying bug, where completed sources
never persisted).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import EmptyTask, FailedTask, Task
from nemo_curator.utils.resumability_actor import ResumabilityActor

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _new_actor(base_dir: Path, writer_id: str):  # noqa: ANN202  (undecorated Ray actor class instance)
    """A real actor instance (undecorated class — no Ray cluster needed),
    writing its own ``<writer_id>.mdb`` and reading the union on startup."""
    cls = ResumabilityActor.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    return cls(str(base_dir), writer_id=writer_id)


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
    name: str = "source"
    n: int = 0
    is_source_stage: bool = True

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, _: EmptyTask) -> list[_IntTask]:
        return [_IntTask(data=i, dataset_name="d") for i in range(self.n)]


@dataclass
class _Sink(ProcessingStage[_IntTask, _IntTask]):
    name: str = "sink"
    is_sink_stage: bool = True
    fail: tuple[int, ...] = ()
    none: tuple[int, ...] = ()
    newobj: bool = False

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: _IntTask) -> _IntTask | FailedTask | None:
        if task.data in self.fail:
            return FailedTask()
        if task.data in self.none:
            return None
        return _IntTask(data=task.data, dataset_name="d") if self.newobj else task


@contextmanager
def _wired(actor) -> Iterator[None]:  # noqa: ANN001
    """Point the worker-side client helpers at ``actor`` directly (no Ray)."""

    def _skip(sids: list[str]) -> set[str]:
        return {s for s, done in zip(sids, actor.are_completed(sids), strict=True) if done}

    with (
        patch("nemo_curator.backends.base.is_resumability_actor_active", return_value=True),
        patch("nemo_curator.backends.base.flush_resumability_deltas", side_effect=actor.apply_deltas),
        patch("nemo_curator.backends.base.completed_resumability_sources", side_effect=_skip),
    ):
        yield


def _run(actor, n: int, **sink_kwargs) -> tuple[list[str], list[str]]:  # noqa: ANN001
    """One source→sink pass through the real adapters. Returns
    ``(source ids that survived the source stage, sink-output source ids)``."""
    with _wired(actor):
        src_out = BaseStageAdapter(_Source(n=n)).process_batch([EmptyTask()])
        src_ids = sorted(t._source_id for t in src_out)
        sink_out = BaseStageAdapter(_Sink(**sink_kwargs)).process_batch(src_out)
    return src_ids, sorted(t._source_id for t in sink_out)


class TestResumeLoop:
    def test_completed_skip_and_failed_reruns(self, tmp_path: Path) -> None:
        # Run 1: sources 0,1,2; source 1 fails at the sink.
        a1 = _new_actor(tmp_path, "w1")
        src1, _ = _run(a1, n=3, fail=(1,))
        assert src1 == ["0", "1", "2"]  # all emitted on the first run
        assert a1.are_completed(["0", "1", "2"]) == [True, False, True]  # 1 stays pending
        a1.close()

        # Run 2: fresh actor reads w1's completions from the union; source 1
        # now succeeds. Only the previously-failed source reruns.
        a2 = _new_actor(tmp_path, "w2")
        src2, sink2 = _run(a2, n=3)
        assert src2 == ["1"]  # 0 and 2 skipped at the source stage
        assert sink2 == ["1"]
        assert a2.are_completed(["0", "1", "2"]) == [True, True, True]
        a2.close()

    def test_new_object_sink_completes(self, tmp_path: Path) -> None:
        # Sink returns a NEW object (not the input) — the counter must key on the
        # output id, not the input's, or the source's +1 and the sink's -1 would
        # collide and the source would never complete. This locks that fix.
        a1 = _new_actor(tmp_path, "w1")
        _run(a1, n=3, newobj=True)
        assert a1.are_completed(["0", "1", "2"]) == [True, True, True]
        a1.close()

        a2 = _new_actor(tmp_path, "w2")
        src2, _ = _run(a2, n=3, newobj=True)
        assert src2 == []  # everything already complete -> nothing reruns
        a2.close()

    def test_none_filtered_source_completes(self, tmp_path: Path) -> None:
        # A source filtered to None at the sink is consumed -> completes (it must
        # NOT behave like Failed). It is skipped, not rerun, on the next run.
        a1 = _new_actor(tmp_path, "w1")
        _, sink1 = _run(a1, n=3, none=(1,))
        assert sink1 == ["0", "2"]  # source 1 produced no sink output (filtered)
        assert a1.are_completed(["0", "1", "2"]) == [True, True, True]  # but it completed
        a1.close()

        a2 = _new_actor(tmp_path, "w2")
        src2, _ = _run(a2, n=3, none=(1,))
        assert src2 == []  # source 1 was completed by the filter, not left pending
        a2.close()
