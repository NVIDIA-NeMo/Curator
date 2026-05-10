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

"""Unit tests for BaseStageAdapter._drop_completed_inputs.

No Ray cluster required — uses fake actor handles to exercise the filter path
in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


class _RecordingStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that records the inputs it receives and returns them unchanged."""

    name = "recording_stage"
    resources = Resources(cpus=0.1)

    def __init__(self) -> None:
        self.seen: list[DocumentBatch] = []

    def process(self, task: DocumentBatch) -> DocumentBatch:
        self.seen.append(task)
        return task


class _FakeRemoteCall:
    """Mimics `actor.method.remote(...)` returning an awaitable-ish ref."""

    def __init__(self, value: Any) -> None:  # noqa: ANN401
        self._value = value

    # _checkpoint_get -> ray.get(ref); we monkey-patch ray.get to call this.
    def _resolve(self) -> Any:  # noqa: ANN401
        return self._value


class _FakeMethod:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def remote(self, *args: Any, **kwargs: Any) -> _FakeRemoteCall:  # noqa: ANN401
        return _FakeRemoteCall(self._fn(*args, **kwargs))


class _FakeCheckpointActor:
    """Stand-in for the singleton _CheckpointActor handle."""

    def __init__(self, completed_pairs: set[tuple[str, str]]) -> None:
        self._completed = completed_pairs
        self.calls: list[list[tuple[str, str]]] = []
        self.are_leaves_completed = _FakeMethod(self._are_leaves_completed)

    def _are_leaves_completed(self, pairs: list[tuple[str, str]]) -> list[bool]:
        self.calls.append(list(pairs))
        return [(k, tk) in self._completed for k, tk in pairs]


@pytest.fixture(autouse=True)
def _patch_ray_get(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``ray.get`` resolve our fake remote refs without a Ray cluster."""
    import ray

    def _fake_get(ref: Any) -> Any:  # noqa: ANN401
        if isinstance(ref, _FakeRemoteCall):
            return ref._resolve()
        msg = f"_patch_ray_get only handles _FakeRemoteCall, got {type(ref)}"
        raise TypeError(msg)

    monkeypatch.setattr(ray, "get", _fake_get)


def _make_task(task_id: str, key: str, task_key: str) -> DocumentBatch:
    import pandas as pd

    return DocumentBatch(
        task_id=task_id,
        dataset_name="test",
        data=pd.DataFrame({"x": [1]}),
        _metadata={"resumability_key": key, "resumability_task_key": task_key},
    )


class TestDropCompletedInputs:
    def test_no_filtering_when_flag_is_false(self) -> None:
        stage = _RecordingStage()
        adapter = BaseStageAdapter(stage)
        actor = _FakeCheckpointActor(completed_pairs={("k", "k::0")})
        adapter._checkpoint_actor = actor
        # _is_last_user_stage left at default False

        tasks = [_make_task("t0", "k", "k::0"), _make_task("t1", "k", "k::1")]
        out = adapter.process_batch(tasks)

        assert len(out) == 2
        assert len(stage.seen) == 2
        assert actor.calls == []  # actor must not be touched

    def test_filters_completed_leaves_when_last_user_stage(self) -> None:
        stage = _RecordingStage()
        stage._is_last_user_stage = True
        adapter = BaseStageAdapter(stage)
        actor = _FakeCheckpointActor(completed_pairs={("k", "k::1")})
        adapter._checkpoint_actor = actor

        tasks = [
            _make_task("t0", "k", "k::0"),
            _make_task("t1", "k", "k::1"),  # should be dropped
            _make_task("t2", "k", "k::2"),
        ]
        out = adapter.process_batch(tasks)

        assert [t.task_id for t in out] == ["t0", "t2"]
        assert [t.task_id for t in stage.seen] == ["t0", "t2"]
        assert actor.calls == [[("k", "k::0"), ("k", "k::1"), ("k", "k::2")]]

    def test_short_circuits_when_all_filtered(self) -> None:
        stage = _RecordingStage()
        stage._is_last_user_stage = True
        adapter = BaseStageAdapter(stage)
        actor = _FakeCheckpointActor(completed_pairs={("k", "k::0"), ("k", "k::1")})
        adapter._checkpoint_actor = actor

        tasks = [_make_task("t0", "k", "k::0"), _make_task("t1", "k", "k::1")]
        out = adapter.process_batch(tasks)

        assert out == []
        assert stage.seen == []  # underlying stage never invoked

    def test_no_filtering_when_actor_missing(self) -> None:
        stage = _RecordingStage()
        stage._is_last_user_stage = True
        adapter = BaseStageAdapter(stage)
        # _checkpoint_actor stays None

        tasks = [_make_task("t0", "k", "k::0")]
        out = adapter.process_batch(tasks)

        assert len(out) == 1
        assert len(stage.seen) == 1

    def test_passes_through_tasks_missing_keys(self) -> None:
        import pandas as pd

        stage = _RecordingStage()
        stage._is_last_user_stage = True
        adapter = BaseStageAdapter(stage)
        actor = _FakeCheckpointActor(completed_pairs=set())
        adapter._checkpoint_actor = actor

        # Task with no resumability metadata at all — should pass through without an RPC.
        bare = DocumentBatch(task_id="bare", dataset_name="test", data=pd.DataFrame({"x": [1]}), _metadata={})
        out = adapter.process_batch([bare])

        assert len(out) == 1
        # No queryable pairs → no RPC issued.
        assert actor.calls == []
