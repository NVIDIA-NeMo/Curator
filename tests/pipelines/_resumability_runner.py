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
"""Subprocess entry point for the SIGINT/resume integration test.

Builds a 4-stage pipeline (fanout -> passthrough -> chunked-fanin -> slow_writer)
and walks it stage-by-stage with a real :class:`LineageWriterActor` writing to
``--checkpoint-path``. Prints one ``completed`` line per terminal-stage emission
to stdout so the parent test can pace SIGINT injection deterministically.

Not a test itself; loaded as a subprocess from
:func:`test_resumable_after_sigint` in ``test_lineage_integration.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from dataclasses import dataclass

import ray

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage, assign_child_lineage
from nemo_curator.tasks import Task
from nemo_curator.utils.lineage_store import (
    LINEAGE_ACTOR_NAME,
    LineageWriterActor,
    record_lineage,
)


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data) if self.data is not None else 0

    def validate(self) -> bool:
        return True


@dataclass
class _FanOut(ProcessingStage[_SimpleTask, _SimpleTask]):
    times: int = 2000
    name: str = "fanout"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> list[_SimpleTask]:
        return [
            _SimpleTask(task_id=f"{task.task_id}_{i}", dataset_name=task.dataset_name, data=task.data)
            for i in range(self.times)
        ]


@dataclass
class _Passthrough(ProcessingStage[_SimpleTask, _SimpleTask]):
    name: str = "passthrough"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        return _SimpleTask(task_id=f"{task.task_id}_pt", dataset_name=task.dataset_name, data=task.data)


@dataclass
class _ChunkedFanIn(ProcessingStage[_SimpleTask, _SimpleTask]):
    """Fan-in that chunks its input into groups of ``fanin_size`` and merges each
    group into one output. Overrides ``process_batch`` (multi-parent emission),
    so it must call :meth:`_filter_completed_tasks`, :func:`assign_child_lineage`,
    and :func:`record_lineage` itself."""

    fanin_size: int = 20
    name: str = "fanin"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        _ = task
        msg = "ChunkedFanIn only supports batched execution"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[_SimpleTask]) -> list[_SimpleTask]:
        tasks = self._filter_completed_tasks(tasks)
        results: list[_SimpleTask] = []
        for start in range(0, len(tasks), self.fanin_size):
            chunk = tasks[start : start + self.fanin_size]
            combined: list[int] = []
            for t in chunk:
                combined.extend(t.data)
            merged = _SimpleTask(task_id=f"merged_{start}", dataset_name=chunk[0].dataset_name, data=combined)
            children = assign_child_lineage([t._lineage_path for t in chunk], merged)
            record_lineage([t._udid for t in chunk], [c._udid for c in children])
            results.extend(children)
        return results


@dataclass
class _SlowWriter(ProcessingStage[_SimpleTask, _SimpleTask]):
    """Terminal stage with a per-task sleep so SIGINT can land mid-batch. Emits
    one stdout line per processed task so the parent test can pace the signal."""

    sleep_s: float = 0.05
    name: str = "slow_writer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        time.sleep(self.sleep_s)
        out = _SimpleTask(task_id=f"{task.task_id}_w", dataset_name=task.dataset_name, data=task.data)
        sys.stdout.write("completed\n")
        sys.stdout.flush()
        return out


def _drive(pipeline: Pipeline, initial_tasks: list[Task]) -> list[Task]:
    current = initial_tasks
    for stage in pipeline.stages:
        current = BaseStageAdapter(stage).process_batch(current)
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--n-tasks", type=int, default=2000)
    parser.add_argument("--fanin-size", type=int, default=20)
    parser.add_argument("--writer-sleep-s", type=float, default=0.05)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    with contextlib.suppress(ValueError):
        ray.kill(ray.get_actor(LINEAGE_ACTOR_NAME))
    actor = LineageWriterActor.options(
        name=LINEAGE_ACTOR_NAME,
        get_if_exists=True,
    ).remote(path=args.checkpoint_path)

    pipeline = Pipeline(
        name="resumable",
        stages=[
            _FanOut(times=args.n_tasks),
            _Passthrough(),
            _ChunkedFanIn(fanin_size=args.fanin_size),
            _SlowWriter(sleep_s=args.writer_sleep_s),
        ],
    )
    pipeline.build()
    root = _SimpleTask(task_id="r", dataset_name="d", data=[1])

    try:
        _drive(pipeline, [root])
    except KeyboardInterrupt:
        sys.stdout.write("interrupted\n")
        sys.stdout.flush()
    finally:
        with contextlib.suppress(Exception):
            ray.get(actor.close.remote())
        with contextlib.suppress(Exception):
            ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
