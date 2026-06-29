# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.core.utils import ignore_ray_head_node
from nemo_curator.tasks import Task
from nemo_curator.tasks.sentinels import FailedTask, NoneTask
from nemo_curator.utils.performance_utils import StageTimer
from nemo_curator.utils.resumability_client import (
    completed_resumability_sources,
    flush_resumability_deltas,
    is_resumability_actor_active,
)

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


def _is_sentinel(task: Task) -> bool:
    """A payload-less marker (NoneTask/FailedTask), stripped before the next stage."""
    return isinstance(task, (NoneTask, FailedTask))


@dataclass
class NodeInfo:
    """Generic node information for setup_on_node calls across backends.
    Simplified to match Xenna's structure.
    """

    node_id: str = ""


@dataclass
class WorkerMetadata:
    """Generic worker metadata for setup_on_node calls across backends.
    Simplified to match Xenna's structure. The allocation field can contain
    backend-specific allocation information.
    """

    worker_id: str = ""
    allocation: Any = None  # Backend-specific allocation info


class BaseExecutor(ABC):
    """Executor for a pipeline."""

    def __init__(self, config: dict[str, Any] | None = None, ignore_head_node: bool = False):
        self.config = config or {}
        self.ignore_head_node = ignore_head_node or ignore_ray_head_node()

    @abstractmethod
    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> None:
        """Execute the pipeline."""


class BaseStageAdapter:
    """Adapts ProcessingStage to an execution backend, if needed."""

    def __init__(self, stage: "ProcessingStage"):
        self.stage = stage

    def process_batch(self, tasks: list[Task]) -> list[Task]:
        """Process a batch of tasks.

        Args:
            tasks (list[Task]): List of tasks to process

        Returns:
            list[Task]: List of processed tasks
        """
        # Lazy initialize timer if needed
        if not hasattr(self, "_timer") or self._timer is None:
            self._timer = StageTimer(self.stage)

        # Calculate input data size for timer
        input_size = sum(task.num_items for task in tasks)
        # Initialize performance timer for this batch
        self._timer.reinit(input_size)

        with self._timer.time_process(input_size):
            # Use the batch processing logic
            results = self.stage.process_batch(tasks)

        # Replace a returned None ("filter this slot") with a NoneTask so every
        # output gets a task_id; sentinels are stripped again below.
        results = [NoneTask() if r is None else r for r in results]

        # Assign every emitted task a task_id (derived, or uuid fallback).
        results = self._post_process_task_ids(tasks, results)

        # Opt-in resumability: fire per-source deltas (no-op when no actor registered).
        if is_resumability_actor_active():
            results = self._apply_resumability_counters(tasks, results)

        # Sentinels never propagate to the next stage.
        results = [r for r in results if not _is_sentinel(r)]

        # Log performance stats and add to result tasks
        _, stage_perf_stats = self._timer.log_stats()
        # Consume and attach any custom metrics recorded by the stage during this call
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)
        for task in results:
            task.add_stage_perf(stage_perf_stats)

        return results

    def _post_process_task_ids(self, input_tasks: list[Task], output_tasks: list[Task | None]) -> list[Task]:
        """Assign a deterministic ``task_id`` (parent id + own segment) to every
        emitted task. Runs once per stage on every backend, so ``process`` vs
        ``process_batch`` makes no difference; ids are re-derived at each stage
        boundary, so one object passing through N stages gets N ids.

        - single input → fan-out: each output is ``parent_<seg>``
        - ``len(output) == len(input)`` → positional 1:1: ``parent_i_<seg>``; a
          ``None`` slot means input ``i`` was filtered (kept for alignment, then
          dropped from the result)
        - any other cardinality → a random ``"r"``-prefixed uuid (non-deterministic,
          ancestry-not-tracked; see ``Task.task_id``)

        ``seg`` is the content id (``get_deterministic_id()``) for a source stage,
        else the positional index. A stage that both filters and fans out in one
        batch can't be mapped positionally and falls to the ``"r"`` case — return
        one value (or ``None``) per input to stay positional.
        """
        is_source = getattr(self.stage, "is_source_stage", False)

        if len(input_tasks) == 1:
            # Fan-out (incl. a source reading from EmptyTask): every non-None
            # output is a child of the single input.
            parent_id = input_tasks[0].task_id
            out: list[Task] = [t for t in output_tasks if t is not None]
            for i, task in enumerate(out):
                suffix = (task.get_deterministic_id() or i) if is_source else i
                task._set_task_id(parent_id, suffix)
            return out

        if len(output_tasks) == len(input_tasks):
            # Positional 1:1. None is kept above so a filtered slot still lines
            # up with its own parent; drop the None slots from the result.
            out = []
            for parent, task in zip(input_tasks, output_tasks, strict=True):
                if task is None:
                    continue
                suffix = (task.get_deterministic_id() or 0) if is_source else 0
                task._set_task_id(parent.task_id, suffix)
                out.append(task)
            return out

        # Ambiguous cardinality across a batch: a derived id is not possible. Use a
        # random "r"-prefixed uuid so task_id is non-empty but clearly flagged
        # non-deterministic.
        out = [t for t in output_tasks if t is not None]
        for task in out:
            task.task_id = "r" + uuid.uuid4().hex
        return out

    # Resumability (opt-in): stamp _source_id, fire per-source deltas, drop
    # completed sources. task_ids are already assigned; sentinels stripped by caller.
    def _apply_resumability_counters(self, input_tasks: list[Task], output_tasks: list[Task]) -> list[Task]:  # noqa: C901
        # Dedup key is always an OUTPUT task_id, never the input's: the source
        # already keyed its +1 on that id, and an output id is one level deeper,
        # so it's unique to the (task, stage) that produced it.
        stage = self.stage
        if getattr(stage, "is_source_stage", False):
            return self._source_counters(output_tasks)

        # No outputs (e.g. a batch entirely filtered, or an end-of-pipeline
        # no-op): nothing to attribute a delta to, so skip.
        if not output_tasks:
            return output_tasks

        # Pre-source: inputs have no _source_id yet; nothing to track.
        if all(not t._source_id for t in input_tasks):
            return output_tasks

        is_sink = stage.is_sink_stage
        per_task: list[tuple[str, str, int]] = []

        if len(input_tasks) == 1 and len(output_tasks) > 1:
            # Fan-out (1->N): parent consumed (-1); each real child continues
            # (+1, or 0 at a sink); each FailedTask keeps the source open (+1);
            # NoneTask contributes 0.
            parent = input_tasks[0]
            real = [t for t in output_tasks if not _is_sentinel(t)]
            n_failed = sum(1 for t in output_tasks if isinstance(t, FailedTask))
            continuing = 0 if is_sink else len(real)
            delta = continuing + n_failed - 1
            # Key on output[0].task_id (not parent.task_id, which collides with the
            # source's +1). Non-source children are indexed positionally, so
            # output[0] is always "<parent>_0".
            per_task.append((output_tasks[0].task_id, parent._source_id, delta))
            for c in real:
                if not c._source_id:
                    c._source_id = parent._source_id
        elif len(output_tasks) == len(input_tasks):
            # Positional 1:1; each delta keys on the output id (r.task_id).
            for parent, r in zip(input_tasks, output_tasks, strict=True):
                sid = parent._source_id
                if isinstance(r, NoneTask):  # filtered -> consumed
                    per_task.append((r.task_id, sid, -1))
                    continue
                if isinstance(r, FailedTask):  # failed -> source stays open (no sink test)
                    per_task.append((r.task_id, sid, 0))
                    continue
                per_task.append((r.task_id, sid, -1 if is_sink else 0))  # real: sink -1, else 0
                if not r._source_id:
                    r._source_id = sid
        else:
            # M->K (M!=K): can't attribute parents; skip (source stays pending -> reprocessed).
            logger.warning(
                f"resumability: {type(stage).__name__} produced {len(output_tasks)} outputs "
                f"for {len(input_tasks)} inputs; can't attribute sources, skipping counter "
                f"update for this batch."
            )
            return output_tasks

        flush_resumability_deltas(per_task)
        return output_tasks

    def _source_counters(self, output_tasks: list[Task]) -> list[Task]:
        """Source stage: each output is a source partition; its ``_source_id`` is
        ``Task.get_source_id()``. Drop already-completed sources; each survivor fires ``+1``."""
        sources = [t for t in output_tasks if not _is_sentinel(t)]
        for t in sources:
            t._source_id = t.get_source_id()
        completed = completed_resumability_sources([t._source_id for t in sources])
        per_task: list[tuple[str, str, int]] = []
        survivors: list[Task] = []
        for t in sources:
            if t._source_id in completed:
                continue
            per_task.append((t.task_id, t._source_id, +1))
            survivors.append(t)
        flush_resumability_deltas(per_task)
        return survivors

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup the stage on a node.

        Args:
            node_info (NodeInfo, optional): Information about the node
            worker_metadata (WorkerMetadata, optional): Information about the worker
        """
        # Call the underlying stage's setup_on_node method
        # Some backends may provide node/worker info, others may not
        self.stage.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup the stage once per actor.

        Args:
            worker_metadata (WorkerMetadata, optional): Information about the worker
        """
        self.stage.setup(worker_metadata)

    def teardown(self) -> None:
        """Teardown the stage once per actor."""
        self.stage.teardown()
