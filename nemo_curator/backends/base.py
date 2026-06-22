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
from nemo_curator.utils.resumability_client import _flush_deltas, _is_active, _skip_completed_sources

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


def _is_sentinel(task: Task) -> bool:
    """A payload-less marker (NoneTask/FailedTask) that is stripped before the
    next stage rather than propagated."""
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

        # A returned ``None`` ("filter this slot") becomes a NoneTask so every
        # output is a real Task that gets a task_id. Sentinels (NoneTask /
        # FailedTask) carry no identity and are stripped again before this
        # method returns.
        results = [NoneTask() if r is None else r for r in results]

        # Guarantee every emitted task has a task_id (derived id, or uuid fallback).
        results = self._post_process_task_ids(tasks, results)

        # Opt-in resumability: fire per-source counter deltas. A no-op (the
        # client helpers self-disable) when no resumability actor is registered.
        if _is_active():
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
        """Assign a deterministic ``task_id`` to every emitted task.

        This is the single place task ids are assigned — it runs for every
        stage on every backend (all backend adapters subclass this), so it
        makes no difference whether a stage defines ``process`` or overrides
        ``process_batch``. ``task_id`` is the task's id path (parents + own segment); ids are
        re-derived at each stage boundary so the same object passing through
        N stages gets N ids.

        The input→output mapping decides each output's PARENT; whether the
        stage is a source decides each output's SEGMENT (content id vs index)
        — the two are independent. ``None`` outputs (Curator's "return None to
        filter") are NOT removed before the length check — keeping them in
        place preserves positional alignment for filter stages — and are then
        dropped from the returned list.

        - single input → every output is its child (fan-out): ``parent_<seg>``
        - ``len(output) == len(input)`` → positional 1:1: each ``parent_i_<seg>``;
          a ``None`` slot just means input ``i`` was filtered.
        - any other (ambiguous) cardinality across a batch → a random ``uuid``
          prefixed with ``"r"`` (e.g. ``"r3f9a…"``), so ``task_id`` is never
          empty even when a derived id is not possible. The ``"r"`` prefix flags
          the id as non-deterministic / ancestry-not-tracked (see
          ``Task.task_id`` docstring).

        ``seg`` is the output's content id (``Task.get_deterministic_id()``)
        for a source stage when available, else the positional index — so a
        source partition keeps a stable id across reorderings regardless of
        whether the source is 1→N or N→N.

        Note: a stage that BOTH filters and fans out within a single batch
        (returning a flat list rather than a per-input slot) cannot be mapped
        positionally; if its length happens to equal the input length the 1:1
        assumption may misattribute parents. Such a stage falls through to the
        ambiguous-cardinality branch above (random ``r``-prefixed ids), so its
        outputs are not ancestry-tracked. To stay positional, a filtering stage
        should return one value (or ``None``) per input rather than a flat list.
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

    # ------------------------------------------------------------------ #
    # Resumability (opt-in). Runs only when a resumability actor is
    # registered. task_ids are already assigned by _post_process_task_ids;
    # this layer only stamps _source_id, fires per-source counter deltas, and
    # drops already-completed sources. Sentinels are stripped by the caller.
    # ------------------------------------------------------------------ #
    def _apply_resumability_counters(self, input_tasks: list[Task], output_tasks: list[Task]) -> list[Task]:  # noqa: C901
        # Every delta's dedup key is an OUTPUT task_id, never an input's
        # (``parent.task_id``). The source fires ``+1`` keyed on its output
        # partition's id; that id is the *input* id of the next stage, so keying
        # a downstream delta on the input would reuse the source's key and the
        # actor would treat the two as one conflicting event. An output id is
        # always one level deeper, so it's unique to the (task, stage) that
        # produced it.
        stage = self.stage
        if getattr(stage, "is_source_stage", False):
            return self._source_counters(output_tasks)

        # No outputs at all. Filtering is expressed as None -> NoneTask (a kept
        # slot), so a stage that emits nothing is degenerate; there is no output
        # to key a delta on, so skip (like the ambiguous-cardinality case).
        if not output_tasks:
            return output_tasks

        # Pre-source stages: inputs carry no _source_id, so there's nothing to
        # track yet. Leave outputs untouched.
        if all(not t._source_id for t in input_tasks):
            return output_tasks

        is_sink = stage.is_sink_stage
        per_task: list[tuple[str, str, int]] = []
        real = [t for t in output_tasks if not _is_sentinel(t)]

        if len(input_tasks) == 1 and len(output_tasks) != 1:
            # Genuine fan-out (1 -> N, N != 1). One net delta for the parent it
            # is consumed (-1); each real child continues (+1) unless this is a
            # sink, where children leave the pipeline (0); each FailedTask keeps
            # the source open (+1); NoneTask contributes nothing. sink and
            # fan-out are independent, so the sink test applies here too.
            parent = input_tasks[0]
            n_failed = sum(1 for t in output_tasks if isinstance(t, FailedTask))
            continuing = 0 if is_sink else len(real)
            delta = continuing + n_failed - 1
            # Key on output_tasks[0].task_id (NOT parent.task_id, which collides
            # with the source's +1). It always ends in "_0": get_deterministic_id()
            # is consulted only for source stages (which return via
            # _source_counters and never reach here), so non-source children are
            # indexed positionally (suffix 0, 1, ...) -> output[0] is "<parent>_0".
            per_task.append((output_tasks[0].task_id, parent._source_id, delta))
            for c in real:
                if not c._source_id:
                    c._source_id = parent._source_id
        elif len(output_tasks) == len(input_tasks):
            # Positional 1:1, including filtered (NoneTask) / failed slots. Each
            # delta keys on the OUTPUT id (r.task_id).
            for parent, r in zip(input_tasks, output_tasks, strict=True):
                sid = parent._source_id
                if isinstance(r, NoneTask):
                    # Filtered: this slot is consumed.
                    per_task.append((r.task_id, sid, -1))
                    continue
                if isinstance(r, FailedTask):
                    # Failed: leave the source open so it reruns (no sink test).
                    per_task.append((r.task_id, sid, 0))
                    continue
                # Real: a sink consumes it (-1); otherwise it passes through (0).
                per_task.append((r.task_id, sid, -1 if is_sink else 0))
                if not r._source_id:
                    r._source_id = sid
        else:
            # M inputs -> K outputs (K != M): the parent of each output can't be
            # determined, so the counter can't be updated correctly. Skip
            # (the source counter stays pending -> reprocessed on resume).
            logger.warning(
                f"resumability: {type(stage).__name__} produced {len(output_tasks)} outputs "
                f"for {len(input_tasks)} inputs; can't attribute sources, skipping counter "
                f"update for this batch."
            )
            return output_tasks

        _flush_deltas(per_task)
        return output_tasks

    def _source_counters(self, output_tasks: list[Task]) -> list[Task]:
        """Source stage: each output is a source partition. Its ``_source_id``
        is its own (last) id segment — the content id or index assigned by
        ``_post_process_task_ids``. Already-completed sources are dropped; each
        surviving source fires a ``+1``."""
        sources = [t for t in output_tasks if not _is_sentinel(t)]
        for t in sources:
            t._source_id = t.task_id.rsplit("_", 1)[-1]
        completed = _skip_completed_sources([t._source_id for t in sources])
        per_task: list[tuple[str, str, int]] = []
        survivors: list[Task] = []
        for t in sources:
            if t._source_id in completed:
                continue
            per_task.append((t.task_id, t._source_id, +1))
            survivors.append(t)
        _flush_deltas(per_task)
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
