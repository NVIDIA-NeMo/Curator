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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.core.utils import ignore_ray_head_node
from nemo_curator.tasks import Task, TransientDrop
from nemo_curator.utils.performance_utils import StageTimer

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


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
        self._checkpoint_actor = None
        self._timer: StageTimer | None = None
        # Stamped on the stage by each executor for the user stage immediately preceding
        # _CheckpointRecorderStage; gates the leaf-level resumability filter.  Setting on
        # the stage (not the adapter) lets the flag survive serialization to remote actors.
        self._is_last_user_stage = getattr(stage, "_is_last_user_stage", False)

    def process_batch(self, tasks: list[Task]) -> list[Task]:
        """Process a batch of tasks.

        Dispatches based on whether the pipeline is running with a
        ``checkpoint_path`` (stamped onto every user stage by
        ``Pipeline._with_checkpoint_stages``):
            - No checkpoint: flat path (legacy behaviour, identical to before).
            - With checkpoint: validates each input carries a
              ``resumability_key`` and drops any leaves already recorded as
              completed before invoking the stage. Per-parent propagation
              of ``resumability_key`` / ``resumability_task_key`` happens
              inside ``ProcessingStage.process_batch`` itself, so backend
              adapters no longer need to do it here.

        Returns the flat list of output tasks for downstream consumption.
        """
        # Normalize: backends like Ray Data hand us numpy arrays of Task objects;
        # later branches use truthiness checks that would otherwise raise
        # "truth value of an array with more than one element is ambiguous".
        if not isinstance(tasks, list):
            tasks = list(tasks)

        # Lazy-attach the checkpoint actor for adapters that never have
        # ``setup`` called (Ray Data task-style UDFs). The actor is a
        # singleton named Ray actor, so this is idempotent across workers.
        checkpoint_path = getattr(self.stage, "_checkpoint_path", None)
        if checkpoint_path is not None:
            if self._checkpoint_actor is None:
                from nemo_curator.utils.checkpoint import get_or_create_checkpoint_actor

                self._checkpoint_actor = get_or_create_checkpoint_actor(checkpoint_path)
            return self._process_batch_with_resume(tasks)
        else:
            return self._process_batch(tasks)

    def _process_batch(self, tasks: list[Task]) -> list[Task]:
        """Legacy non-checkpointed path: call the stage's flat process_batch."""
        if not hasattr(self, "_timer") or self._timer is None:
            self._timer = StageTimer(self.stage)

        # Calculate input data size for timer
        input_size = sum(task.num_items for task in tasks)
        # Initialize performance timer for this batch
        self._timer.reinit(input_size)

        with self._timer.time_process(input_size):
            # Use the batch processing logic
            results = self.stage.process_batch(tasks)

        # Log performance stats and add to result tasks
        _, stage_perf_stats = self._timer.log_stats()
        # Consume and attach any custom metrics recorded by the stage during this call
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)
        for task in results:
            task.add_stage_perf(stage_perf_stats)

        return results

    def _process_batch_with_resume(self, tasks: list[Task]) -> list[Task]:
        """Checkpointed path: per-input groups, propagation, and per-parent recording."""
        kept_tasks, already_completed = self._select_inputs_to_run(tasks)

        # Source stages receive a synthetic EmptyTask placeholder as input and emit
        # tasks stamped with a fresh resumability_key; validating their inputs would
        # always fail. Downstream stages must inherit resumability_key from upstream.
        if not self.stage.is_source_stage():
            for t in kept_tasks:
                if not t._metadata.get("resumability_key"):
                    msg = (
                        f"Task {t.task_id} is missing 'resumability_key' in _metadata at stage "
                        f"'{self.stage.name}'. Source stages must stamp a stable, unique "
                        "resumability_key per partition when running with checkpoint_path."
                    )
                    raise ValueError(msg)

        # Snapshot each parent's (resumability_key, resumability_task_key) BEFORE
        # process_batch runs. 1:1 stages typically return the same task object, so
        # propagation overwrites the input's resumability_task_key with the child's
        # extended path; reading from parent._metadata after the fact would yield
        # the child path and mis-account fan-out / drop events.
        parent_snapshot: list[tuple[Task, str, str]] = [
            (t, t._metadata.get("resumability_key", ""), t._metadata.get("resumability_task_key", ""))
            for t in kept_tasks
        ]

        results = self._process_batch(kept_tasks)
        # Fallback for stages that override process_batch but don't call
        # _propagate_resumability_metadata themselves: attribute outputs to parents
        # by Task._uuid match (typical for in-place-mutation pass-through stages).
        self._fallback_stamp_parent_task_key(parent_snapshot, results)
        self._record_checkpoint_events(parent_snapshot, results, already_completed)

        return results

    @staticmethod
    def _fallback_stamp_parent_task_key(
        parent_snapshot: list[tuple[Task, str, str]],
        output_tasks: list[Task],
    ) -> None:
        """Re-stamp ``parent_resumability_task_key`` on outputs from this stage via UUID match.

        Stages that override the flat ``process_batch`` may forget to call
        ``_propagate_resumability_metadata``. For the in-place-mutation pattern
        (the output task IS the input), the input and output share
        ``Task._uuid``, so we can recover the (parent -> child) attribution
        post-hoc. Even when an output already carries a
        ``parent_resumability_task_key`` (left by an upstream propagate), we
        OVERWRITE it: the field must reflect THIS stage's parent for
        ``_record_checkpoint_events`` to attribute fan-out / drop events
        correctly. We do NOT extend ``resumability_task_key`` — downstream
        stages that follow the contract will do that on their next call.
        """
        parents_by_uuid: dict[str, tuple[str, str]] = {
            p[0]._uuid: (p[1], p[2]) for p in parent_snapshot if p[1] and p[2]
        }
        for out in output_tasks:
            if isinstance(out, TransientDrop):
                continue
            record = parents_by_uuid.get(out._uuid)
            if record is None:
                continue
            key, parent_task_key = record
            out._metadata.setdefault("resumability_key", key)
            out._metadata["parent_resumability_task_key"] = parent_task_key

    def _select_inputs_to_run(self, tasks: list[Task]) -> tuple[list[Task], set[str]]:
        """Drop inputs whose leaf is already recorded as completed; pass-through otherwise.

        Returns ``(kept_tasks, already_completed)`` where ``already_completed`` is the
        set of ``task_id``s dropped this call (empty if none).
        """
        if self._is_last_user_stage and tasks:
            return self._drop_completed_inputs(tasks)
        else:
            return tasks, set()

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
        checkpoint_path = getattr(self.stage, "_checkpoint_path", None)
        if checkpoint_path:
            from nemo_curator.utils.checkpoint import get_or_create_checkpoint_actor

            self._checkpoint_actor = get_or_create_checkpoint_actor(checkpoint_path)

    def _drop_completed_inputs(self, tasks: list[Task]) -> tuple[list[Task], set[str]]:
        """Return ``(kept_tasks, already_completed)`` — ``already_completed`` holds the
        ``task_id``s of inputs whose leaves are already recorded as completed.

        Only invoked on the user stage immediately preceding ``_CheckpointRecorderStage``
        (gated by ``self._is_last_user_stage``). One batched RPC to the checkpoint actor
        regardless of input batch size.
        """
        from nemo_curator.utils.checkpoint import _checkpoint_get

        queryable: list[tuple[Task, tuple[str, str]]] = []
        for t in tasks:
            key = t._metadata.get("resumability_key", "")
            task_key = t._metadata.get("resumability_task_key", "")
            if key and task_key:
                queryable.append((t, (key, task_key)))

        if not queryable:
            return tasks, set()

        flags = _checkpoint_get(self._checkpoint_actor.are_leaves_completed.remote([p for _, p in queryable]))
        already_completed = {t.task_id for (t, _), done in zip(queryable, flags, strict=True) if done}
        if not already_completed:
            return tasks, set()

        logger.debug(f"Resumability: skipping {len(already_completed)} already-completed leaves at last stage")
        return [t for t in tasks if t.task_id not in already_completed], already_completed

    def _record_checkpoint_events(
        self,
        parent_snapshot: list[tuple[Task, str, str]],
        output_tasks: list[Task],
        already_completed: set[str],
    ) -> None:
        """Per-parent fan-out / drop accounting against the flat output list.

        Reconstructs the (parent -> children) grouping from
        ``parent_resumability_task_key`` (stamped by
        ``ProcessingStage._propagate_resumability_metadata``). For each input
        parent processed this run:
            - flagged transient (``_resumability_transient_drop``) -> no-op;
              resume will retry.
            - len(group) == 0 -> permanent drop (``process`` returned ``None``);
              mark the parent's ``resumability_task_key`` complete so resume
              does not replay it.
            - len(group) >  1 -> fan-out; bump ``expected`` for the partition by
              ``len(group) - 1`` synchronously so the recorder cannot satisfy
              the partition's completion check before the new leaves land.
            - len(group) == 1 -> 1:1; the recorder handles it downstream.

        ``parent_snapshot`` carries the parent's ``resumability_key`` and
        ``resumability_task_key`` captured before ``process_batch`` ran;
        relying on the live task metadata is unsafe because 1:1 stages
        return the same object and the propagation step rewrites its
        ``resumability_task_key`` to the child path.
        """
        if self._checkpoint_actor is None:
            return

        by_parent = self._count_outputs_by_parent(output_tasks)
        for parent, key, parent_task_key in parent_snapshot:
            self._record_parent_event(parent, key, parent_task_key, by_parent, already_completed)

    @staticmethod
    def _count_outputs_by_parent(output_tasks: list[Task]) -> dict[str, int]:
        from collections import defaultdict

        by_parent: dict[str, int] = defaultdict(int)
        for o in output_tasks:
            if isinstance(o, TransientDrop):
                continue
            ptk = o._metadata.get("parent_resumability_task_key", "")
            if ptk:
                by_parent[ptk] += 1
        return by_parent

    def _record_parent_event(
        self,
        parent: Task,
        key: str,
        parent_task_key: str,
        by_parent: dict[str, int],
        already_completed: set[str],
    ) -> None:
        if parent.task_id in already_completed:
            return
        if not key or not parent_task_key:
            return
        # Transient drop: parent will be retried on resume; do not record.
        if parent._metadata.pop("_resumability_transient_drop", False):
            return

        from nemo_curator.utils.checkpoint import _checkpoint_get

        n_out = by_parent.get(parent_task_key, 0)
        if n_out == 0:
            _checkpoint_get(self._checkpoint_actor.mark_completed.remote(parent_task_key, key))
        elif n_out > 1:
            _checkpoint_get(self._checkpoint_actor.add_expected.remote(key, n_out - 1))

    def teardown(self) -> None:
        """Teardown the stage once per actor."""
        self.stage.teardown()
