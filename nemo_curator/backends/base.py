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
from nemo_curator.tasks import Task
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
            - With checkpoint: routes through ``stage.process_batch_grouped``
              so per-input lineage is preserved end-to-end. Propagation and
              completion accounting happen inside this method; the backend
              adapters no longer need to call
              ``_propagate_resumability_metadata`` /
              ``_record_checkpoint_events`` themselves.

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
        if self._checkpoint_actor is None:
            checkpoint_path = getattr(self.stage, "_checkpoint_path", None)
            if checkpoint_path is not None:
                from nemo_curator.utils.checkpoint import get_or_create_checkpoint_actor

                self._checkpoint_actor = get_or_create_checkpoint_actor(checkpoint_path)

        if self._checkpoint_actor is None:
            return self._run_flat(tasks)
        return self._run_grouped_and_account(tasks)

    def _run_flat(self, tasks: list[Task]) -> list[Task]:
        """Legacy non-checkpointed path: call the stage's flat process_batch."""
        if not hasattr(self, "_timer") or self._timer is None:
            self._timer = StageTimer(self.stage)
        input_size = sum(task.num_items for task in tasks)
        self._timer.reinit(input_size)

        with self._timer.time_process(input_size):
            results = self.stage.process_batch(tasks)

        _, stage_perf_stats = self._timer.log_stats()
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)
        for task in results:
            task.add_stage_perf(stage_perf_stats)

        return results

    def _run_grouped_and_account(self, tasks: list[Task]) -> list[Task]:
        """Checkpointed path: per-input groups, propagation, and per-parent recording."""
        # Drop inputs whose leaf is already recorded as completed.
        if self._is_last_user_stage and len(tasks) > 0:
            kept_tasks, kept_mask = self._drop_completed_inputs_with_mask(tasks)
        else:
            kept_tasks = tasks
            kept_mask = [True] * len(tasks)

        if not hasattr(self, "_timer") or self._timer is None:
            self._timer = StageTimer(self.stage)
        input_size = sum(task.num_items for task in kept_tasks)
        self._timer.reinit(input_size)

        # Snapshot resumability keys before the user stage runs so we can detect
        # the Obs 6 footgun: a stage that mutates ``task._metadata`` in place to
        # remove ``resumability_key`` silently breaks downstream completion
        # accounting. We warn loudly when this happens but cannot recover.
        keys_before = [t._metadata.get("resumability_key", "") for t in kept_tasks]

        with self._timer.time_process(input_size):
            groups_kept = self.stage.process_batch_grouped(kept_tasks) if kept_tasks else []

        self._warn_if_metadata_stripped(kept_tasks, keys_before)

        _, stage_perf_stats = self._timer.log_stats()
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)

        from nemo_curator.tasks import TransientDrop

        # Re-align groups with the ORIGINAL ``tasks`` list (pre-drop). Dropped
        # inputs map to an empty group; the recorder will treat that as a
        # no-op for them (they were already finalised in a prior run).
        groups_full: list[list[Task]] = []
        kept_iter = iter(groups_kept)
        for is_kept in kept_mask:
            if is_kept:
                grp = next(kept_iter)
                for t in grp:
                    # TransientDrop sentinels are not Tasks; skip perf stamping.
                    if not isinstance(t, TransientDrop):
                        t.add_stage_perf(stage_perf_stats)
                groups_full.append(grp)
            else:
                groups_full.append([])

        # Propagate metadata to outputs (skipped at the last user stage so the
        # recorder records the exact key the user stage emitted) and record
        # per-parent completion / fan-out events.
        if not self._is_last_user_stage:
            self._propagate_resumability_metadata(tasks, groups_full)
        self._record_checkpoint_events(tasks, groups_full, kept_mask)

        # Drop entire groups that contain any TransientDrop (the warning for
        # "mixed group" was logged by _record_checkpoint_events above). Strip
        # any stray TransientDrop sentinels from the remaining groups before
        # handing the flat list downstream.
        return [
            t
            for g in groups_full
            if not any(isinstance(o, TransientDrop) for o in g)
            for t in g
            if not isinstance(t, TransientDrop)
        ]

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

    def _propagate_resumability_metadata(
        self,
        input_tasks: list[Task],
        output_groups: list[list[Task]],
    ) -> None:
        """Propagate resumability_key and per-task resumability_task_key paths.

        Operates per parent: each input maps to its own group of outputs (one
        group per index). For each ``(parent, group)`` pair we copy the
        partition key from the parent to any output that lacks one and
        rewrite the per-task path as ``<parent_task_key>::<i>``. This is the
        Obs 2 fix — the prior implementation read only ``input_tasks[0]``
        and stamped every output with that parent's key, collapsing the
        parent-child relationship for any batched stage with ``batch_size > 1``.
        """
        from nemo_curator.tasks import TransientDrop

        for parent, group in zip(input_tasks, output_groups, strict=True):
            # Transient groups produce no downstream tasks; nothing to propagate to.
            if any(isinstance(t, TransientDrop) for t in group):
                continue
            source_key = parent._metadata.get("resumability_key", "")
            if not source_key:
                continue
            parent_task_key = parent._metadata.get("resumability_task_key", source_key)
            for i, task in enumerate(group):
                if "resumability_key" not in task._metadata:
                    task._metadata["resumability_key"] = source_key
                task._metadata["resumability_task_key"] = f"{parent_task_key}::{i}"

    def _drop_completed_inputs_with_mask(self, tasks: list[Task]) -> tuple[list[Task], list[bool]]:
        """Return ``(kept_tasks, kept_mask)`` — ``kept_mask[i]`` is True iff ``tasks[i]`` was kept.

        Only invoked on the user stage immediately preceding ``_CheckpointRecorderStage``
        (gated by ``self._is_last_user_stage``). One batched RPC to the checkpoint actor
        regardless of input batch size.
        """
        from nemo_curator.utils.checkpoint import _checkpoint_get

        queryable: list[tuple[int, tuple[str, str]]] = []
        for i, t in enumerate(tasks):
            key = t._metadata.get("resumability_key", "")
            task_key = t._metadata.get("resumability_task_key", "")
            if key and task_key:
                queryable.append((i, (key, task_key)))

        kept_mask = [True] * len(tasks)
        if not queryable:
            return tasks, kept_mask

        flags = _checkpoint_get(self._checkpoint_actor.are_leaves_completed.remote([p for _, p in queryable]))
        completed_indices = {idx for (idx, _), done in zip(queryable, flags, strict=True) if done}
        if not completed_indices:
            return tasks, kept_mask

        for idx in completed_indices:
            kept_mask[idx] = False
        logger.debug(f"Resumability: skipping {len(completed_indices)} already-completed leaves at last stage")
        return [t for i, t in enumerate(tasks) if i not in completed_indices], kept_mask

    def _record_checkpoint_events(
        self,
        input_tasks: list[Task],
        output_groups: list[list[Task]],
        kept_mask: list[bool] | None = None,
    ) -> None:
        """Per-parent fan-out / drop accounting (Obs 1 fix).

        For each ``(parent, group)`` pair where the parent was processed this
        run (``kept_mask[i]`` is True):
            - len(group) == 0 -> the stage dropped this parent; mark its
              ``resumability_task_key`` complete so resume doesn't replay it.
            - len(group) > 1  -> fan-out; bump ``expected`` for the parent's
              partition by ``len(group) - 1`` so the recorder can't satisfy
              the partition's completion check before the new leaves land.
            - len(group) == 1 -> 1:1 transformation; no checkpoint event
              needed at this stage (the recorder handles it downstream).

        ``kept_mask`` distinguishes "stage dropped this input" (record it
        complete) from "we skipped this input because it was already
        complete in a prior run" (do not re-record — already done).
        """
        if self._checkpoint_actor is None:
            return

        from nemo_curator.tasks import TransientDrop
        from nemo_curator.utils.checkpoint import _checkpoint_get

        if kept_mask is None:
            kept_mask = [True] * len(input_tasks)

        for parent, group, was_kept in zip(input_tasks, output_groups, kept_mask, strict=True):
            if not was_kept:
                continue
            key = parent._metadata.get("resumability_key", "")
            if not key:
                continue

            # Obs 3: a TransientDrop sentinel anywhere in the group means
            # this parent failed transiently this run; do NOT mark it
            # complete so resume retries it.
            if any(isinstance(o, TransientDrop) for o in group):
                self._log_transient_drop(parent, group)
                continue

            n_out = len(group)
            if n_out == 0:
                task_key = parent._metadata.get("resumability_task_key", "")
                if task_key:
                    _checkpoint_get(self._checkpoint_actor.mark_completed.remote(task_key, key))
            elif n_out > 1:
                # Commit increment synchronously so the recorder can't satisfy the check early.
                _checkpoint_get(self._checkpoint_actor.add_expected.remote(key, n_out - 1))

    def _warn_if_metadata_stripped(self, kept_tasks: list[Task], keys_before: list[str]) -> None:
        """Detect Obs 6: user stage stripped resumability_key from _metadata.

        Compares per-input resumability_key snapshots taken before the stage
        ran against the post-run state. If any input that had a key now lacks
        one, the stage mutated _metadata in place. Resumability cannot
        recover; we warn once per adapter so the user has a clear breadcrumb.
        """
        if getattr(self, "_warned_metadata_strip", False):
            return
        for task, key_before in zip(kept_tasks, keys_before, strict=True):
            if key_before and not task._metadata.get("resumability_key", ""):
                logger.warning(
                    f"Stage {self.stage.name!r} stripped 'resumability_key' from "
                    f"task {task.task_id!r}._metadata. Resumability/checkpoint "
                    f"recording is not supported for affected tasks; partitions "
                    f"involving them will not finalize. Stages must preserve "
                    f"task._metadata['resumability_key'] and 'resumability_task_key'."
                )
                self._warned_metadata_strip = True
                return

    def _log_transient_drop(self, parent: Task, group: list[Task]) -> None:
        """Log diagnostics for a parent whose group contained a TransientDrop sentinel."""
        from nemo_curator.tasks import TransientDrop

        transient = next(o for o in group if isinstance(o, TransientDrop))
        if any(not isinstance(o, TransientDrop) for o in group):
            logger.warning(
                f"Stage {self.stage.name!r} returned a group mixing TransientDrop "
                f"with real outputs for parent {parent.task_id!r}; treating as "
                f"transient (real outputs are discarded)."
            )
        logger.debug(
            f"Resumability: transient drop for {parent.task_id!r} (reason: {transient.reason!r}); resume will retry."
        )

    def teardown(self) -> None:
        """Teardown the stage once per actor."""
        self.stage.teardown()
