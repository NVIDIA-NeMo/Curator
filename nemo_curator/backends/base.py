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

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.failed_task_markers import record_failed_tasks
from nemo_curator.backends.perf_identity import apply_worker_perf_identity, read_worker_metadata_identity
from nemo_curator.backends.slurm_array import (
    filter_slurm_array_source_tasks,
    resolve_slurm_array_config,
)
from nemo_curator.core.utils import ignore_ray_head_node
from nemo_curator.tasks import Task
from nemo_curator.tasks.sentinels import FailedTask, NoneTask
from nemo_curator.utils.performance_utils import StagePerfStats, StageTimer
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
    backend-specific allocation information. Backends may also stamp performance
    identity fields at worker setup.
    """

    worker_id: str = ""
    allocation: Any = None  # Backend-specific allocation info
    actor_id: str = ""
    node_id: str = ""
    gpu_id: str = ""
    physical_address: str = ""
    pod_ip: str = ""
    hostname: str = ""
    gpu_indices: list[int] = field(default_factory=list)
    gpu_uuids: list[str] = field(default_factory=list)


class BaseExecutor(ABC):
    """Executor for a pipeline."""

    def __init__(self, config: dict[str, Any] | None = None, ignore_head_node: bool = False):
        self.config = config or {}
        self.ignore_head_node = ignore_head_node or ignore_ray_head_node()

    @abstractmethod
    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> None:
        """Execute the pipeline."""

    def _start_pipeline_hardware_sampler(self) -> list[Any]:
        if not bool(self.config.get("pipeline_hardware_sampler_enabled", False)):
            return []
        try:
            from nemo_curator.utils.pipeline_hardware_sampler import start_pipeline_hardware_samplers

            interval_s = float(self.config.get("pipeline_hardware_sampler_interval_s", 0.5))
            startup_timeout_s = float(self.config.get("pipeline_hardware_sampler_startup_timeout_s", 5.0))
            return start_pipeline_hardware_samplers(interval_s=interval_s, startup_timeout_s=startup_timeout_s)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pipeline hardware sampler disabled: {}", exc)
            return []

    def _stop_pipeline_hardware_sampler(self, sampler_actors: list[Any]) -> StagePerfStats | None:
        if not sampler_actors:
            return None
        try:
            from nemo_curator.utils.pipeline_hardware_sampler import stop_pipeline_hardware_samplers

            stop_timeout_s = float(self.config.get("pipeline_hardware_sampler_stop_timeout_s", 10.0))
            metrics = stop_pipeline_hardware_samplers(sampler_actors, stop_timeout_s=stop_timeout_s)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pipeline hardware sampler stop failed: {}", exc)
            return None
        wall_time_s = float(metrics.pop("pipeline_hardware_wall_time_s", 0.0))
        return StagePerfStats(
            stage_name="pipeline_hardware_sampler",
            process_time=wall_time_s,
            num_items_processed=1,
            custom_metrics=metrics,
        )

    @staticmethod
    def _attach_pipeline_hardware_perf(tasks: list[Task], perf_stats: StagePerfStats | None) -> None:
        if perf_stats is None:
            return
        for task in tasks:
            task.add_stage_perf(perf_stats)

    def _publish_external_perf(self, stages: list["ProcessingStage"], perf_stats: StagePerfStats | None) -> bool:
        """Publish a run-level performance record to the last accepting stage."""
        if perf_stats is None:
            return False
        for stage in reversed(stages):
            recorder = getattr(stage, "record_external_stage_perf", None)
            if not callable(recorder):
                continue
            try:
                if bool(recorder(perf_stats)):
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("External perf publish failed for stage {}: {}", stage, exc)
        return False

    def _start_stage_perf_collector(self, stages: list["ProcessingStage"]) -> Any | None:  # noqa: ANN401
        """Start run-scoped transport for authoritative extended telemetry."""
        try:
            from nemo_curator.utils.stage_perf_collector import start_stage_perf_collector

            return start_stage_perf_collector(stages)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stage performance collector disabled: {}", exc)
            return None

    def _stop_stage_perf_collector(
        self,
        collector: Any | None,  # noqa: ANN401
        stages: list["ProcessingStage"],
    ) -> list[Any]:
        """Drain run-scoped extended telemetry and remove its collector."""
        try:
            from nemo_curator.utils.stage_perf_collector import stop_stage_perf_collector

            return stop_stage_perf_collector(collector, stages)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stage performance collector stop failed: {}", exc)
            return []

    def _publish_collected_stage_perf(self, stages: list["ProcessingStage"], records: list[Any]) -> bool:
        """Publish the full authoritative record set to the terminal consumer."""
        if not records:
            return False
        perf_stats = [record.perf_stats for record in records]
        for stage in reversed(stages):
            recorder = getattr(stage, "record_external_stage_perfs", None)
            if not callable(recorder):
                continue
            try:
                if bool(recorder(perf_stats)):
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("Collected stage perf publish failed for stage {}: {}", stage, exc)
        return False

    @staticmethod
    def _attach_unpublished_stage_perf(tasks: list[Task], records: list[Any]) -> None:
        """Preserve only records that did not already travel on a surviving task."""
        unattached = [record.perf_stats for record in records if not record.attached_to_output]
        for task in tasks:
            for perf_stats in unattached:
                task.add_stage_perf(perf_stats)


class BaseStageAdapter:
    """Adapts ProcessingStage to an execution backend, if needed."""

    def __init__(self, stage: "ProcessingStage"):
        self.stage = stage

    def _cache_perf_identity(self) -> None:
        """Copy backend-stamped identity from fixed worker metadata."""
        worker_metadata = getattr(self, "_worker_metadata", None)
        self._perf_identity = read_worker_metadata_identity(str(self.stage.name), worker_metadata)

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
        extended_metrics = bool(getattr(self.stage, "extended_performance_metrics", False))

        window_start = time.time() if extended_metrics else 0.0
        with self._timer.time_process(input_size):
            # Use the batch processing logic
            results = self.stage.process_batch(tasks)
        window_end = time.time() if extended_metrics else 0.0

        # A returned ``None`` ("filter this slot") becomes a NoneTask so every
        # output is a real Task that gets a task_id. Sentinels (NoneTask /
        # FailedTask) carry no identity and are stripped again before this
        # method returns.
        results = [NoneTask() if r is None else r for r in results]

        # Guarantee every emitted task has a task_id (derived id, or uuid fallback).
        results = self._post_process_task_ids(tasks, results)

        # Failed tasks on the source stage are not supported.
        is_source_stage = getattr(self.stage, "is_source_stage", False)
        failed_tasks = [r for r in results if isinstance(r, FailedTask)]
        if failed_tasks and is_source_stage:
            msg = f"Source stage {self.stage.name} emitted FailedTask, which is not supported."
            raise ValueError(msg)

        # Record failed tasks for later inspection or retry bookkeeping.
        if failed_tasks:
            record_failed_tasks()

        # Source-stage sentinels (NoneTask only; FailedTask already raised above) are
        # not real partitions and must not influence shard assignment or resumability
        # counters. Non-source stages keep sentinels here so _apply_resumability_counters
        # can fire the correct -1 delta for filtered (NoneTask) slots in the 1:1 path.
        if is_source_stage:
            results = [r for r in results if not _is_sentinel(r)]

        # Filter tasks based on the Slurm array configuration.
        slurm_array = resolve_slurm_array_config(is_source_stage=is_source_stage)
        if slurm_array is not None and is_source_stage:
            results = filter_slurm_array_source_tasks(results, slurm_array, self.stage.name)

        # Opt-in resumability: fire per-source deltas (no-op when no actor registered).
        if is_resumability_actor_active():
            results = self._apply_resumability_counters(tasks, results)

        # Sentinels never propagate to the next stage.
        results = [r for r in results if not _is_sentinel(r)]

        self._attach_stage_perf(results, window_start, window_end, extended_metrics=extended_metrics)
        return results

    def _attach_stage_perf(
        self,
        results: list[Task],
        window_start: float,
        window_end: float,
        *,
        extended_metrics: bool,
    ) -> None:
        """Attach one invocation record, with optional extended diagnostics."""
        _, stage_perf_stats = self._timer.log_stats()
        if extended_metrics:
            stage_perf_stats.invocation_id = uuid.uuid4().hex
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)
        if extended_metrics:
            self._add_gpu_sampler_metrics(stage_perf_stats, window_start, window_end)
            if not hasattr(self, "_perf_identity") or self._perf_identity is None:
                self._cache_perf_identity()
            apply_worker_perf_identity(stage_perf_stats, self._perf_identity)
            try:
                from nemo_curator.utils.stage_perf_collector import record_stage_perf

                record_stage_perf(
                    self.stage,
                    stage_perf_stats,
                    attached_to_output=bool(results),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Stage performance collector unavailable for {}: {}", self.stage.name, exc)
        for task in results:
            task.add_stage_perf(stage_perf_stats)

    def _add_gpu_sampler_metrics(
        self, stage_perf_stats: StagePerfStats, window_start: float, window_end: float
    ) -> None:
        """Add optional per-device diagnostics for one invocation window."""
        sampler = getattr(self, "_gpu_sampler", None)
        if sampler is not None:
            stage_perf_stats.custom_metrics.update(sampler.window_metrics(window_start, window_end))

    def _post_process_task_ids(self, input_tasks: list[Task], output_tasks: list[Task]) -> list[Task]:
        """Assign a deterministic ``task_id`` (parent id + own segment) to every
        emitted task. Runs once per stage on every backend, so ``process`` vs
        ``process_batch`` makes no difference; ids are re-derived at each stage
        boundary, so one object passing through N stages gets N ids.

        - single input → fan-out: each output is ``parent_<seg>``
        - ``len(output) == len(input)`` → positional 1:1: ``parent_i_<seg>``; a
          ``NoneTask`` slot means input ``i`` was filtered (kept for alignment, then
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
            # Fan-out (including a source reading from EmptyTask): every
            # output is a child of the single input.
            parent_id = input_tasks[0].task_id
            out = list(output_tasks)
            for i, task in enumerate(out):
                suffix = (task.get_deterministic_id() or i) if is_source else i
                task._set_task_id(parent_id, suffix)
            return out

        if len(output_tasks) == len(input_tasks):
            # Positional 1:1. A NoneTask sentinel remains aligned with the
            # parent whose output was filtered.
            out = []
            for parent, task in zip(input_tasks, output_tasks, strict=True):
                suffix = (task.get_deterministic_id() or 0) if is_source else 0
                task._set_task_id(parent.task_id, suffix)
                out.append(task)
            return out

        # Ambiguous cardinality across a batch: a derived id is not possible. Use a
        # random "r"-prefixed uuid so task_id is non-empty but clearly flagged
        # non-deterministic.
        out = list(output_tasks)
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
        self._worker_metadata = worker_metadata
        if bool(getattr(self.stage, "extended_performance_metrics", False)):
            self._cache_perf_identity()
        else:
            self._perf_identity = None
        self.stage.setup(worker_metadata)
        self._gpu_sampler = self._maybe_start_gpu_sampler()

    def _maybe_start_gpu_sampler(self) -> object | None:
        """Start a background NVML sampler for an assigned GPU stage."""
        if not bool(getattr(self.stage, "extended_performance_metrics", False)):
            return None
        resources = getattr(self.stage, "resources", None)
        if resources is None or not getattr(resources, "requires_gpu", False):
            return None
        try:
            from nemo_curator.utils.gpu_sampler import GpuUtilSampler

            gpu_uuids = tuple(getattr(self._perf_identity, "gpu_uuids", ()) or ())
            if not gpu_uuids:
                return None
            sampler = GpuUtilSampler(gpu_uuids=gpu_uuids, sample_all_visible=False)
            sampler.start()
        except Exception:  # noqa: BLE001
            return None
        return sampler

    def teardown(self) -> None:
        """Teardown the stage once per actor."""
        sampler = getattr(self, "_gpu_sampler", None)
        if sampler is not None:
            sampler.stop()
        self.stage.teardown()
