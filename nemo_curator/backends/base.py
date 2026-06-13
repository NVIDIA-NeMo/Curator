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

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nemo_curator.backends.perf_identity import apply_worker_perf_identity, read_worker_metadata_identity
from nemo_curator.core.utils import ignore_ray_head_node
from nemo_curator.tasks import Task
from nemo_curator.utils.performance_utils import StageTimer

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


@dataclass
class NodeInfo:
    """Generic node information for setup_on_node calls across backends."""

    node_id: str = ""


@dataclass
class WorkerMetadata:
    """Generic worker metadata for setup_on_node calls across backends.

    Backends stamp ``actor_id``/``node_id``/``gpu_id`` at setup; perf records
    copy them verbatim (see ``backends/perf_identity.py``).
    """

    worker_id: str = ""
    allocation: Any = None  # Backend-specific allocation info (Xenna)
    actor_id: str = ""
    node_id: str = ""
    gpu_id: str = ""
    physical_address: str = ""
    pod_ip: str = ""
    hostname: str = ""
    gpu_indices: list[int] = field(default_factory=list)
    gpu_uuids: list[str] = field(default_factory=list)


@dataclass
class SchedulerReadyTaskBatch:
    """A backend row containing one scheduler-ready worker dispatch batch."""

    tasks: list[Task]
    total_cost: float = 0.0
    source_indices: list[int] = field(default_factory=list)


@dataclass
class ScheduledTaskBatchPlan:
    """Scheduler-ready backend rows plus the original parent tasks they reassemble into."""

    parent_tasks: list[Task]
    ready_batches: list[SchedulerReadyTaskBatch]

    @property
    def task_batches(self) -> list[list[Task]]:
        """Return raw task batches for compatibility with existing callers."""
        return [ready_batch.tasks for ready_batch in self.ready_batches]


class BaseExecutor(ABC):
    """Executor for a pipeline."""

    def __init__(self, config: dict[str, Any] | None = None, ignore_head_node: bool = False):
        self.config = config or {}
        self.ignore_head_node = ignore_head_node or ignore_ray_head_node()

    @abstractmethod
    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> None:
        """Execute the pipeline."""


def _enabled_batch_policy(stage: "ProcessingStage") -> object | None:
    policy = getattr(stage, "batch_policy", None)
    if policy is None or not getattr(policy, "enabled", False):
        return None
    return policy


def stage_uses_centralized_batching(stage: "ProcessingStage") -> bool:
    """Return whether a stage opts into the centralized executor scheduler.

    Centralized stages split once before worker dispatch, let the shared batch
    policy scheduler bucket those work units, then stitch
    worker results back after all planned batches complete. This is the full
    scheduler path used by GPU stages that can fan out a parent task into model
    work units, such as ASR audio chunks.
    """
    policy = _enabled_batch_policy(stage)
    if policy is None:
        return False
    build_tasks = getattr(stage, "build_prebucketed_tasks", None)
    assemble_results = getattr(stage, "assemble_prebucketed_task_results", None)
    return callable(build_tasks) and callable(assemble_results) and _scheduler_task_cost_fn(stage) is not None


def stage_uses_upstream_prebatching(stage: "ProcessingStage") -> bool:
    """Return whether a stage explicitly opts into executor-level prebatching.

    This is intentionally structural rather than audio-specific: any modality can
    opt in by carrying an enabled policy and exposing either a stage-owned planner
    or a per-task cost hook.
    """
    if stage_uses_centralized_batching(stage):
        return True

    policy = _enabled_batch_policy(stage)
    if policy is None:
        return False

    stage_planner = getattr(stage, "plan_upstream_batches", None)
    batch_task_cost = getattr(stage, "batch_task_cost", None)
    return callable(stage_planner) or callable(batch_task_cost)


def plan_upstream_task_batches(stage: "ProcessingStage", tasks: list[Task]) -> list[list[Task]]:
    """Build executor-visible task batches before backend workers run a stage.

    Centralized stages return scheduler-ready work-unit batches. Simple
    cost-aware stages fall back to stage-specific planners or
    ``BatchPolicy.bucketize`` over parent-task costs.
    """
    if not tasks:
        return []

    centralized_plan = build_scheduled_task_batch_plan(stage, tasks)
    if centralized_plan is not None:
        return centralized_plan.task_batches

    stage_planner = getattr(stage, "plan_upstream_batches", None)
    if callable(stage_planner):
        planned = stage_planner(tasks)
        if planned is not None:
            return [list(batch) for batch in planned if batch]

    policy = _enabled_batch_policy(stage)
    batch_task_cost = getattr(stage, "batch_task_cost", None)
    if policy is None or not callable(batch_task_cost):
        return [list(tasks)]

    return [
        list(sub_tasks)
        for _, sub_tasks, _total_cost in policy.bucketize_with_costs(tasks, cost_fn=batch_task_cost)
        if sub_tasks
    ]


def build_scheduled_task_batch_plan(
    stage: "ProcessingStage",
    tasks: list[Task],
) -> ScheduledTaskBatchPlan | None:
    """Build the one centralized work-unit and bucketed-dispatch plan, if any."""
    if not tasks or not stage_uses_centralized_batching(stage):
        return None

    build_tasks = getattr(stage, "build_prebucketed_tasks", None)
    scheduler_tasks = build_tasks(tasks)
    if scheduler_tasks is None:
        return None
    scheduler_tasks = list(scheduler_tasks)
    if not scheduler_tasks:
        return ScheduledTaskBatchPlan(parent_tasks=list(tasks), ready_batches=[])

    policy = _enabled_batch_policy(stage)
    cost_fn = _scheduler_task_cost_fn(stage)
    if policy is None or cost_fn is None:
        msg = f"{type(stage).__name__} is missing a batch policy or scheduler cost hook"
        raise RuntimeError(msg)

    ready_batches = [
        SchedulerReadyTaskBatch(
            tasks=list(sub_tasks),
            total_cost=total_cost,
            source_indices=list(source_indices),
        )
        for source_indices, sub_tasks, total_cost in policy.bucketize_with_costs(scheduler_tasks, cost_fn=cost_fn)
        if sub_tasks
    ]
    return ScheduledTaskBatchPlan(
        parent_tasks=list(tasks),
        ready_batches=ready_batches,
    )


def assemble_scheduled_task_batch_results(
    stage: "ProcessingStage",
    plan: ScheduledTaskBatchPlan,
    processed_tasks: list[Task],
) -> list[Task]:
    """Stitch scheduler-dispatched work units back to parent tasks."""
    assemble_results = getattr(stage, "assemble_prebucketed_task_results", None)
    if not callable(assemble_results):
        msg = f"{type(stage).__name__} does not implement assemble_prebucketed_task_results"
        raise TypeError(msg)
    return assemble_results(plan.parent_tasks, processed_tasks)


def _scheduler_task_cost_fn(stage: "ProcessingStage") -> Callable[[Task], float] | None:
    """Return the cost hook for scheduler-created work units, if available."""
    scheduler_task_cost = getattr(stage, "scheduler_task_cost", None)
    if callable(scheduler_task_cost):
        return scheduler_task_cost
    batch_task_cost = getattr(stage, "batch_task_cost", None)
    if callable(batch_task_cost):
        return batch_task_cost
    return None


def scheduler_ready_batch_tasks(ready_batch: SchedulerReadyTaskBatch | list[Task]) -> list[Task]:
    """Return dispatch tasks from a scheduler-ready backend row."""
    if isinstance(ready_batch, SchedulerReadyTaskBatch):
        return ready_batch.tasks
    return list(ready_batch)


def upstream_prebatching_batch_size(stage: "ProcessingStage", fallback_batch_size: int | None) -> int:
    """Suggest a scheduler input window large enough to fill policy buckets."""
    fallback = fallback_batch_size if fallback_batch_size is not None and fallback_batch_size > 0 else 1
    if not stage_uses_upstream_prebatching(stage):
        return fallback

    policy = _enabled_batch_policy(stage)
    configured_window = getattr(policy, "prebatching_window_size", None)
    if configured_window is not None:
        return int(configured_window)

    caps = getattr(policy, "max_items_per_batch_by_bucket", None)
    if not caps:
        return fallback
    try:
        policy_window = sum(int(cap) for cap in caps)
    except (TypeError, ValueError):
        return fallback
    return max(1, policy_window)


class BaseStageAdapter:
    """Adapts ProcessingStage to an execution backend, if needed."""

    def __init__(self, stage: "ProcessingStage"):
        self.stage = stage

    def _cache_perf_identity(self) -> None:
        """Copy backend-stamped identity from ``WorkerMetadata`` (fixed per worker)."""
        worker_metadata = getattr(self, "_worker_metadata", None)
        self._perf_identity = read_worker_metadata_identity(str(self.stage.name), worker_metadata)

    def process_batch(self, tasks: list[Task]) -> list[Task]:
        """Process a batch of tasks, timing and stamping perf stats on outputs."""
        centralized_plan = build_scheduled_task_batch_plan(self.stage, tasks)
        if centralized_plan is not None:
            return self._process_scheduled_task_batch_plan(centralized_plan)
        return self._process_batch_after_worker_planning(tasks)

    def _process_batch_after_worker_planning(self, tasks: list[Task]) -> list[Task]:
        """Process one worker-visible batch after upstream-style planning ran."""
        return self._process_batch_once(tasks)

    def _process_scheduled_task_batch_plan(self, plan: ScheduledTaskBatchPlan) -> list[Task]:
        """Run a centralized scheduler plan in-process and assemble the result."""
        processed_tasks: list[Task] = []
        for ready_batch in plan.ready_batches:
            processed_tasks.extend(self.process_scheduler_ready_batch(ready_batch))
        return assemble_scheduled_task_batch_results(self.stage, plan, processed_tasks)

    def process_scheduler_ready_batch(self, ready_batch: SchedulerReadyTaskBatch | list[Task]) -> list[Task]:
        """Process an already-planned scheduler batch without recursive planning."""
        return self._process_batch_once(scheduler_ready_batch_tasks(ready_batch))

    def _process_batch_once(self, tasks: list[Task]) -> list[Task]:
        """Run exactly one ``stage.process_batch`` invocation."""
        if not hasattr(self, "_timer") or self._timer is None:
            self._timer = StageTimer(self.stage)

        input_size = sum(task.num_items for task in tasks)
        self._timer.reinit(input_size)

        window_start = time.time()
        with self._timer.time_process(input_size):
            results = self.stage.process_batch(tasks)
        window_end = time.time()

        _, stage_perf_stats = self._timer.log_stats()
        # Unique id per invocation: the same record is attached to every output
        # task, so downstream accumulators dedup on it (N tasks count once).
        stage_perf_stats.invocation_id = uuid.uuid4().hex
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)
        # Fold in windowed GPU utilization (no-op for CPU / no NVML). Namespaced
        # per physical device UUID (``gpu_util_pct::<uuid>``) so the summary can
        # attribute it to a GPU index and roll the actor up from its devices.
        sampler = getattr(self, "_gpu_sampler", None)
        if sampler is not None:
            for uuid_key, metrics in sampler.window_stats(window_start, window_end).items():
                for metric, value in metrics.items():
                    stage_perf_stats.custom_metrics[f"{metric}::{uuid_key}"] = value
        # Identity is resolved once per worker in setup() and stamped on WorkerMetadata.
        if not hasattr(self, "_perf_identity") or self._perf_identity is None:
            self._cache_perf_identity()
        apply_worker_perf_identity(stage_perf_stats, self._perf_identity)
        for task in results:
            task.add_stage_perf(stage_perf_stats)

        return results

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup the stage on a node (node/worker info may be absent on some backends)."""
        self.stage.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup the stage once per actor."""
        self._worker_metadata = worker_metadata
        self._cache_perf_identity()
        self.stage.setup(worker_metadata)
        self._gpu_sampler = self._maybe_start_gpu_sampler()

    def _maybe_start_gpu_sampler(self) -> object | None:
        """Start a background NVML sampler for GPU stages (else ``None``)."""
        resources = getattr(self.stage, "resources", None)
        if resources is None or not getattr(resources, "requires_gpu", False):
            return None
        try:
            from nemo_curator.utils.gpu_sampler import GpuUtilSampler

            gpu_uuids = tuple(getattr(self._perf_identity, "gpu_uuids", ()) or ())
            sampler = GpuUtilSampler(gpu_uuids=gpu_uuids)
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
