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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.perf_identity import apply_worker_perf_identity, read_worker_metadata_identity
from nemo_curator.core.utils import ignore_ray_head_node
from nemo_curator.tasks import Task
from nemo_curator.tasks.task_terminals import preserve_dropped_terminal_tasks
from nemo_curator.utils.performance_utils import StagePerfStats, StageTimer

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

    def _cleanup_stage_run_resources(self, stages: list["ProcessingStage"]) -> None:
        """Release run-scoped resources created by pipeline helper stages.

        Some helpers intentionally create named Ray actors so payload handles can
        cross backend-visible stage boundaries. Executors own the run lifecycle,
        so cleanup belongs here rather than in one row-processing stage.
        """
        for stage in reversed(stages):
            cleanup = getattr(stage, "cleanup_run_resources", None)
            if not callable(cleanup):
                continue
            try:
                cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Run-scoped cleanup failed for stage {stage}: {exc}")

    def _start_pipeline_hardware_sampler(self) -> list[Any]:
        # Observability is opt-in so existing pipelines keep main's actor count,
        # timings, and terminal performance-record shape.
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

    def _publish_external_perf(self, stages: list["ProcessingStage"], perf_stats: StagePerfStats | None) -> None:
        """Publish a run-level perf record to the terminal artifact writer when one exists."""
        if perf_stats is None:
            return
        for stage in reversed(stages):
            recorder = getattr(stage, "record_external_stage_perf", None)
            if not callable(recorder):
                continue
            try:
                recorder(perf_stats)
            except Exception as exc:  # noqa: BLE001
                logger.debug("External perf publish failed for stage {}: {}", stage, exc)
            return


class BaseStageAdapter:
    """Adapts ProcessingStage to an execution backend, if needed."""

    def __init__(self, stage: "ProcessingStage"):
        self.stage = stage

    @staticmethod
    def _stage_resource_expectation_metrics(stage: "ProcessingStage") -> dict[str, float]:
        """Return non-summing resource expectations attached by wrapper stages."""
        metrics: dict[str, float] = {}
        for attr_name, metric_name in (
            ("_curator_expected_stage_gpu_count", "expected_stage_gpu_count"),
            ("_curator_expected_stage_worker_count", "expected_stage_worker_count"),
            ("_curator_expected_worker_gpu_count", "expected_worker_gpu_count"),
        ):
            value = getattr(stage, attr_name, None)
            if isinstance(value, bool) or value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                metrics[metric_name] = numeric
        return metrics

    def _cache_perf_identity(self) -> None:
        """Copy backend-stamped identity from ``WorkerMetadata`` (fixed per worker)."""
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
        tracks_payload_refs = bool(getattr(self.stage, "_curator_tracks_payload_refs", False))
        input_payload_refs = self._collect_payload_refs(tasks) if tracks_payload_refs else {}
        extended_metrics = bool(getattr(self.stage, "extended_performance_metrics", False))

        window_start = time.time() if extended_metrics else 0.0
        try:
            with self._timer.time_process(input_size):
                # Use the batch processing logic
                results = self.stage.process_batch(tasks)
        except Exception:
            self._release_payload_refs(input_payload_refs.values())
            raise
        window_end = time.time() if extended_metrics else 0.0
        if bool(getattr(self.stage, "_curator_preserves_terminal_tasks", False)):
            results = preserve_dropped_terminal_tasks(self.stage, tasks, results)
        if input_payload_refs:
            self._release_dropped_payload_refs(input_payload_refs, results)

        # Guarantee every emitted task has a task_id (derived id, or uuid fallback).
        results = self._post_process_task_ids(tasks, results)

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
        # Unique id per invocation: the same record is attached to every output
        # task, so downstream accumulators dedup on it (N tasks count once).
        if extended_metrics:
            stage_perf_stats.invocation_id = uuid.uuid4().hex
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)
        if extended_metrics:
            stage_perf_stats.custom_metrics.update(self._stage_resource_expectation_metrics(self.stage))
        # Fold in windowed GPU utilization (no-op for CPU / no NVML). Namespaced
        # per physical device UUID (``gpu_util_pct::<uuid>``) so the summary can
        # attribute it to a GPU index and roll the actor up from its devices.
        if extended_metrics:
            self._add_gpu_sampler_metrics(stage_perf_stats, window_start, window_end)
        # Identity is resolved once per worker in setup() and stamped on WorkerMetadata.
        if extended_metrics:
            if not hasattr(self, "_perf_identity") or self._perf_identity is None:
                self._cache_perf_identity()
            apply_worker_perf_identity(stage_perf_stats, self._perf_identity)
        for task in results:
            task.add_stage_perf(stage_perf_stats)

    def _add_gpu_sampler_metrics(
        self, stage_perf_stats: StagePerfStats, window_start: float, window_end: float
    ) -> None:
        """Add optional per-device diagnostics for one invocation window."""
        sampler = getattr(self, "_gpu_sampler", None)
        if sampler is not None:
            diagnostics = getattr(sampler, "diagnostics", None)
            if callable(diagnostics):
                stage_perf_stats.custom_metrics.update(diagnostics())
            for uuid_key, metrics in sampler.window_stats(window_start, window_end).items():
                for metric, value in metrics.items():
                    stage_perf_stats.custom_metrics[f"{metric}::{uuid_key}"] = value

    def _collect_payload_refs(self, tasks: list[Task]) -> dict[str, object]:
        refs: dict[str, object] = {}
        if not bool(getattr(self.stage, "_curator_tracks_payload_refs", False)):
            return refs
        try:
            from nemo_curator.pipeline.payload_refs import task_payload_refs
        except ImportError:
            return refs
        for task in tasks:
            for payload_ref in task_payload_refs(task):
                payload_id = getattr(payload_ref, "payload_id", None)
                if payload_id:
                    refs[str(payload_id)] = payload_ref
        return refs

    def _release_dropped_payload_refs(self, input_refs: dict[str, object], output_tasks: list[Task]) -> None:
        if not input_refs:
            return
        try:
            from nemo_curator.pipeline.payload_refs import task_payload_refs
        except ImportError:
            return
        output_ids: set[str] = set()
        for task in output_tasks:
            if task is None:
                continue
            for payload_ref in task_payload_refs(task):
                payload_id = getattr(payload_ref, "payload_id", None)
                if payload_id:
                    output_ids.add(str(payload_id))
        dropped = [payload_ref for payload_id, payload_ref in input_refs.items() if payload_id not in output_ids]
        self._release_payload_refs(dropped)

    @staticmethod
    def _release_payload_refs(payload_refs: object) -> None:
        if not payload_refs:
            return
        try:
            from nemo_curator.pipeline.payload_refs import PayloadRef, release_payload_ref
        except ImportError:
            return
        for payload_ref in payload_refs:
            if isinstance(payload_ref, PayloadRef):
                release_payload_ref(payload_ref)

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
        assumption may misattribute parents. That combination is unsupported
        until per-slot sentinels (NoneTask/FailedTask) land in a later PR.
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
        """Start a background NVML sampler for GPU stages (else ``None``)."""
        if not bool(getattr(self.stage, "extended_performance_metrics", False)):
            return None
        resources = getattr(self.stage, "resources", None)
        if resources is None or not getattr(resources, "requires_gpu", False):
            return None
        try:
            from nemo_curator.utils.gpu_sampler import GpuUtilSampler

            gpu_uuids = tuple(getattr(self._perf_identity, "gpu_uuids", ()) or ())
            sampler = GpuUtilSampler(gpu_uuids=gpu_uuids, sample_all_visible=True)
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
