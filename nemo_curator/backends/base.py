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

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nemo_curator.core.utils import ignore_ray_head_node
from nemo_curator.tasks import Task
from nemo_curator.utils.performance_utils import StageTimer

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


def _resolve_gpu_label(node_label: str, allocation: object, requires_gpu: bool) -> str:
    """Best-effort local GPU index label, in descending order of authority.

    The source of truth for "which GPU is this actor on" is backend-specific:

      1. **Backend allocation** (authoritative under Xenna). Xenna creates its
         Ray actors WITHOUT a Ray ``num_gpus`` request -- it runs its own
         allocator and sets ``CUDA_VISIBLE_DEVICES`` itself -- so
         ``ray.get_gpu_ids()`` is empty under Xenna. The ``WorkerMetadata``
         allocation it hands each actor is the same object it used to pin the
         GPU, so ``allocation.gpus[0].index`` is the correct per-actor index
         (and works for SPMD, where it leaves CUDA_VISIBLE_DEVICES untouched).
      2. **Ray-assigned ids** (``ray.get_gpu_ids()``): correct for Ray Data /
         Ray Actor Pool, which DO request ``num_gpus`` from Ray.
      3. **``CUDA_VISIBLE_DEVICES``** first token: last-resort, and only for
         stages that actually asked for a GPU -- guards against CPU actors that
         inherit the node's full device list.

    Returns ``"<node>:<idx>"`` (or bare ``"<idx>"`` when node is unknown), or
    ``""`` for CPU stages / when nothing resolves.
    """
    def _label(idx: object) -> str:
        idx_str = str(idx).strip()
        if not idx_str:
            return ""
        return f"{node_label}:{idx_str}" if node_label else idx_str

    gpus = getattr(allocation, "gpus", None)
    if gpus:
        idx = getattr(gpus[0], "index", None)
        if idx is not None:
            return _label(idx)

    try:
        import ray

        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            return _label(gpu_ids[0])
    except Exception:  # noqa: BLE001, S110 - gpu ids are best-effort
        pass

    if requires_gpu:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        first = cvd.split(",")[0].strip() if cvd else ""
        if first:
            return _label(first)

    return ""


def resolve_perf_identity(
    stage_name: str,
    allocation: object = None,
    requires_gpu: bool = False,
) -> tuple[str, str, str]:
    """Best-effort ``(actor_id, node_id, gpu_id)`` labels for a perf record.

    Resolved from the Ray runtime context plus the MPI rank env so the audio
    perf summary can attribute per-actor / per-GPU work without the framework
    plumbing real telemetry (that is the separate NVML/DCGM proposal). All
    lookups are wrapped: any backend without a Ray context (or a CPU-only
    stage) yields empty strings, which downstream code treats as "unknown".

    Args:
      * ``allocation``  -> the backend ``WorkerMetadata.allocation`` (Xenna
        populates it; Ray Data / Actor Pool leave it ``None``). Authoritative
        GPU source under Xenna -- see ``_resolve_gpu_label``.
      * ``requires_gpu`` -> whether the stage requested a GPU; gates the
        ``CUDA_VISIBLE_DEVICES`` fallback so CPU stages are never mislabeled.

    Labels:
      * ``node_id``   -> ``"node-<MPI rank>"`` when ``OMPI_COMM_WORLD_RANK`` is
        set (one node per rank in a multi-node MPI job), else the short Ray
        node-id hex. Empty if neither is available.
      * ``gpu_id``    -> ``"<node>:<local_gpu_idx>"`` (see ``_resolve_gpu_label``
        for the source precedence). Empty for CPU stages / no GPU.
      * ``actor_id``  -> ``"<stage_name>:actor-<short id>"`` from the Ray actor
        id (fallback worker id). Stage-name only when no id is resolvable.
    """
    node_label = ""
    actor_label = ""
    try:
        import ray

        ctx = ray.get_runtime_context()

        rank = os.environ.get("OMPI_COMM_WORLD_RANK")
        if rank is not None and rank != "":
            node_label = f"node-{rank}"
        else:
            try:
                node_hex = ctx.get_node_id()
                if node_hex:
                    node_label = f"node-{str(node_hex)[:8]}"
            except Exception:  # noqa: BLE001 - node id is best-effort
                node_label = ""

        short_id = ""
        try:
            short_id = (ctx.get_actor_id() or "") if hasattr(ctx, "get_actor_id") else ""
        except Exception:  # noqa: BLE001
            short_id = ""
        if not short_id:
            try:
                short_id = ctx.get_worker_id() or ""
            except Exception:  # noqa: BLE001
                short_id = ""
        actor_label = f"{stage_name}:actor-{str(short_id)[:8]}" if short_id else str(stage_name)
    except Exception:  # noqa: BLE001 - no Ray context (e.g. unit tests): leave blank
        return "", "", ""

    gpu_label = _resolve_gpu_label(node_label, allocation, requires_gpu)
    return actor_label, node_label, gpu_label


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

        # Log performance stats and add to result tasks
        _, stage_perf_stats = self._timer.log_stats()
        # Consume and attach any custom metrics recorded by the stage during this call
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)
        # Stamp best-effort actor/node/GPU identity so terminal stages can
        # attribute per-actor / per-GPU work. Resolved once per adapter instance
        # (identity is fixed for the life of the worker). Never fatal.
        if not hasattr(self, "_perf_identity") or self._perf_identity is None:
            allocation = getattr(getattr(self, "_worker_metadata", None), "allocation", None)
            requires_gpu = bool(getattr(getattr(self.stage, "resources", None), "requires_gpu", False))
            self._perf_identity = resolve_perf_identity(
                str(self.stage.name), allocation=allocation, requires_gpu=requires_gpu
            )
        stage_perf_stats.actor_id, stage_perf_stats.node_id, stage_perf_stats.gpu_id = self._perf_identity
        for task in results:
            task.add_stage_perf(stage_perf_stats)

        return results

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
        # Cache so process_batch can read the backend GPU allocation (Xenna
        # populates worker_metadata.allocation; Ray Data / Actor Pool leave it
        # None and we fall back to ray.get_gpu_ids()).
        self._worker_metadata = worker_metadata
        self.stage.setup(worker_metadata)

    def teardown(self) -> None:
        """Teardown the stage once per actor."""
        self.stage.teardown()
