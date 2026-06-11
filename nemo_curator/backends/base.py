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

    def _cache_perf_identity(self) -> None:
        """Copy backend-stamped identity from ``WorkerMetadata`` (fixed per worker)."""
        worker_metadata = getattr(self, "_worker_metadata", None)
        self._perf_identity = read_worker_metadata_identity(str(self.stage.name), worker_metadata)

    def process_batch(self, tasks: list[Task]) -> list[Task]:
        """Process a batch of tasks, timing and stamping perf stats on outputs."""
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

    def _maybe_start_gpu_sampler(self) -> Any:
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
