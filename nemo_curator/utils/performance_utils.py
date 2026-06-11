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

from __future__ import annotations

import contextlib
import statistics
import time
from typing import TYPE_CHECKING

import attrs
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from nemo_curator.stages.base import ProcessingStage


#: Identity fields on ``StagePerfStats``. These are best-effort string labels
#: identifying the actor/node/GPU that produced the record (populated by the
#: backend adapter from the Ray runtime context). They are metadata, NOT
#: numeric metrics, so they MUST be excluded from ``items()`` -- downstream
#: framework metric collection (``TaskPerfUtils.collect_stage_metrics``) calls
#: ``float()`` on every value ``items()`` yields and would crash on a string.
_IDENTITY_FIELDS = (
    "actor_id",
    "node_id",
    "gpu_id",
    "physical_address",
    "pod_ip",
    "hostname",
    "gpu_indices",
    "gpu_uuids",
)


@attrs.define
class StagePerfStats:
    """Statistics for tracking stage performance metrics.
    Attributes:
        stage_name: Name of the processing stage.
        process_time: Total processing time in seconds.
        actor_idle_time: Time the actor spent idle in seconds.
        input_data_size_mb: Size of input data in megabytes.
        num_items_processed: Number of items processed in this stage.
        custom_metrics: Custom metrics to track.
        actor_id: Best-effort label of the actor that produced this record
            (e.g. ``"QwenOmni_inference:actor-3f9c"``). Empty when unknown.
        node_id: Best-effort node label (e.g. ``"node-2"``). Empty when unknown.
        gpu_id: Best-effort GPU label ``"<node>:<local_gpu_idx>"``
            (e.g. ``"node-2:1"``). Empty for CPU stages / when unknown.
    """

    stage_name: str
    process_time: float = 0.0
    actor_idle_time: float = 0.0
    input_data_size_mb: float = 0.0
    num_items_processed: int = 0
    custom_metrics: dict[str, float] = attrs.field(factory=dict)
    # ----- identity (metadata, never a numeric metric -- see _IDENTITY_FIELDS) -----
    actor_id: str = ""
    node_id: str = ""
    gpu_id: str = ""
    physical_address: str = ""
    pod_ip: str = ""
    hostname: str = ""
    gpu_indices: list[int] = attrs.field(factory=list)
    gpu_uuids: list[str] = attrs.field(factory=list)

    def __add__(self, other: StagePerfStats) -> StagePerfStats:
        """Add two StagePerfStats (identity is metadata; keep self's labels)."""
        return StagePerfStats(
            stage_name=self.stage_name,
            process_time=self.process_time + other.process_time,
            actor_idle_time=self.actor_idle_time + other.actor_idle_time,
            input_data_size_mb=self.input_data_size_mb + other.input_data_size_mb,
            num_items_processed=self.num_items_processed + other.num_items_processed,
            custom_metrics={
                key: self.custom_metrics.get(key, 0.0) + other.custom_metrics.get(key, 0.0)
                for key in set(self.custom_metrics.keys()) | set(other.custom_metrics.keys())
            },
            actor_id=self.actor_id,
            node_id=self.node_id,
            gpu_id=self.gpu_id,
            physical_address=self.physical_address,
            pod_ip=self.pod_ip,
            hostname=self.hostname,
            gpu_indices=list(self.gpu_indices),
            gpu_uuids=list(self.gpu_uuids),
        )

    def __radd__(self, other: int | StagePerfStats) -> StagePerfStats:
        """Add two StagePerfStats together, if right is 0, returns itself."""
        if other == 0:
            return self
        if not isinstance(other, StagePerfStats):
            msg = f"Cannot add {type(other)} to {type(self)}"
            raise TypeError(msg)
        return self.__add__(other)

    def reset(self) -> None:
        """Reset the stats."""
        self.process_time = 0.0
        self.actor_idle_time = 0.0
        self.input_data_size_mb = 0.0
        self.num_items_processed = 0
        self.custom_metrics = {}
        self.actor_id = ""
        self.node_id = ""
        self.gpu_id = ""
        self.physical_address = ""
        self.pod_ip = ""
        self.hostname = ""
        self.gpu_indices = []
        self.gpu_uuids = []

    def to_dict(self) -> dict[str, float | int]:
        """Convert the stats to a dictionary."""
        return attrs.asdict(self)

    def items(self) -> list[tuple[str, float | int]]:
        """Returns (metric_name, metric_value) pairs
        custom_metrics are flattened into the format (custom.<metric_name>, metric_value)
        """
        res = self.to_dict()
        res.pop("stage_name", None)
        # Identity fields are string metadata, not numeric metrics. Downstream
        # collectors (TaskPerfUtils.collect_stage_metrics) call float() on every
        # value yielded here, so they MUST be dropped from the flattened output.
        for identity_field in _IDENTITY_FIELDS:
            res.pop(identity_field, None)
        # Extract and drop the raw custom_metrics dict from the flattened output
        custom_metrics = res.pop("custom_metrics", {})
        # Flatten custom_metrics with a stable prefix
        for key, value in custom_metrics.items():
            res[f"custom.{key}"] = value
        return list(res.items())


class StageTimer:
    """Tracker for stage performance stats.
    Tracks processing time and other metrics at a per process_data call level.
    """

    def __init__(self, stage: ProcessingStage) -> None:
        """Initialize the stage timer.
        Args:
            stage: The stage to track.
        """
        self._stage_name = str(stage.name)
        self._reset()
        self._last_active_time = time.time()
        self._initialized = False

    def _reset(self) -> None:
        """Reset internal counters."""
        self._num_items = 0
        self._durations_s: list[float] = []
        self._input_data_size_b = 0
        self._start = 0.0
        self._idle_time_s = 0.0
        self._startup_time_s = 0.0

    def reinit(self, stage_input_size: int = 1) -> None:
        """Reinitialize the stage timer.
        Args:
            stage: The stage to reinitialize the timer for.
            stage_input_size: The size of the stage input.
        """
        self._reset()
        self._input_data_size_b = stage_input_size
        self._start = time.time()
        if self._initialized:
            self._idle_time_s = self._start - self._last_active_time
        else:
            self._startup_time_s = self._start - self._last_active_time
        self._initialized = True

    @contextlib.contextmanager
    def time_process(self, num_items: int = 1) -> Generator[None, None, None]:
        """Time the processing of the stage.
        Args:
            num_items: The number of items being processed.
        """
        start_time = time.time()
        yield
        end_time = time.time()
        duration = end_time - start_time
        self._num_items += num_items
        for _ in range(num_items):
            self._durations_s.append(duration / num_items)

    def log_stats(self, *, verbose: bool = False) -> tuple[str, StagePerfStats]:
        """Log the stats of the stage.
        Args:
            verbose: Whether to log the stats verbosely.
        Returns:
            A tuple of the stage name and the stage performance stats.
        """
        end = time.time()
        process_data_dur_s = end - self._start
        num_items = self._num_items
        avg_dur_s = statistics.mean(self._durations_s) if self._durations_s else 0
        input_data_size_mb = self._input_data_size_b / 1024 / 1024
        start_time_s = self._startup_time_s
        idle_time_s = self._idle_time_s

        if verbose:
            logger.info(
                f"Stats: {process_data_dur_s=:.3f} - {num_items=} - {avg_dur_s=:.3f} - "
                f"{start_time_s=:.3f} - {idle_time_s=:.3f} - {input_data_size_mb=:.3f}"
            )
        self._last_active_time = time.time()

        stage_perf_stats = StagePerfStats(
            stage_name=self._stage_name,
            process_time=process_data_dur_s,
            actor_idle_time=idle_time_s,
            input_data_size_mb=input_data_size_mb,
            num_items_processed=num_items,
        )
        return self._stage_name, stage_perf_stats
