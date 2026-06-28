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
from typing import TYPE_CHECKING, Any

import attrs
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from nemo_curator.stages.base import ProcessingStage


#: Identity fields on ``StagePerfStats``: best-effort string labels for the
#: actor/node/GPU that produced the record. They are metadata, not numeric
#: metrics, so they MUST be excluded from ``items()`` -- downstream collection
#: calls ``float()`` on every yielded value and would crash on a string.
_IDENTITY_FIELDS = (
    "invocation_id",
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
        invocation_id: Unique id for ONE ``process_batch`` call. The same record
            is attached to every output task of that call, so the audio summary
            dedups on it. Empty when unset -- consumers then fall back to a
            value-tuple fingerprint.
        actor_id: Best-effort label of the producing actor. Empty when unknown.
        node_id: Best-effort node label. Empty when unknown.
        gpu_id: Best-effort GPU label ``"<node>:<local_gpu_idx>"``. Empty for
            CPU stages / when unknown.
    """

    stage_name: str
    process_time: float = 0.0
    actor_idle_time: float = 0.0
    input_data_size_mb: float = 0.0
    num_items_processed: int = 0
    custom_metrics: dict[str, float] = attrs.field(factory=dict)
    # identity (metadata, never a numeric metric -- see _IDENTITY_FIELDS)
    invocation_id: str = ""
    actor_id: str = ""
    node_id: str = ""
    gpu_id: str = ""
    physical_address: str = ""
    pod_ip: str = ""
    hostname: str = ""
    gpu_indices: list[int] = attrs.field(factory=list)
    gpu_uuids: list[str] = attrs.field(factory=list)

    def __add__(self, other: StagePerfStats) -> StagePerfStats:
        """Add two StagePerfStats, summing scalars and custom metrics.

        Identity is per-worker, so it survives only when both operands share it;
        a cross-worker sum clears identity + invocation_id rather than mis-attribute.
        """
        same_worker = (
            self.actor_id == other.actor_id
            and self.node_id == other.node_id
            and self.physical_address == other.physical_address
        )
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
            # invocation_id identifies a single call -- a sum is not one call.
            invocation_id="",
            actor_id=self.actor_id if same_worker else "",
            node_id=self.node_id if same_worker else "",
            gpu_id=self.gpu_id if same_worker else "",
            physical_address=self.physical_address if same_worker else "",
            pod_ip=self.pod_ip if same_worker else "",
            hostname=self.hostname if same_worker else "",
            gpu_indices=list(self.gpu_indices) if same_worker else [],
            gpu_uuids=list(self.gpu_uuids) if same_worker else [],
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
        self.invocation_id = ""
        self.actor_id = ""
        self.node_id = ""
        self.gpu_id = ""
        self.physical_address = ""
        self.pod_ip = ""
        self.hostname = ""
        self.gpu_indices = []
        self.gpu_uuids = []

    def to_dict(self) -> dict[str, float | int]:
        """Convert to the stable main-branch public dictionary schema."""
        return {
            "stage_name": self.stage_name,
            "process_time": self.process_time,
            "actor_idle_time": self.actor_idle_time,
            "input_data_size_mb": self.input_data_size_mb,
            "num_items_processed": self.num_items_processed,
            "custom_metrics": dict(self.custom_metrics),
        }

    def to_extended_dict(self) -> dict[str, Any]:
        """Convert to the complete observability schema, including identity."""
        return attrs.asdict(self)

    def items(self) -> list[tuple[str, float | int]]:
        """Returns (metric_name, metric_value) pairs
        custom_metrics are flattened into the format (custom.<metric_name>, metric_value)
        """
        res = self.to_dict()
        res.pop("stage_name", None)
        # Identity fields are string metadata; downstream collectors call float()
        # on every yielded value, so they MUST be dropped here.
        for identity_field in _IDENTITY_FIELDS:
            res.pop(identity_field, None)
        custom_metrics = res.pop("custom_metrics", {})
        # Flatten custom_metrics with a stable prefix
        for key, value in custom_metrics.items():
            res[f"custom.{key}"] = value
        return list(res.items())


class StageTimer:
    """Tracks processing time and other metrics per process_data call."""

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
