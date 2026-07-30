# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run-scoped transport for stage invocations, including zero-output calls."""

from __future__ import annotations

import contextlib
import json
import tempfile
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import ray
from loguru import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.utils.performance_utils import StagePerfStats

COLLECTOR_NAME_ATTR = "_curator_stage_perf_collector_name"


class _StagePerfSpool:
    """Append invocation records to disk without retaining them in memory."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._record_count = 0
        self._file = Path(path).open("w", encoding="utf-8")  # noqa: SIM115

    def record(self, perf_stats: StagePerfStats) -> None:
        json.dump(perf_stats.to_extended_dict(), self._file, sort_keys=True)
        self._file.write("\n")
        self._record_count += 1

    def finish(self) -> tuple[str, int]:
        self._file.close()
        return self._path, self._record_count


def _cleanup_spool_path(path: str) -> None:
    """Remove one owned spool file and its now-empty temporary directory."""
    if not path:
        return
    spool_path = Path(path)
    with contextlib.suppress(OSError):
        spool_path.unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        spool_path.parent.rmdir()


@dataclass
class PerformanceRecordStore:
    """Re-iterable, disk-backed stage-invocation records.

    The store owns its temporary spool until ``close()``/``cleanup()`` is
    called or the last store reference is released. It can also be used as a
    context manager for deterministic cleanup.
    """

    path: str = ""
    record_count: int = 0
    _finalizer: weakref.finalize | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.path:
            self._finalizer = weakref.finalize(
                self,
                _cleanup_spool_path,
                self.path,
            )

    @classmethod
    def from_records(cls, records: Iterable[StagePerfStats]) -> PerformanceRecordStore:
        """Spool an existing iterable without creating another in-memory copy."""
        spool = _StagePerfSpool(_new_spool_path())
        for record in records:
            spool.record(record)
        path, record_count = spool.finish()
        store = cls(path=path, record_count=record_count)
        if record_count == 0:
            store.cleanup()
        return store

    def __iter__(self) -> Iterator[StagePerfStats]:
        from nemo_curator.utils.performance_utils import StagePerfStats

        for payload in self.iter_dicts():
            yield StagePerfStats(**payload)

    def __len__(self) -> int:
        return self.record_count

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def iter_dicts(self) -> Iterator[dict[str, Any]]:
        """Yield complete invocation dictionaries one at a time."""
        if not self.path:
            return
        with Path(self.path).open(encoding="utf-8") as records_file:
            for line in records_file:
                yield json.loads(line)

    def cleanup(self) -> None:
        """Remove the run-scoped spool after its records are no longer needed."""
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer()
        else:
            _cleanup_spool_path(self.path)
        self.path = ""
        self.record_count = 0

    def close(self) -> None:
        """Release the owned spool explicitly."""
        self.cleanup()


@dataclass(frozen=True)
class _StagePerfCollectorHandle:
    actor: Any
    spool_path: str


def _new_spool_path() -> str:
    spool_dir = Path(tempfile.mkdtemp(prefix="curator-stage-perf-"))
    return str(spool_dir / "records.jsonl")


@ray.remote(num_cpus=0)
class _StagePerfCollector:
    def __init__(self, spool_path: str) -> None:
        self._spool = _StagePerfSpool(spool_path)

    def ready(self) -> bool:
        return True

    def record(self, perf_stats: StagePerfStats, _attached_to_output: bool) -> None:
        self._spool.record(perf_stats)

    def finish(self) -> tuple[str, int]:
        return self._spool.finish()


def performance_collection_enabled(stages: list[ProcessingStage]) -> bool:
    """Return whether any stage or terminal consumer requests full collection."""
    return any(
        bool(getattr(stage, "extended_performance_metrics", False)) or bool(getattr(stage, "write_perf_stats", False))
        for stage in stages
    )


def start_stage_perf_collector(stages: list[ProcessingStage]) -> Any | None:  # noqa: ANN401
    """Start one collector when extended metrics or a terminal consumer is enabled."""
    if not performance_collection_enabled(stages):
        return None
    name = f"curator-stage-perf-{uuid.uuid4().hex}"
    spool_path = _new_spool_path()
    driver_node_id = ray.get_runtime_context().get_node_id()
    collector = _StagePerfCollector.options(
        name=name,
        scheduling_strategy=NodeAffinitySchedulingStrategy(driver_node_id, soft=False),
    ).remote(spool_path)
    handle = _StagePerfCollectorHandle(actor=collector, spool_path=spool_path)
    ray.get(handle.actor.ready.remote())
    for stage in stages:
        setattr(stage, COLLECTOR_NAME_ATTR, name)
    return handle


def record_stage_perf(
    stage: ProcessingStage,
    perf_stats: StagePerfStats,
    *,
    attached_to_output: bool,
) -> bool:
    """Synchronously publish one record so the driver cannot drain too early."""
    collector_name = str(getattr(stage, COLLECTOR_NAME_ATTR, "") or "")
    if not collector_name:
        return False
    try:
        collector = ray.get_actor(collector_name)
        ray.get(collector.record.remote(perf_stats, attached_to_output))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Stage performance collector publish failed for {}: {}", stage.name, exc)
        return False
    return True


def stop_stage_perf_collector(
    collector: _StagePerfCollectorHandle | None,
    stages: list[ProcessingStage],
) -> PerformanceRecordStore:
    """Drain and remove the collector, clearing its run-scoped routing."""
    if collector is None:
        return PerformanceRecordStore()
    try:
        path, record_count = ray.get(collector.actor.finish.remote())
        return PerformanceRecordStore(path=path, record_count=record_count)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Stage performance collector finish failed: {}", exc)
        failed_store = PerformanceRecordStore(path=collector.spool_path)
        failed_store.cleanup()
        return PerformanceRecordStore()
    finally:
        for stage in stages:
            with contextlib.suppress(AttributeError):
                delattr(stage, COLLECTOR_NAME_ATTR)
        with contextlib.suppress(Exception):
            ray.kill(collector.actor, no_restart=True)
