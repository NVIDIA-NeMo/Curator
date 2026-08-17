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
COLLECTOR_ACTOR_ATTR = "_curator_stage_perf_collector_actor"
COLLECTOR_PENDING_ATTR = "_curator_stage_perf_pending_records"
COLLECTOR_REQUIRED_ATTR = "_curator_stage_perf_report_required"
MAX_PENDING_RECORDS = 64
MAX_COLLECTOR_ERRORS = 10


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
    report_required: bool


def _new_spool_path() -> str:
    spool_dir = Path(tempfile.mkdtemp(prefix="curator-stage-perf-"))
    return str(spool_dir / "records.jsonl")


@ray.remote(num_cpus=0)
class _StagePerfCollector:
    def __init__(self, spool_path: str) -> None:
        self._spool = _StagePerfSpool(spool_path)
        self._dropped_records = 0
        self._errors: list[str] = []

    def ready(self) -> bool:
        return True

    def record(self, perf_stats: StagePerfStats, _attached_to_output: bool) -> None:
        try:
            self._spool.record(perf_stats)
        except Exception as exc:  # noqa: BLE001
            self._dropped_records += 1
            if len(self._errors) < MAX_COLLECTOR_ERRORS:
                self._errors.append(f"{type(exc).__name__}: {exc}")

    def barrier(self) -> tuple[int, list[str]]:
        """Acknowledge all record calls ordered before this actor call."""
        return self._dropped_records, list(self._errors)

    def finish(self) -> tuple[str, int, int, list[str]]:
        path, record_count = self._spool.finish()
        return path, record_count, self._dropped_records, list(self._errors)


def performance_report_requested(stages: list[ProcessingStage]) -> bool:
    """Return whether a terminal consumer explicitly requires a complete report."""
    for stage in stages:
        request_records = getattr(stage, "requests_performance_records", None)
        if callable(request_records) and request_records():
            return True
        if bool(getattr(stage, "write_perf_stats", False)):
            return True
    return False


def performance_collection_enabled(stages: list[ProcessingStage]) -> bool:
    """Return whether a terminal consumer requests complete invocation records."""
    return performance_report_requested(stages)


def start_stage_perf_collector(stages: list[ProcessingStage]) -> Any | None:  # noqa: ANN401
    """Start one collector when a terminal consumer requests complete records."""
    if not performance_collection_enabled(stages):
        return None
    name = f"curator-stage-perf-{uuid.uuid4().hex}"
    spool_path = _new_spool_path()
    report_required = performance_report_requested(stages)
    collector = None
    try:
        driver_node_id = ray.get_runtime_context().get_node_id()
        collector = _StagePerfCollector.options(
            name=name,
            scheduling_strategy=NodeAffinitySchedulingStrategy(driver_node_id, soft=False),
        ).remote(spool_path)
        handle = _StagePerfCollectorHandle(actor=collector, spool_path=spool_path, report_required=report_required)
        ray.get(handle.actor.ready.remote())
    except Exception:
        if collector is not None:
            with contextlib.suppress(Exception):
                ray.kill(collector, no_restart=True)
        _cleanup_spool_path(spool_path)
        raise
    for stage in stages:
        setattr(stage, COLLECTOR_NAME_ATTR, name)
        setattr(stage, COLLECTOR_ACTOR_ATTR, collector)
        setattr(stage, COLLECTOR_PENDING_ATTR, [])
        setattr(stage, COLLECTOR_REQUIRED_ATTR, report_required)
    return handle


def record_stage_perf(
    stage: ProcessingStage,
    perf_stats: StagePerfStats,
    *,
    attached_to_output: bool,
) -> bool:
    """Publish asynchronously, acknowledging records in bounded batches."""
    collector = getattr(stage, COLLECTOR_ACTOR_ATTR, None)
    if collector is None:
        collector_name = str(getattr(stage, COLLECTOR_NAME_ATTR, "") or "")
        if not collector_name:
            return False
    pending_records = getattr(stage, COLLECTOR_PENDING_ATTR, None)
    if pending_records is None:
        pending_records = []
        setattr(stage, COLLECTOR_PENDING_ATTR, pending_records)
    try:
        if collector is None:
            collector = ray.get_actor(collector_name)
        pending_records.append(collector.record.remote(perf_stats, attached_to_output))
        if len(pending_records) >= MAX_PENDING_RECORDS:
            ray.get(pending_records)
            pending_records.clear()
    except Exception as exc:
        pending_records.clear()
        if bool(getattr(stage, COLLECTOR_REQUIRED_ATTR, False)):
            msg = f"Required stage performance publication failed for {stage.name}: {exc}"
            raise RuntimeError(msg) from exc
        logger.debug("Stage performance collector publish failed for {}: {}", stage.name, exc)
        return False
    return True


def flush_stage_perf_records(stage: ProcessingStage) -> None:
    """Wait for one producer's final asynchronous publication batch."""
    pending_records = getattr(stage, COLLECTOR_PENDING_ATTR, None)
    if not pending_records:
        return
    try:
        ray.get(pending_records)
    except Exception as exc:
        if bool(getattr(stage, COLLECTOR_REQUIRED_ATTR, False)):
            msg = f"Required stage performance publication flush failed for {stage.name}: {exc}"
            raise RuntimeError(msg) from exc
        logger.debug("Stage performance collector flush failed for {}: {}", stage.name, exc)
    finally:
        pending_records.clear()


def stop_stage_perf_collector(
    collector: _StagePerfCollectorHandle | None,
    stages: list[ProcessingStage],
    *,
    raise_on_failure: bool = False,
) -> PerformanceRecordStore:
    """Drain and remove the collector, clearing its run-scoped routing."""
    if collector is None:
        return PerformanceRecordStore()
    try:
        try:
            ray.get(collector.actor.barrier.remote())
            path, record_count, dropped_records, errors = ray.get(collector.actor.finish.remote())
        except Exception as exc:
            logger.debug("Stage performance collector finish failed: {}", exc)
            failed_store = PerformanceRecordStore(path=collector.spool_path)
            failed_store.cleanup()
            if raise_on_failure:
                msg = f"Required stage performance collector finish failed: {exc}"
                raise RuntimeError(msg) from exc
            return PerformanceRecordStore()
        if dropped_records:
            details = "; ".join(errors) or "unknown collector error"
            msg = f"Stage performance collector dropped {dropped_records} record(s): {details}"
            failed_store = PerformanceRecordStore(path=path, record_count=record_count)
            failed_store.cleanup()
            if raise_on_failure:
                raise RuntimeError(msg)
            logger.warning(msg)
            return PerformanceRecordStore()
        return PerformanceRecordStore(path=path, record_count=record_count)
    finally:
        for stage in stages:
            for attr_name in (
                COLLECTOR_NAME_ATTR,
                COLLECTOR_ACTOR_ATTR,
                COLLECTOR_PENDING_ATTR,
                COLLECTOR_REQUIRED_ATTR,
            ):
                with contextlib.suppress(AttributeError):
                    delattr(stage, attr_name)
        with contextlib.suppress(Exception):
            ray.kill(collector.actor, no_restart=True)
