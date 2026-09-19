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
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ray.actor import ActorHandle

    from nemo_curator.stages.base import ProcessingStage

COLLECTOR_ACTOR_ATTR = "_curator_stage_perf_collector_actor"


class _StagePerfSpool:
    """Append invocation records to disk without retaining them in memory."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._record_count = 0
        self._file = Path(path).open("w", encoding="utf-8")  # noqa: SIM115
        self._failure_message: str | None = None
        self._finished = False

    def _terminal_error(self) -> RuntimeError:
        failure_message = self._failure_message or "unknown terminal failure"
        return RuntimeError(f"Stage performance collector is poisoned: {failure_message}")

    def _raise_if_unusable(self) -> None:
        if self._failure_message is not None:
            raise self._terminal_error()
        if self._finished:
            msg = "Stage performance collector is already finished"
            raise RuntimeError(msg)

    def _poison(self, failure_message: str) -> None:
        """Latch the first failure and prevent this spool from becoming a store."""
        if self._failure_message is None:
            self._failure_message = failure_message
        with contextlib.suppress(Exception):
            self._file.close()

    def _write_line(self, line: str) -> None:
        written = self._file.write(line)
        if written != len(line):
            msg = f"Short stage performance spool write: expected {len(line)} characters, wrote {written}"
            raise OSError(msg)

    def record(self, record: dict[str, Any]) -> None:
        """Encode and append one raw record, poisoning the spool on any failure."""
        self._raise_if_unusable()
        try:
            line = json.dumps(record) + "\n"
            self._write_line(line)
        except Exception as exc:
            self._poison(f"{type(exc).__name__}: {exc}")
            raise self._terminal_error() from exc
        self._record_count += 1

    def fail(self, failure_message: str) -> None:
        """Poison the spool explicitly after a transport-side uncertainty."""
        self._raise_if_unusable()
        self._poison(failure_message)
        raise self._terminal_error()

    def finish(self) -> tuple[str, int]:
        """Close a healthy spool; a poisoned spool can never become a store."""
        self._raise_if_unusable()
        try:
            self._file.close()
        except Exception as exc:
            self._poison(f"{type(exc).__name__}: {exc}")
            raise self._terminal_error() from exc
        self._finished = True
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

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield raw dictionaries so future schema fields remain readable."""
        yield from self.iter_dicts()

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
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    msg = f"Expected a stage performance record object, got {type(payload).__name__}"
                    raise TypeError(msg)
                yield payload

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
    actor: ActorHandle
    spool_path: str


def _new_spool_path() -> str:
    spool_dir = Path(tempfile.mkdtemp(prefix="curator-stage-perf-"))
    return str(spool_dir / "records.jsonl")


@ray.remote(num_cpus=0, max_concurrency=1)
class _StagePerfCollector:
    def __init__(self, spool_path: str) -> None:
        self._spool = _StagePerfSpool(spool_path)

    def ready(self) -> bool:
        return True

    def record(self, record: dict[str, Any]) -> None:
        self._spool.record(record)

    def fail(self, failure_message: str) -> None:
        self._spool.fail(failure_message)

    def finish(self) -> tuple[str, int]:
        return self._spool.finish()


def start_stage_perf_collector(stages: list[ProcessingStage]) -> _StagePerfCollectorHandle:
    """Start one collector after the executor has resolved that a report is required."""
    spool_path = _new_spool_path()
    collector = None
    try:
        driver_node_id = ray.get_runtime_context().get_node_id()
        collector = _StagePerfCollector.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
        ).remote(spool_path)
        handle = _StagePerfCollectorHandle(actor=collector, spool_path=spool_path)
        ray.get(handle.actor.ready.remote())
    except Exception:
        if collector is not None:
            with contextlib.suppress(Exception):
                ray.kill(collector, no_restart=True)
        _cleanup_spool_path(spool_path)
        raise
    for stage in stages:
        setattr(stage, COLLECTOR_ACTOR_ATTR, collector)
    return handle


def record_stage_perf(
    stage: ProcessingStage,
    record: dict[str, Any],
) -> None:
    """Publish one required invocation and wait for the collector acknowledgement."""
    collector = getattr(stage, COLLECTOR_ACTOR_ATTR, None)
    if collector is None:
        msg = f"Stage performance collector is not configured for {stage.name}"
        raise RuntimeError(msg)
    try:
        record_ref = collector.record.remote(record)
        # The driver cannot fence actor calls submitted by Ray Data/Xenna
        # workers. Each producer therefore waits for its own call before
        # returning, so executor completion fences every publication.
        ray.get(record_ref)
    except Exception as exc:
        # The append may have succeeded even when its acknowledgement was lost.
        # Poison the run best-effort so a later finish cannot expose a possibly
        # incomplete or ambiguously acknowledged store.
        with contextlib.suppress(Exception):
            fail_ref = collector.fail.remote(f"{type(exc).__name__}: {exc}")
            ray.get(fail_ref)
        msg = f"Required stage performance publication failed for {stage.name}: {exc}"
        raise RuntimeError(msg) from exc


def stop_stage_perf_collector(
    collector: _StagePerfCollectorHandle | None,
    stages: list[ProcessingStage],
) -> PerformanceRecordStore:
    """Drain and remove the collector, clearing its run-scoped routing."""
    if collector is None:
        return PerformanceRecordStore()
    try:
        try:
            path, record_count = ray.get(collector.actor.finish.remote())
        except Exception:
            _cleanup_spool_path(collector.spool_path)
            raise
        return PerformanceRecordStore(path=path, record_count=record_count)
    finally:
        for stage in stages:
            with contextlib.suppress(AttributeError):
                delattr(stage, COLLECTOR_ACTOR_ATTR)
        with contextlib.suppress(Exception):
            ray.kill(collector.actor, no_restart=True)
