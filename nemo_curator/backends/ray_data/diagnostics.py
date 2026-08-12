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

"""Runtime Ray Data scheduler diagnostics for the supported Ray release.

The diagnostics were developed as a Ray source patch, but Curator cannot
distribute a patched Ray wheel.  This module installs the equivalent Python
hooks in the driver process.  The hooks emit through child loggers of
``ray.data`` so Ray's ``SessionFileHandler`` writes them to
``session_latest/logs/ray-data/ray-data.log``.

All affected scheduler components run in the Ray Data driver.  Worker
environments therefore do not need modified Ray installations.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# Runtime monkeypatch callbacks necessarily accept objects owned by Ray's
# private, untyped implementation modules.
# ruff: noqa: ANN401

_SUPPORTED_RAY_VERSION = "2.57.0"
_INSTALL_MARKER = "_nemo_curator_ray_data_diagnostics_installed"
_INSTALL_LOCK = threading.Lock()
RAY_DATA_DIAGNOSTICS_ENV_VAR = "NEMO_CURATOR_RAY_DATA_DIAGNOSTICS"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_METADATA_FETCH_STALL_THRESHOLD_S = 1.0
_MAX_METADATA_FETCH_STALL_PAIRS_PER_OPERATOR = 10


class DiagnosticsInstallStatus(StrEnum):
    """Result of attempting to enable Ray Data diagnostics."""

    DISABLED = "disabled"
    INSTALLED = "installed"
    ALREADY_INSTALLED = "already_installed"
    NATIVE = "native"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class _TaskAdmissionDecision:
    allowed: bool
    reason: str
    incremental_resources: Any
    remaining_budget: Any
    pending_output_estimate: float | None
    op_usage: Any = None
    allocation: Any = None


@dataclass
class _MetadataRefTiming:
    operator: str
    submitted_at: float
    fetched_at: float | None = None


@dataclass
class _MetadataOperatorStats:
    mode: str
    submitted: int = 0
    emitted: int = 0
    failed: int = 0
    retry_count: int = 0
    pending: int = 0
    pending_high_watermark: int = 0
    fetch_latency_s_total: float = 0.0
    fetch_latency_s_max: float = 0.0
    delivery_delay_s_total: float = 0.0
    delivery_delay_s_max: float = 0.0
    stall_count: int = 0
    stall_duration_s_total: float = 0.0
    stall_duration_s_max: float = 0.0
    stall_started_at: float | None = None
    stall_was_logged: bool = False
    logged_stall_pairs: int = 0
    suppressed_transition_events: int = 0


class _MetadataFetchDiagnostics:
    def __init__(self, logger: logging.Logger, mode: str, clock: Callable[[], float]) -> None:
        self._logger = logger
        self._mode = mode
        self._clock = clock
        self._refs: dict[object, _MetadataRefTiming] = {}
        self._stats: dict[str, _MetadataOperatorStats] = {}
        self._summaries_logged = False
        self._lock = threading.Lock()

    def _operator_stats(self, operator: str) -> _MetadataOperatorStats:
        return self._stats.setdefault(operator, _MetadataOperatorStats(mode=self._mode))

    def submit(self, deferred: list[Any]) -> None:
        now = self._clock()
        with self._lock:
            for item in deferred:
                operator = item.task.operator_name
                if item.meta_ref in self._refs:
                    continue
                self._refs[item.meta_ref] = _MetadataRefTiming(operator=operator, submitted_at=now)
                stats = self._operator_stats(operator)
                stats.submitted += 1
                stats.pending += 1
                stats.pending_high_watermark = max(stats.pending_high_watermark, stats.pending)

    def record_fetch_pass(self, pending: list[object], remaining: list[object]) -> None:
        now = self._clock()
        remaining_refs = set(remaining)
        with self._lock:
            operators = set()
            for ref in pending:
                timing = self._refs.get(ref)
                if timing is None:
                    continue
                operators.add(timing.operator)
                stats = self._operator_stats(timing.operator)
                if ref in remaining_refs:
                    stats.retry_count += 1
                elif timing.fetched_at is None:
                    timing.fetched_at = now
                    latency = now - timing.submitted_at
                    stats.fetch_latency_s_total += latency
                    stats.fetch_latency_s_max = max(stats.fetch_latency_s_max, latency)
            for operator in operators:
                stats = self._operator_stats(operator)
                if stats.stall_started_at is not None and not self._has_stalled_unfetched_ref(operator, now):
                    self._recover_from_stall(operator, stats, now)

    def record_not_ready(self, ref: object) -> None:
        now = self._clock()
        with self._lock:
            timing = self._refs.get(ref)
            if timing is None:
                return
            stats = self._operator_stats(timing.operator)
            pending_duration = now - timing.submitted_at
            if pending_duration < _METADATA_FETCH_STALL_THRESHOLD_S or stats.stall_started_at is not None:
                return
            stats.stall_started_at = timing.submitted_at
            stats.stall_count += 1
            if stats.logged_stall_pairs < _MAX_METADATA_FETCH_STALL_PAIRS_PER_OPERATOR:
                stats.stall_was_logged = True
                self._logger.debug(
                    format_logfmt_event(
                        "ray_data_metadata_fetch_state",
                        {
                            "operator": timing.operator,
                            "mode": self._mode,
                            "state": "stalled",
                            "pending_refs": stats.pending,
                            "oldest_pending_ms": _milliseconds(pending_duration),
                        },
                    )
                )
            else:
                stats.stall_was_logged = False
                stats.suppressed_transition_events += 1

    def record_result(self, ref: object, *, failed: bool) -> None:
        now = self._clock()
        with self._lock:
            timing = self._refs.pop(ref, None)
            if timing is None:
                return
            stats = self._operator_stats(timing.operator)
            if timing.fetched_at is None:
                timing.fetched_at = now
                latency = now - timing.submitted_at
                stats.fetch_latency_s_total += latency
                stats.fetch_latency_s_max = max(stats.fetch_latency_s_max, latency)
            delivery_delay = now - timing.fetched_at
            stats.delivery_delay_s_total += delivery_delay
            stats.delivery_delay_s_max = max(stats.delivery_delay_s_max, delivery_delay)
            stats.emitted += 1
            stats.failed += int(failed)
            stats.pending -= 1
            if stats.stall_started_at is not None and not self._has_stalled_unfetched_ref(timing.operator, now):
                self._recover_from_stall(timing.operator, stats, now)

    def record_inline_result(self, ref: object, operator: str, *, emitted: bool, failed: bool) -> None:
        not_ready = False
        with self._lock:
            if ref not in self._refs:
                now = self._clock()
                self._refs[ref] = _MetadataRefTiming(operator=operator, submitted_at=now)
                stats = self._operator_stats(operator)
                stats.submitted += 1
                stats.pending += 1
                stats.pending_high_watermark = max(stats.pending_high_watermark, stats.pending)
            if not emitted:
                self._operator_stats(operator).retry_count += 1
                not_ready = True
        if not_ready:
            self.record_not_ready(ref)
            return
        self.record_result(ref, failed=failed)

    def record_inline_start(self, ref: object, operator: str) -> None:
        with self._lock:
            if ref in self._refs:
                return
            self._refs[ref] = _MetadataRefTiming(operator=operator, submitted_at=self._clock())
            stats = self._operator_stats(operator)
            stats.submitted += 1
            stats.pending += 1
            stats.pending_high_watermark = max(stats.pending_high_watermark, stats.pending)

    def _has_stalled_unfetched_ref(self, operator: str, now: float) -> bool:
        return any(
            timing.operator == operator
            and timing.fetched_at is None
            and now - timing.submitted_at >= _METADATA_FETCH_STALL_THRESHOLD_S
            for timing in self._refs.values()
        )

    def _recover_from_stall(self, operator: str, stats: _MetadataOperatorStats, now: float) -> None:
        if stats.stall_started_at is None:
            return
        duration = now - stats.stall_started_at
        stats.stall_duration_s_total += duration
        stats.stall_duration_s_max = max(stats.stall_duration_s_max, duration)
        if stats.stall_was_logged:
            self._logger.debug(
                format_logfmt_event(
                    "ray_data_metadata_fetch_state",
                    {
                        "operator": operator,
                        "mode": self._mode,
                        "state": "recovered",
                        "pending_refs": stats.pending,
                        "stall_duration_ms": _milliseconds(duration),
                    },
                )
            )
            stats.logged_stall_pairs += 1
        else:
            stats.suppressed_transition_events += 1
        stats.stall_started_at = None
        stats.stall_was_logged = False

    def log_summaries(self) -> None:
        now = self._clock()
        with self._lock:
            if self._summaries_logged:
                return
            self._summaries_logged = True
            for operator, stats in self._stats.items():
                if stats.stall_started_at is not None:
                    duration = now - stats.stall_started_at
                    stats.stall_duration_s_total += duration
                    stats.stall_duration_s_max = max(stats.stall_duration_s_max, duration)
                self._logger.debug(
                    format_logfmt_event(
                        "ray_data_metadata_fetch_summary",
                        {
                            "operator": operator,
                            "mode": stats.mode,
                            "submitted": stats.submitted,
                            "emitted": stats.emitted,
                            "failed": stats.failed,
                            "retry_count": stats.retry_count,
                            "pending": stats.pending,
                            "pending_high_watermark": stats.pending_high_watermark,
                            "fetch_latency_ms_total": _milliseconds(stats.fetch_latency_s_total),
                            "fetch_latency_ms_max": _milliseconds(stats.fetch_latency_s_max),
                            "delivery_delay_ms_total": _milliseconds(stats.delivery_delay_s_total),
                            "delivery_delay_ms_max": _milliseconds(stats.delivery_delay_s_max),
                            "stall_count": stats.stall_count,
                            "stall_duration_ms_total": _milliseconds(stats.stall_duration_s_total),
                            "stall_duration_ms_max": _milliseconds(stats.stall_duration_s_max),
                            "suppressed_transition_events": stats.suppressed_transition_events,
                        },
                    )
                )


def _milliseconds(seconds: float) -> float:
    return round(seconds * 1000, 3)


def format_logfmt_event(event: str, fields: dict[str, object]) -> str:
    """Format an event as stable, parseable logfmt-like tokens."""

    tokens = [event]
    for key, value in fields.items():
        if isinstance(value, str):
            formatted_value = json.dumps(value)
        elif value is None:
            formatted_value = "null"
        elif isinstance(value, bool):
            formatted_value = str(value).lower()
        else:
            formatted_value = str(value)
        tokens.append(f"{key}={formatted_value}")
    return " ".join(tokens)


def execution_resource_fields(prefix: str, resources: Any) -> dict[str, object]:
    """Flatten Ray ``ExecutionResources`` into stable scalar fields."""

    return {
        f"{prefix}_cpu": None if resources is None else resources.cpu,
        f"{prefix}_gpu": None if resources is None else resources.gpu,
        f"{prefix}_heap_memory": None if resources is None else resources.memory,
        f"{prefix}_object_store_memory": None if resources is None else resources.object_store_memory,
    }


def _object_store_memory_fields(resource_manager: Any, op: Any) -> dict[str, object]:
    return {
        "object_store_internal_bytes": resource_manager.get_mem_op_internal(op),
        "object_store_output_bytes": resource_manager.get_mem_op_outputs(op),
    }


def install_ray_data_diagnostics() -> DiagnosticsInstallStatus:
    """Install driver-side diagnostics without modifying the Ray installation.

    Diagnostics are opt-in through ``NEMO_CURATOR_RAY_DATA_DIAGNOSTICS``.
    The shim is intentionally restricted to the Ray version whose private APIs
    it targets.  A future Ray release containing the upstream diagnostics is
    detected and left untouched.
    """

    import ray

    with _INSTALL_LOCK:
        enabled = os.environ.get(RAY_DATA_DIAGNOSTICS_ENV_VAR, "").strip().lower()
        if enabled not in _TRUE_ENV_VALUES:
            return DiagnosticsInstallStatus.DISABLED

        if getattr(ray, _INSTALL_MARKER, False):
            return DiagnosticsInstallStatus.ALREADY_INSTALLED

        try:
            from ray.data._internal.actor_autoscaler import default_actor_autoscaler as autoscaler_module
            from ray.data._internal.execution import metadata_fetcher as metadata_fetcher_module
            from ray.data._internal.execution import resource_manager as resource_manager_module
            from ray.data._internal.execution import streaming_executor_state as executor_state_module
            from ray.data._internal.execution.backpressure_policy import (
                downstream_capacity_backpressure_policy as downstream_policy_module,
            )
            from ray.data._internal.execution.backpressure_policy import (
                resource_budget_backpressure_policy as resource_policy_module,
            )
        except ImportError:
            return DiagnosticsInstallStatus.UNSUPPORTED

        if _has_native_diagnostics(
            autoscaler_module,
            resource_manager_module,
            executor_state_module,
        ):
            return DiagnosticsInstallStatus.NATIVE

        if ray.__version__ != _SUPPORTED_RAY_VERSION:
            return DiagnosticsInstallStatus.UNSUPPORTED

        _install_resource_admission_diagnostics(resource_manager_module, resource_policy_module)
        _install_downstream_capacity_diagnostics(downstream_policy_module)
        _install_metadata_fetch_diagnostics(metadata_fetcher_module)
        _install_scheduling_reasons(executor_state_module)
        _install_actor_autoscaling_diagnostics(autoscaler_module)

        setattr(ray, _INSTALL_MARKER, True)
        return DiagnosticsInstallStatus.INSTALLED


def _has_native_diagnostics(autoscaler_module: Any, resource_manager_module: Any, executor_state_module: Any) -> bool:
    scheduling_fields = getattr(executor_state_module.OpSchedulingStatus, "__dataclass_fields__", {})
    return (
        hasattr(resource_manager_module.OpResourceAllocator, "get_task_admission_decision")
        and hasattr(autoscaler_module.DefaultActorAutoscaler, "_log_scaling_decision")
        and "reason" in scheduling_fields
    )


def _install_resource_admission_diagnostics(  # noqa: C901, PLR0915
    resource_manager_module: Any,
    resource_policy_module: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    allocator_cls = resource_manager_module.OpResourceAllocator
    reservation_cls = resource_manager_module.ReservationOpResourceAllocator
    policy_cls = resource_policy_module.ResourceBudgetBackpressurePolicy

    def get_generic_decision(self: Any, op: Any) -> _TaskAdmissionDecision:
        allowed = self.can_submit_new_task(op)
        return _TaskAdmissionDecision(
            allowed=allowed,
            reason="allowed" if allowed else "denied",
            incremental_resources=None,
            remaining_budget=None,
            pending_output_estimate=None,
        )

    def get_reservation_decision(self: Any, op: Any) -> _TaskAdmissionDecision:
        budget = self.get_budget(op)
        if budget is None:
            return _TaskAdmissionDecision(True, "unlimited", None, None, None)

        incremental = op.incremental_resource_usage()
        pending_output = op.metrics.obj_store_mem_max_pending_output_per_task or 0
        allowed = incremental.satisfies_limit(budget) and budget.object_store_memory >= pending_output
        if allowed:
            reason = "allowed"
        elif not incremental.cpu <= budget.cpu:
            reason = "incremental_cpu_exceeds_budget"
        elif not incremental.gpu <= budget.gpu:
            reason = "incremental_gpu_exceeds_budget"
        elif not incremental.memory <= budget.memory:
            reason = "incremental_heap_memory_exceeds_budget"
        elif not incremental.object_store_memory <= budget.object_store_memory:
            reason = "incremental_object_store_memory_exceeds_budget"
        else:
            reason = "pending_output_exceeds_object_store_budget"
        return _TaskAdmissionDecision(allowed, reason, incremental, budget, pending_output)

    allocator_cls.get_task_admission_decision = get_generic_decision
    reservation_cls.get_task_admission_decision = get_reservation_decision

    original_init = policy_cls.__init__

    def policy_init(self: Any, data_context: Any, topology: Any, resource_manager: Any) -> None:
        original_init(self, data_context, topology, resource_manager)
        self._nemo_curator_previous_decisions = {}
        self._nemo_curator_resource_blocked_since = {}

    def can_add_input(self: Any, op: Any) -> bool:
        allocator = self._resource_manager._op_resource_allocator
        if allocator is None:
            return True
        if not resource_policy_module.logger.isEnabledFor(logging.DEBUG):
            return allocator.can_submit_new_task(op)

        decision = allocator.get_task_admission_decision(op)
        signature = (decision.allowed, decision.reason)
        previous = self._nemo_curator_previous_decisions
        if previous.get(op) != signature:
            blocked_since = self._nemo_curator_resource_blocked_since
            if decision.allowed:
                started_at = blocked_since.pop(op, None)
                blocked_duration_ms = None if started_at is None else _milliseconds(clock() - started_at)
            else:
                blocked_since.setdefault(op, clock())
                blocked_duration_ms = None
            usage = None
            allocation = None
            if decision.remaining_budget is not None:
                usage = self._resource_manager.get_op_usage(op)
                allocation = decision.remaining_budget.add(usage)
            fields = {
                "operator": op.name,
                "state": "allowed" if decision.allowed else "blocked",
                "reason": decision.reason,
                **execution_resource_fields("requested", decision.incremental_resources),
                **execution_resource_fields("remaining_budget", decision.remaining_budget),
                "pending_output_estimate": decision.pending_output_estimate,
                **execution_resource_fields("usage", usage),
                **execution_resource_fields("allocation", allocation),
                **_object_store_memory_fields(self._resource_manager, op),
                "blocked_duration_ms": blocked_duration_ms,
            }
            resource_policy_module.logger.debug(format_logfmt_event("ray_data_resource_budget_admission", fields))
            previous[op] = signature
        return decision.allowed

    policy_cls.__init__ = policy_init
    policy_cls.can_add_input = can_add_input


def _install_downstream_capacity_diagnostics(
    downstream_policy_module: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    policy_cls = downstream_policy_module.DownstreamCapacityBackpressurePolicy
    original_init = policy_cls.__init__

    def policy_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._nemo_curator_downstream_blocked_since = {}

    def should_apply_backpressure(self: Any, op: Any) -> bool:
        if self._should_skip_backpressure(op):
            return False

        utilized_fraction = downstream_policy_module.get_utilized_object_store_budget_fraction(
            self._resource_manager,
            op,
            consider_downstream_ineligible_ops=True,
        )
        queue_ratio = self._get_queue_ratio(op)
        if utilized_fraction is not None and utilized_fraction <= self.OBJECT_STORE_BUDGET_UTIL_THRESHOLD:
            result = False
        else:
            result = queue_ratio > self._backpressure_capacity_ratio

        previous = self._prev_should_backpressure.get(op)
        if previous != result:
            blocked_since = self._nemo_curator_downstream_blocked_since
            if result:
                blocked_since.setdefault(op, clock())
                blocked_duration_ms = None
            else:
                started_at = blocked_since.pop(op, None)
                blocked_duration_ms = None if started_at is None else _milliseconds(clock() - started_at)
            queue_bytes = self._get_queue_size_bytes(op)
            downstream_capacity_bytes = self._get_downstream_capacity_size_bytes(op)
            downstream_policy_module.logger.debug(
                format_logfmt_event(
                    "ray_data_downstream_capacity_admission",
                    {
                        "operator": op.name,
                        "state": "blocked" if result else "allowed",
                        "queue_bytes": queue_bytes,
                        "downstream_capacity_bytes": downstream_capacity_bytes,
                        "queue_ratio": f"{queue_ratio:.2f}",
                        "configured_ratio": self._backpressure_capacity_ratio,
                        "utilized_object_store_budget_fraction": utilized_fraction,
                        **_object_store_memory_fields(self._resource_manager, op),
                        "blocked_duration_ms": blocked_duration_ms,
                    },
                )
            )
            self._prev_should_backpressure[op] = result
        return result

    policy_cls.__init__ = policy_init
    policy_cls._should_apply_backpressure = should_apply_backpressure


def _install_metadata_fetch_diagnostics(
    metadata_fetcher_module: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    _install_threaded_metadata_fetch_diagnostics(metadata_fetcher_module, clock)
    _install_inline_metadata_fetch_diagnostics(metadata_fetcher_module, clock)


def _install_threaded_metadata_fetch_diagnostics(metadata_fetcher_module: Any, clock: Callable[[], float]) -> None:
    threaded_cls = metadata_fetcher_module.ThreadedMetadataFetcher
    not_ready = metadata_fetcher_module._Signal.NOT_READY

    original_threaded_init = threaded_cls.__init__
    original_threaded_submit = threaded_cls.submit
    original_threaded_fetch = threaded_cls._fetch
    original_threaded_pop_result = threaded_cls._pop_result
    original_threaded_stop = threaded_cls.stop

    def threaded_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_threaded_init(self, *args, **kwargs)
        self._nemo_curator_metadata_fetch_diagnostics = _MetadataFetchDiagnostics(
            metadata_fetcher_module.logger,
            "threaded",
            clock,
        )

    def threaded_submit(self: Any, op_key: Any, tasks: list[Any]) -> None:
        deferred = list(self._pending_deferred)
        self._nemo_curator_metadata_fetch_diagnostics.submit(deferred)
        original_threaded_submit(self, op_key, tasks)

    def threaded_fetch(self: Any, pending: list[Any]) -> list[Any]:
        remaining = original_threaded_fetch(self, pending)
        self._nemo_curator_metadata_fetch_diagnostics.record_fetch_pass(pending, remaining)
        return remaining

    def threaded_pop_result(self: Any, ref: Any) -> Any:
        result = original_threaded_pop_result(self, ref)
        diagnostics = self._nemo_curator_metadata_fetch_diagnostics
        if result is not_ready:
            diagnostics.record_not_ready(ref)
        else:
            diagnostics.record_result(ref, failed=isinstance(result, BaseException))
        return result

    def threaded_stop(self: Any) -> None:
        try:
            original_threaded_stop(self)
        finally:
            self._nemo_curator_metadata_fetch_diagnostics.log_summaries()

    threaded_cls.__init__ = threaded_init
    threaded_cls.submit = threaded_submit
    threaded_cls._fetch = threaded_fetch
    threaded_cls._pop_result = threaded_pop_result
    threaded_cls.stop = threaded_stop


def _install_inline_metadata_fetch_diagnostics(metadata_fetcher_module: Any, clock: Callable[[], float]) -> None:
    inline_cls = metadata_fetcher_module.InlineMetadataFetcher
    original_inline_fetch = inline_cls.in_data_ready_get_object_size
    original_inline_stop = inline_cls.stop

    def inline_init(self: Any) -> None:
        self._nemo_curator_metadata_fetch_diagnostics = _MetadataFetchDiagnostics(
            metadata_fetcher_module.logger,
            "inline",
            clock,
        )

    def inline_fetch(self: Any, task: Any) -> int | None:
        diagnostics = self._nemo_curator_metadata_fetch_diagnostics
        ref = getattr(task, "pending_meta_ref", task)
        diagnostics.record_inline_start(ref, task.operator_name)
        try:
            result = original_inline_fetch(self, task)
        except BaseException:
            diagnostics.record_inline_result(ref, task.operator_name, emitted=True, failed=True)
            raise
        diagnostics.record_inline_result(ref, task.operator_name, emitted=result is not None, failed=False)
        return result

    def inline_stop(self: Any) -> None:
        try:
            original_inline_stop(self)
        finally:
            self._nemo_curator_metadata_fetch_diagnostics.log_summaries()

    inline_cls.__init__ = inline_init
    inline_cls.in_data_ready_get_object_size = inline_fetch
    inline_cls.stop = inline_stop


def _install_scheduling_reasons(executor_state_module: Any) -> None:  # noqa: C901
    # Existing constructors pass only runnable/under_resource_limits, so a class
    # default keeps them compatible.  Our scheduler hook adds an instance value.
    executor_state_module.OpSchedulingStatus.reason = "no_pending_inputs"

    def get_eligible_operators(
        topology: Any,
        backpressure_policies: list[Any],
        *,
        ensure_liveness: bool,
    ) -> list[Any]:
        dispatchable_ops = []
        eligible_ops = []

        for op, state in topology.items():
            triggered_policy = None
            for policy in backpressure_policies:
                if not policy.can_add_input(op):
                    triggered_policy = policy.name
                    break
            in_backpressure = triggered_policy is not None

            completed = op.has_completed()
            has_input_slot = op.can_add_input() if not completed else False
            has_pending_inputs = state.has_pending_bundles() if not completed and has_input_slot else False
            runnable = not completed and has_pending_inputs and has_input_slot and not in_backpressure

            if not completed and has_pending_inputs and has_input_slot:
                (dispatchable_ops if in_backpressure else eligible_ops).append(op)

            if completed:
                reason = "completed"
            elif not has_input_slot:
                reason = "no_actor_slot" if op.get_autoscaling_actor_pools() else "operator_cannot_accept_input"
            elif not has_pending_inputs:
                reason = "no_pending_inputs"
            elif triggered_policy is not None:
                reason = triggered_policy
            else:
                reason = "runnable"

            status = executor_state_module.OpSchedulingStatus(
                runnable=runnable,
                under_resource_limits=not in_backpressure,
            )
            status.reason = reason
            state._scheduling_status = status
            op.notify_in_task_submission_backpressure(in_backpressure, triggered_policy)

        if not eligible_ops and ensure_liveness and all(op.num_active_tasks() == 0 for op in topology):
            return dispatchable_ops
        return eligible_ops

    executor_state_module.get_eligible_operators = get_eligible_operators


def _install_actor_autoscaling_diagnostics(autoscaler_module: Any) -> None:
    autoscaler_cls = autoscaler_module.DefaultActorAutoscaler
    original_init = autoscaler_cls.__init__

    def autoscaler_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._nemo_curator_previous_scaling_decisions = {}

    def log_scaling_decision(  # noqa: PLR0913
        self: Any,
        op: Any,
        op_state: Any,
        actor_pool: Any,
        request: Any,
        decision: str,
        scheduling_reason: str,
    ) -> None:
        allocation = self._resource_manager.get_allocation(op)
        usage = self._resource_manager.get_op_usage(op)
        remaining_budget = self._resource_manager.get_budget(op)
        autoscaler_module.logger.debug(
            format_logfmt_event(
                "ray_data_actor_autoscaling_decision",
                {
                    "operator": op.name,
                    "decision": decision,
                    "delta": request.delta,
                    "scaling_reason": request.reason,
                    "scheduling_reason": scheduling_reason,
                    "current_actors": actor_pool.current_size(),
                    "min_actors": actor_pool.min_size(),
                    "max_actors": actor_pool.max_size(),
                    "running_actors": actor_pool.num_running_actors(),
                    "pending_actors": actor_pool.num_pending_actors(),
                    "active_actors": actor_pool.num_active_actors(),
                    "idle_actors": actor_pool.num_idle_actors(),
                    "utilization": actor_pool.get_pool_util(),
                    "tasks_in_flight": actor_pool.num_tasks_in_flight(),
                    "queued_input_blocks": op_state.total_enqueued_input_blocks(),
                    "queued_input_bytes": op_state.total_enqueued_input_blocks_bytes(),
                    **execution_resource_fields("allocation", allocation),
                    **execution_resource_fields("usage", usage),
                    **execution_resource_fields("remaining_budget", remaining_budget),
                    **_object_store_memory_fields(self._resource_manager, op),
                },
            )
        )

    def try_trigger_scaling(self: Any) -> None:
        for op, state in self._topology.items():
            for actor_pool in op.get_autoscaling_actor_pools():
                request = self._derive_target_scaling_config(actor_pool, op, state)
                decision = "scale_up" if request.delta > 0 else "scale_down" if request.delta < 0 else "no_op"
                scheduling_reason = state._scheduling_status.reason
                signature = (decision, request.delta, request.reason, scheduling_reason)
                previous = self._nemo_curator_previous_scaling_decisions
                if autoscaler_module.logger.isEnabledFor(logging.DEBUG) and previous.get(actor_pool) != signature:
                    self._log_scaling_decision(
                        op,
                        state,
                        actor_pool,
                        request,
                        decision,
                        scheduling_reason,
                    )
                    previous[actor_pool] = signature
                actor_pool.scale(request)

    autoscaler_cls.__init__ = autoscaler_init
    autoscaler_cls._log_scaling_decision = log_scaling_decision
    autoscaler_cls.try_trigger_scaling = try_trigger_scaling
