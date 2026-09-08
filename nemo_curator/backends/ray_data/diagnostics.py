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
from typing import Any

# Runtime monkeypatch callbacks necessarily accept objects owned by Ray's
# private, untyped implementation modules.
# ruff: noqa: ANN401

_SUPPORTED_RAY_VERSION = "2.57.0"
_INSTALL_MARKER = "_nemo_curator_ray_data_diagnostics_installed"
_INSTALL_LOCK = threading.Lock()
RAY_DATA_DIAGNOSTICS_ENV_VAR = "NEMO_CURATOR_RAY_DATA_DIAGNOSTICS"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


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
    resource_manager_module: Any, resource_policy_module: Any
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
                blocked_duration_ms = None if started_at is None else _milliseconds(time.perf_counter() - started_at)
            else:
                blocked_since.setdefault(op, time.perf_counter())
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


def _install_downstream_capacity_diagnostics(downstream_policy_module: Any) -> None:
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
                blocked_since.setdefault(op, time.perf_counter())
                blocked_duration_ms = None
            else:
                started_at = blocked_since.pop(op, None)
                blocked_duration_ms = None if started_at is None else _milliseconds(time.perf_counter() - started_at)
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
