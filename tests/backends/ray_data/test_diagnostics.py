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

import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources

from nemo_curator.backends.ray_data import diagnostics
from nemo_curator.backends.ray_data.diagnostics import (
    RAY_DATA_DIAGNOSTICS_ENV_VAR,
    DiagnosticsInstallStatus,
    execution_resource_fields,
    format_logfmt_event,
    install_ray_data_diagnostics,
)


class _IdentityActor:
    def __call__(self, batch: dict) -> dict:
        return batch


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _constant(value: object) -> Callable[..., object]:
    def return_value(*_args: object, **_kwargs: object) -> object:
        return value

    return return_value


def _event_message(caplog: pytest.LogCaptureFixture, event: str, state: str) -> str:
    return next(
        record.getMessage()
        for record in caplog.records
        if event in record.getMessage() and f'state="{state}"' in record.getMessage()
    )


def test_resource_admission_recovery_reports_blocked_duration_and_object_store_attribution(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _Clock()
    logger = logging.getLogger("test_resource_admission")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    class OpResourceAllocator:
        def can_submit_new_task(self, op: object) -> bool:
            return True

    class ReservationOpResourceAllocator(OpResourceAllocator):
        def __init__(self) -> None:
            self.budget = ExecutionResources(cpu=1, object_store_memory=1000)

        def get_budget(self, op: object) -> ExecutionResources:
            return self.budget

    class ResourceBudgetBackpressurePolicy:
        def __init__(self, data_context: object, topology: object, resource_manager: object) -> None:
            self._resource_manager = resource_manager

    resource_manager_module = SimpleNamespace(
        OpResourceAllocator=OpResourceAllocator,
        ReservationOpResourceAllocator=ReservationOpResourceAllocator,
    )
    policy_module = SimpleNamespace(
        ResourceBudgetBackpressurePolicy=ResourceBudgetBackpressurePolicy,
        logger=logger,
    )
    allocator = ReservationOpResourceAllocator()
    usage = ExecutionResources(cpu=1, object_store_memory=300)
    resource_manager = SimpleNamespace(
        _op_resource_allocator=allocator,
        get_op_usage=_constant(usage),
        get_mem_op_internal=_constant(100),
        get_mem_op_outputs=_constant(200),
    )

    class Op:
        name = "ReadImages"
        metrics = SimpleNamespace(obj_store_mem_max_pending_output_per_task=50)

        def incremental_resource_usage(self) -> ExecutionResources:
            return ExecutionResources(cpu=2, object_store_memory=10)

    op = Op()

    diagnostics._install_resource_admission_diagnostics(
        resource_manager_module,
        policy_module,
        clock=clock,
    )
    policy = ResourceBudgetBackpressurePolicy(None, None, resource_manager)
    assert not policy.can_add_input(op)

    clock.now = 2.5
    allocator.budget = ExecutionResources(cpu=3, object_store_memory=1000)
    assert policy.can_add_input(op)

    recovery = _event_message(caplog, "ray_data_resource_budget_admission", "allowed")
    assert "blocked_duration_ms=2500.0" in recovery
    assert "object_store_internal_bytes=100" in recovery
    assert "object_store_output_bytes=200" in recovery


def test_downstream_capacity_recovery_reports_blocked_duration_and_object_store_attribution(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _Clock()
    logger = logging.getLogger("test_downstream_capacity")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    class DownstreamCapacityBackpressurePolicy:
        OBJECT_STORE_BUDGET_UTIL_THRESHOLD = 0.5

        def __init__(self, resource_manager: object) -> None:
            self._resource_manager = resource_manager
            self._prev_should_backpressure = {}
            self._backpressure_capacity_ratio = 2.0
            self.queue_ratio = 3.0

        def _should_skip_backpressure(self, op: object) -> bool:
            return False

        def _get_queue_ratio(self, op: object) -> float:
            return self.queue_ratio

        def _get_queue_size_bytes(self, op: object) -> int:
            return 600

        def _get_downstream_capacity_size_bytes(self, op: object) -> int:
            return 200

    resource_manager = SimpleNamespace(
        get_mem_op_internal=_constant(100),
        get_mem_op_outputs=_constant(500),
    )
    module = SimpleNamespace(
        DownstreamCapacityBackpressurePolicy=DownstreamCapacityBackpressurePolicy,
        get_utilized_object_store_budget_fraction=_constant(0.75),
        logger=logger,
    )
    op = type("Op", (), {"name": "ReadImages"})()

    diagnostics._install_downstream_capacity_diagnostics(module, clock=clock)
    policy = DownstreamCapacityBackpressurePolicy(resource_manager)
    assert policy._should_apply_backpressure(op)

    clock.now = 4.0
    policy.queue_ratio = 1.0
    assert not policy._should_apply_backpressure(op)

    recovery = _event_message(caplog, "ray_data_downstream_capacity_admission", "allowed")
    assert "blocked_duration_ms=4000.0" in recovery
    assert "object_store_internal_bytes=100" in recovery
    assert "object_store_output_bytes=500" in recovery


def test_logfmt_event_escapes_strings_and_flattens_resources() -> None:
    resources = type(
        "Resources",
        (),
        {"cpu": 2.0, "gpu": 1.0, "memory": 3.0, "object_store_memory": 4.0},
    )()

    fields = {
        "reason": 'limited by "memory"',
        "allowed": False,
        "missing": None,
        **execution_resource_fields("requested", resources),
    }

    assert format_logfmt_event("event", fields) == (
        'event reason="limited by \\"memory\\"" allowed=false missing=null '
        "requested_cpu=2.0 requested_gpu=1.0 requested_heap_memory=3.0 "
        "requested_object_store_memory=4.0"
    )


def test_install_ray_data_diagnostics_is_opt_in_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RAY_DATA_DIAGNOSTICS_ENV_VAR, raising=False)
    assert install_ray_data_diagnostics() is DiagnosticsInstallStatus.DISABLED

    monkeypatch.setenv(RAY_DATA_DIAGNOSTICS_ENV_VAR, "1")
    first_status = install_ray_data_diagnostics()
    second_status = install_ray_data_diagnostics()

    assert first_status in {DiagnosticsInstallStatus.INSTALLED, DiagnosticsInstallStatus.NATIVE}
    if first_status is DiagnosticsInstallStatus.INSTALLED:
        assert second_status is DiagnosticsInstallStatus.ALREADY_INSTALLED
        from ray.data._internal.actor_autoscaler.default_actor_autoscaler import DefaultActorAutoscaler
        from ray.data._internal.execution.resource_manager import OpResourceAllocator

        assert hasattr(DefaultActorAutoscaler, "_log_scaling_decision")
        assert hasattr(OpResourceAllocator, "get_task_admission_decision")
    else:
        assert second_status is DiagnosticsInstallStatus.NATIVE


def test_scheduler_diagnostics_are_written_to_ray_session_log(
    shared_ray_client: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RAY_DATA_DIAGNOSTICS_ENV_VAR, "1")
    install_ray_data_diagnostics()

    ray.data.range(8, override_num_blocks=4).map_batches(
        _IdentityActor,
        concurrency=(1, 2),
        batch_size=1,
    ).materialize()

    session_dir = Path(ray._private.worker._global_node.get_session_dir_path())
    ray_data_log = session_dir / "logs" / "ray-data" / "ray-data.log"

    assert ray_data_log.exists()
    log_contents = ray_data_log.read_text()
    assert {
        "ray_data_resource_budget_admission",
        "ray_data_downstream_capacity_admission",
        "ray_data_actor_autoscaling_decision",
    } <= set(log_contents.split())
    actor_event = next(line for line in log_contents.splitlines() if "ray_data_actor_autoscaling_decision" in line)
    assert "object_store_internal_bytes=" in actor_event
    assert "object_store_output_bytes=" in actor_event
