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


class _Resources:
    def __init__(
        self,
        *,
        cpu: float = 0,
        gpu: float = 0,
        memory: float = 0,
        object_store_memory: float = 0,
    ) -> None:
        self.cpu = cpu
        self.gpu = gpu
        self.memory = memory
        self.object_store_memory = object_store_memory

    def satisfies_limit(self, limit: "_Resources") -> bool:
        return (
            self.cpu <= limit.cpu
            and self.gpu <= limit.gpu
            and self.memory <= limit.memory
            and self.object_store_memory <= limit.object_store_memory
        )

    def add(self, other: "_Resources") -> "_Resources":
        return _Resources(
            cpu=self.cpu + other.cpu,
            gpu=self.gpu + other.gpu,
            memory=self.memory + other.memory,
            object_store_memory=self.object_store_memory + other.object_store_memory,
        )


def _constant(value: object) -> Callable[..., object]:
    def return_value(*_args: object, **_kwargs: object) -> object:
        return value

    return return_value


def _make_metadata_fetcher_module() -> tuple[SimpleNamespace, object]:
    not_ready = object()

    class ThreadedMetadataFetcher:
        def __init__(self) -> None:
            self._pending_deferred = []
            self.fetch_remaining = []
            self.pop_results = []

        def submit(self, op_key: object, tasks: list[object]) -> None:
            self._pending_deferred = []

        def _fetch(self, pending: list[object]) -> list[object]:
            return self.fetch_remaining

        def _pop_result(self, ref: object) -> object:
            return self.pop_results.pop(0)

        def stop(self) -> None:
            pass

    class InlineMetadataFetcher:
        def in_data_ready_get_object_size(self, task: object) -> int:
            return 1

        def stop(self) -> None:
            pass

    module = SimpleNamespace(
        ThreadedMetadataFetcher=ThreadedMetadataFetcher,
        InlineMetadataFetcher=InlineMetadataFetcher,
        _Signal=SimpleNamespace(NOT_READY=not_ready),
        logger=logging.getLogger("test_metadata_fetcher"),
    )
    return module, not_ready


def test_threaded_metadata_fetch_diagnostics_report_stall_recovery_and_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    module, not_ready = _make_metadata_fetcher_module()
    clock = _Clock()
    caplog.set_level(logging.DEBUG, logger=module.logger.name)
    diagnostics._install_metadata_fetch_diagnostics(module, clock=clock)

    fetcher = module.ThreadedMetadataFetcher()
    metadata_ref = object()
    second_metadata_ref = object()
    task = SimpleNamespace(operator_name="ReadImages")
    fetcher._pending_deferred = [
        SimpleNamespace(task=task, meta_ref=metadata_ref),
        SimpleNamespace(task=task, meta_ref=second_metadata_ref),
    ]
    fetcher.submit(SimpleNamespace(op=SimpleNamespace(name="ReadImages")), [])

    clock.now = 1.1
    fetcher.pop_results = [not_ready]
    assert fetcher._pop_result(metadata_ref) is not_ready

    clock.now = 2.0
    fetcher.fetch_remaining = [second_metadata_ref]
    assert fetcher._fetch([metadata_ref, second_metadata_ref]) == [second_metadata_ref]

    clock.now = 2.5
    fetcher.pop_results = [b"metadata"]
    assert fetcher._pop_result(metadata_ref) == b"metadata"

    clock.now = 2.8
    fetcher.fetch_remaining = []
    assert fetcher._fetch([second_metadata_ref]) == []

    clock.now = 3.0
    fetcher.pop_results = [b"second metadata"]
    assert fetcher._pop_result(second_metadata_ref) == b"second metadata"
    fetcher.stop()

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "ray_data_metadata_fetch_state" in message
        and 'operator="ReadImages"' in message
        and 'state="stalled"' in message
        and "oldest_pending_ms=1100.0" in message
        for message in messages
    )
    assert any(
        "ray_data_metadata_fetch_state" in message
        and 'state="recovered"' in message
        and "stall_duration_ms=2800.0" in message
        for message in messages
    )
    assert any(
        "ray_data_metadata_fetch_summary" in message
        and 'mode="threaded"' in message
        and "submitted=2" in message
        and "emitted=2" in message
        and "fetch_latency_ms_total=4800.0" in message
        and "delivery_delay_ms_total=700.0" in message
        and "pending_high_watermark=2" in message
        for message in messages
    )


def test_inline_metadata_fetch_diagnostics_report_operator_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    module, _ = _make_metadata_fetcher_module()
    clock = _Clock()
    original_fetch = module.InlineMetadataFetcher.in_data_ready_get_object_size

    calls = 0

    def advance_during_fetch(self: object, task: object) -> int | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            clock.now = 1.1
            return None
        clock.now = 2.5
        return original_fetch(self, task)

    module.InlineMetadataFetcher.in_data_ready_get_object_size = advance_during_fetch
    caplog.set_level(logging.DEBUG, logger=module.logger.name)
    diagnostics._install_metadata_fetch_diagnostics(module, clock=clock)

    fetcher = module.InlineMetadataFetcher()
    task = SimpleNamespace(operator_name="ReadAudio", pending_meta_ref=object())
    assert fetcher.in_data_ready_get_object_size(task) is None
    assert fetcher.in_data_ready_get_object_size(task) == 1
    fetcher.stop()

    assert any(
        "ray_data_metadata_fetch_summary" in record.getMessage()
        and 'operator="ReadAudio"' in record.getMessage()
        and 'mode="inline"' in record.getMessage()
        and "submitted=1" in record.getMessage()
        and "emitted=1" in record.getMessage()
        and "fetch_latency_ms_total=2500.0" in record.getMessage()
        for record in caplog.records
    )
    assert any(
        "ray_data_metadata_fetch_state" in record.getMessage() and 'state="stalled"' in record.getMessage()
        for record in caplog.records
    )
    assert any(
        "ray_data_metadata_fetch_state" in record.getMessage()
        and 'state="recovered"' in record.getMessage()
        and "stall_duration_ms=2500.0" in record.getMessage()
        for record in caplog.records
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
            self.budget = _Resources(cpu=1, object_store_memory=1000)

        def get_budget(self, op: object) -> _Resources:
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
    usage = _Resources(cpu=1, object_store_memory=300)
    resource_manager = SimpleNamespace(
        _op_resource_allocator=allocator,
        get_op_usage=_constant(usage),
        get_mem_op_internal=_constant(100),
        get_mem_op_outputs=_constant(200),
    )

    class Op:
        name = "ReadImages"
        metrics = SimpleNamespace(obj_store_mem_max_pending_output_per_task=50)

        def incremental_resource_usage(self) -> _Resources:
            return _Resources(cpu=2, object_store_memory=10)

    op = Op()

    diagnostics._install_resource_admission_diagnostics(
        resource_manager_module,
        policy_module,
        clock=clock,
    )
    policy = ResourceBudgetBackpressurePolicy(None, None, resource_manager)
    assert not policy.can_add_input(op)

    clock.now = 2.5
    allocator.budget = _Resources(cpu=3, object_store_memory=1000)
    assert policy.can_add_input(op)

    assert any(
        "ray_data_resource_budget_admission" in record.getMessage()
        and 'state="allowed"' in record.getMessage()
        and "blocked_duration_ms=2500.0" in record.getMessage()
        and "object_store_internal_bytes=100" in record.getMessage()
        and "object_store_output_bytes=200" in record.getMessage()
        for record in caplog.records
    )


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

    assert any(
        "ray_data_downstream_capacity_admission" in record.getMessage()
        and 'state="allowed"' in record.getMessage()
        and "blocked_duration_ms=4000.0" in record.getMessage()
        and "object_store_internal_bytes=100" in record.getMessage()
        and "object_store_output_bytes=500" in record.getMessage()
        for record in caplog.records
    )


def test_actor_autoscaling_decision_reports_object_store_attribution(  # noqa: C901
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test_actor_autoscaling")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    class ActorPool:
        def current_size(self) -> int:
            return 4

        def min_size(self) -> int:
            return 1

        def max_size(self) -> int:
            return 8

        def num_running_actors(self) -> int:
            return 4

        def num_pending_actors(self) -> int:
            return 0

        def num_active_actors(self) -> int:
            return 3

        def num_idle_actors(self) -> int:
            return 1

        def get_pool_util(self) -> float:
            return 1.5

        def num_tasks_in_flight(self) -> int:
            return 6

        def scale(self, request: object) -> None:
            pass

    actor_pool = ActorPool()

    class Op:
        name = "ImageEmbedding"

        def get_autoscaling_actor_pools(self) -> list[ActorPool]:
            return [actor_pool]

    op = Op()
    state = SimpleNamespace(
        _scheduling_status=SimpleNamespace(reason="ResourceBudget"),
        total_enqueued_input_blocks=lambda: 10,
        total_enqueued_input_blocks_bytes=lambda: 1000,
    )
    resources = _Resources(cpu=4, gpu=1, object_store_memory=300)
    resource_manager = SimpleNamespace(
        get_allocation=_constant(resources),
        get_op_usage=_constant(resources),
        get_budget=_constant(resources),
        get_mem_op_internal=_constant(100),
        get_mem_op_outputs=_constant(200),
    )

    class DefaultActorAutoscaler:
        def __init__(self) -> None:
            self._resource_manager = resource_manager
            self._topology = {op: state}

        def _derive_target_scaling_config(self, pool: ActorPool, current_op: Op, current_state: object) -> object:
            return SimpleNamespace(delta=1, reason="high utilization")

    module = SimpleNamespace(DefaultActorAutoscaler=DefaultActorAutoscaler, logger=logger)
    diagnostics._install_actor_autoscaling_diagnostics(module)

    DefaultActorAutoscaler().try_trigger_scaling()

    assert any(
        "ray_data_actor_autoscaling_decision" in record.getMessage()
        and 'operator="ImageEmbedding"' in record.getMessage()
        and "object_store_internal_bytes=100" in record.getMessage()
        and "object_store_output_bytes=200" in record.getMessage()
        for record in caplog.records
    )


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
        "ray_data_metadata_fetch_summary",
        "ray_data_resource_budget_admission",
        "ray_data_downstream_capacity_admission",
        "ray_data_actor_autoscaling_decision",
    } <= set(log_contents.split())
