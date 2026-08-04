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

from pathlib import Path

import ray

from nemo_curator.backends.ray_data.diagnostics import (
    DiagnosticsInstallStatus,
    execution_resource_fields,
    format_logfmt_event,
    install_ray_data_diagnostics,
)


class _IdentityActor:
    def __call__(self, batch: dict) -> dict:
        return batch


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


def test_install_ray_data_diagnostics_is_idempotent() -> None:
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


def test_scheduler_diagnostics_are_written_to_ray_session_log(shared_ray_client: None) -> None:  # noqa: ARG001
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
