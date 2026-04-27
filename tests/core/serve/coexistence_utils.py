# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared helpers for inference-server + pipeline-GPU coexistence tests.

Used by both ``dynamo/test_integration.py`` and
``ray_serve/test_integration.py`` to verify that a pipeline's GPU stages
get scheduled on a GPU that is not already held by the running
InferenceServer backend (Dynamo or Ray Serve).
"""

from __future__ import annotations

import os

import pytest

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


def gpu_uuids_in_use() -> set[str]:
    """GPU UUIDs currently in use by processes on **our** cluster's GPUs.

    ``gpustat`` uses NVML and sees every physical GPU on the host, so on a
    shared machine it picks up other users' processes on GPUs this test
    does not own. We restrict the scan to the physical indices listed in
    ``CUDA_VISIBLE_DEVICES`` (Ray inherits the same view when it schedules
    actors). When the variable is unset we fall back to every GPU.
    """
    import gpustat

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_indices: set[int] | None = None
    if cvd:
        try:
            visible_indices = {int(x.strip()) for x in cvd.split(",") if x.strip()}
        except ValueError:
            visible_indices = None  # CVD may carry UUIDs on some Ray versions

    return {
        gpu.uuid
        for gpu in gpustat.new_query()
        if gpu.processes and (visible_indices is None or gpu.index in visible_indices)
    }


class CaptureGpuStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """GPU-reserved pipeline stage that asserts it did not land on an inference GPU.

    Allocates a small CUDA tensor so the Ray worker process becomes visible
    in ``gpustat``, looks up its own GPU UUID, and asserts it's not in the
    ``inference_gpu_uuids`` snapshot the driver captured right before
    running the pipeline.
    """

    name = "capture_gpu"
    resources = Resources(gpus=1.0)
    batch_size = 1

    def __init__(self, inference_gpu_uuids: set[str]) -> None:
        super().__init__()
        self._inference_gpu_uuids = set(inference_gpu_uuids)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> DocumentBatch:
        import gpustat
        import torch

        _ = torch.zeros(1, device="cuda")

        my_pid = os.getpid()
        my_gpu_uuid = next(
            (gpu.uuid for gpu in gpustat.new_query() if any(p["pid"] == my_pid for p in gpu.processes)),
            None,
        )
        assert my_gpu_uuid is not None, f"pid {my_pid} not visible in gpustat after CUDA allocation"
        assert my_gpu_uuid not in self._inference_gpu_uuids, (
            f"Pipeline stage (pid={my_pid}) landed on inference GPU {my_gpu_uuid}. "
            f"inference_gpu_uuids={sorted(self._inference_gpu_uuids)}"
        )
        return task


COEXISTENCE_EXECUTOR_PARAMS = [
    # Same matrix for Dynamo and Ray Serve integration tests so both
    # backends exercise identical coexistence semantics. Xenna is xfailed
    # because ``Pipeline.run`` explicitly rejects it when any inference
    # server is active (see ``pipeline.py``). Letting Xenna run with the
    # guard disabled produces a genuine GPU overlap — the guard is
    # preventing a real bug, not being overly cautious.
    pytest.param(
        ("nemo_curator.backends.ray_data", "RayDataExecutor"),
        {},
        id="ray_data",
    ),
    pytest.param(
        ("nemo_curator.backends.ray_actor_pool", "RayActorPoolExecutor"),
        {},
        id="ray_actor_pool",
    ),
    pytest.param(
        ("nemo_curator.backends.xenna", "XennaExecutor"),
        {"execution_mode": "streaming"},
        id="xenna",
        marks=pytest.mark.xfail(
            strict=True,
            raises=RuntimeError,
            reason=(
                "Pipeline.run guards against XennaExecutor + GPU stages while any "
                "InferenceServer is active (see pipeline.py): Xenna schedules GPU "
                "actors independently of Ray's resource accounting, so it would "
                "overlap the serving backend's GPU. The guard raises RuntimeError "
                "upfront instead of letting the overlap happen."
            ),
        ),
    ),
]
