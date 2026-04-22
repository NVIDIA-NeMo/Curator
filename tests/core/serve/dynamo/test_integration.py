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

"""End-to-end GPU integration tests for the Dynamo backend.

Layout (each ``server.start()`` is ~100 s, so classes batch as many
assertions as possible onto one class-scoped fixture):

- ``TestDynamoSingleGpuServer``: one 1-GPU aggregated server shared across
  (a) pipeline-GPU-stage coexistence — parametrized over Ray Data, Ray
  Actor Pool, and Xenna executors — and (b) restart-after-stop. The
  restart test runs **last in file order** because it mutates the shared
  fixture (pytest preserves file order by default; the repo does not use
  pytest-randomly).
- ``TestDynamoServes``: parametrized across multi-model aggregated and
  disaggregated. Each parametrization spins up one server once and the
  single test method exercises every registered model.

Total across the whole file: **4** ``start()`` calls (1 shared fixture +
1 restart + 2 parametrizations). Every configuration uses **at most 2
GPUs** to match the GPU CI runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from nemo_curator.core.serve import (
    DynamoRoleConfig,
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
    is_inference_server_active,
)
from nemo_curator.tasks import DocumentBatch
from tests.core.serve.coexistence_utils import (
    COEXISTENCE_EXECUTOR_PARAMS,
    CaptureGpuStage,
    gpu_uuids_in_use,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

INTEGRATION_TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # pragma: allowlist secret
INTEGRATION_TEST_MODEL_2 = "HuggingFaceTB/SmolLM-135M-Instruct"  # pragma: allowlist secret


def _engine_kwargs() -> dict[str, Any]:
    """Shared vLLM engine kwargs that minimise startup time on integration runs."""
    return {
        "tensor_parallel_size": 1,
        "max_model_len": 512,
        "enforce_eager": True,
    }


def _make_single_aggregated_server() -> InferenceServer:
    """Single aggregated model — uses 1 GPU."""
    return InferenceServer(
        models=[
            DynamoVLLMModelConfig(
                model_identifier=INTEGRATION_TEST_MODEL,
                engine_kwargs=_engine_kwargs(),
            ),
        ],
        backend=DynamoServerConfig(),
        health_check_timeout_s=600,
    )


def _make_multi_aggregated_server() -> InferenceServer:
    """Two distinct aggregated models, each TP=1 — uses 2 GPUs total."""
    return InferenceServer(
        models=[
            DynamoVLLMModelConfig(
                model_identifier=INTEGRATION_TEST_MODEL,
                engine_kwargs=_engine_kwargs(),
            ),
            DynamoVLLMModelConfig(
                model_identifier=INTEGRATION_TEST_MODEL_2,
                engine_kwargs=_engine_kwargs(),
            ),
        ],
        backend=DynamoServerConfig(),
        health_check_timeout_s=600,
    )


def _make_disagg_server() -> InferenceServer:
    """Single disaggregated model: 1 prefill + 1 decode, each TP=1 — uses 2 GPUs total."""
    return InferenceServer(
        models=[
            DynamoVLLMModelConfig(
                model_identifier=INTEGRATION_TEST_MODEL,
                mode="disagg",
                engine_kwargs=_engine_kwargs(),
                prefill=DynamoRoleConfig(num_replicas=1),
                decode=DynamoRoleConfig(num_replicas=1),
            ),
        ],
        backend=DynamoServerConfig(),
        health_check_timeout_s=600,
    )


# ---------------------------------------------------------------------------
# TestDynamoSingleGpuServer — one shared 1-GPU server for coexistence + restart
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def single_gpu_server(shared_ray_cluster: str) -> Iterator[InferenceServer]:  # noqa: ARG001
    """One aggregated TP=1 server shared across every method in the class.

    Class scope so the ~100 s ``start()`` amortises across multiple
    assertions. ``_started`` is re-checked on teardown — the restart test
    stops this server early, and that's fine.
    """
    server = _make_single_aggregated_server()
    server.start()
    try:
        yield server
    finally:
        if server._started:
            server.stop()


@pytest.mark.gpu
@pytest.mark.usefixtures("single_gpu_server")
class TestDynamoSingleGpuServer:
    """One 1-GPU aggregated Dynamo server; multiple assertions share it."""

    @pytest.mark.parametrize(("executor_import", "executor_kwargs"), COEXISTENCE_EXECUTOR_PARAMS)
    def test_pipeline_gpu_stage_uses_different_gpu_than_inference(
        self,
        single_gpu_server: InferenceServer,
        executor_import: tuple[str, str],
        executor_kwargs: dict[str, Any],
    ) -> None:
        """Pipeline GPU stages never land on the inference server's GPU.

        Runs a 1-stage pipeline with **10 initial tasks** so the executor has
        enough work to parallelise and (if buggy) try to grab more than one
        GPU concurrently. On a 2-GPU runner with 1 GPU held by the Dynamo
        worker, only 1 GPU is free — the pipeline must schedule every stage
        actor onto the non-inference GPU.

        ``RayDataExecutor`` and ``RayActorPoolExecutor`` honor Ray's GPU
        accounting and pass. ``XennaExecutor`` is expected to fail (marked
        ``xfail(strict=True)``); it schedules against the cluster's raw GPU
        count and overlaps with the detached Dynamo actor's reservation.

        The stage itself does the overlap check: it allocates CUDA, finds
        its own GPU via ``gpustat``, and asserts it is the sole process on
        that GPU. No driver-side snapshot required.
        """
        import importlib

        import pandas as pd
        from openai import OpenAI

        from nemo_curator.pipeline.pipeline import Pipeline

        module_name, cls_name = executor_import
        executor_cls = getattr(importlib.import_module(module_name), cls_name)

        # Snapshot inference's GPUs just before the pipeline runs. At this
        # point the single_gpu_server fixture has already awaited
        # ``/v1/models`` and any prior parametrization's Ray actors have
        # been shut down by their executor, so the only process on our
        # cluster's GPUs is the Dynamo EngineCore.
        inference_gpu_uuids = gpu_uuids_in_use()
        assert inference_gpu_uuids, "expected the Dynamo worker to be visible in gpustat"

        initial_tasks = [
            DocumentBatch(
                task_id=f"gpu-sep-{i}",
                dataset_name="dynamo-coexistence",
                data=pd.DataFrame({"text": [f"hello {i}"]}),
            )
            for i in range(10)
        ]
        pipeline = Pipeline(name="gpu-sep", stages=[CaptureGpuStage(inference_gpu_uuids)])
        executor = executor_cls(executor_kwargs) if executor_kwargs else executor_cls()
        outputs = pipeline.run(executor, initial_tasks=initial_tasks)
        assert outputs, "pipeline produced no output tasks"

        # Server must still answer /v1/models after the pipeline hit the cluster.
        assert is_inference_server_active()
        client = OpenAI(base_url=single_gpu_server.endpoint, api_key="na")
        assert INTEGRATION_TEST_MODEL in {m.id for m in client.models.list()}

    def test_restart_after_stop(self, single_gpu_server: InferenceServer) -> None:
        """Stop the shared fixture, start a fresh server, verify it serves.

        Exercises the ``start()`` orphan-PG and orphan-actor sweeps; if the
        first ``stop()`` left state behind, the second ``start()`` would
        refuse to place a PG with the same name or hit an actor-name
        collision.

        Must be the **last** method in file order — it mutates the
        class-scoped fixture.
        """
        from openai import OpenAI

        single_gpu_server.stop()
        assert not is_inference_server_active()

        server2 = _make_single_aggregated_server()
        server2.start()
        try:
            client = OpenAI(base_url=server2.endpoint, api_key="na")
            assert INTEGRATION_TEST_MODEL in {m.id for m in client.models.list()}
        finally:
            server2.stop()


# ---------------------------------------------------------------------------
# TestDynamoServes — multi-model aggregated AND disaggregated, parametrized
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="class",
    params=[
        pytest.param(_make_multi_aggregated_server, id="multi_model_aggregated"),
        pytest.param(_make_disagg_server, id="disagg_single_model"),
    ],
)
def dynamo_server(request: pytest.FixtureRequest, shared_ray_cluster: str) -> Iterator[InferenceServer]:  # noqa: ARG001
    """Start one server per parametrization, teardown at class scope."""
    server = request.param()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.gpu
class TestDynamoServes:
    """Multi-model aggregated and disaggregated both serve chat completions."""

    def test_serves_registered_models(self, dynamo_server: InferenceServer) -> None:
        """Every configured model is registered and answers chat completions."""
        from openai import OpenAI

        assert is_inference_server_active()
        assert dynamo_server._started is True

        client = OpenAI(base_url=dynamo_server.endpoint, api_key="na")

        expected = {m.resolved_model_name for m in dynamo_server.models}
        served = {m.id for m in client.models.list()}
        assert expected.issubset(served), f"missing models: expected={expected} served={served}"

        for model_name in expected:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say hi."}],
                max_tokens=8,
                temperature=0.0,
            )
            assert response.choices, f"no choices for {model_name}"
            assert response.choices[0].message.content, f"empty content for {model_name}"
