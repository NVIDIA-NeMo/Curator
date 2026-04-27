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

from typing import Any

import pytest

from nemo_curator.core.serve import InferenceServer, RayServeModelConfig, is_inference_server_active
from nemo_curator.tasks import DocumentBatch
from tests.core.serve.coexistence_utils import (
    COEXISTENCE_EXECUTOR_PARAMS,
    CaptureGpuStage,
    gpu_uuids_in_use,
)

INTEGRATION_TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # pragma: allowlist secret
INTEGRATION_TEST_MODEL_2 = "HuggingFaceTB/SmolLM-135M-Instruct"  # pragma: allowlist secret


@pytest.fixture(scope="class")
def model_server(shared_ray_cluster: str) -> InferenceServer:  # noqa: ARG001
    """Start InferenceServer once for all Ray Serve integration tests.

    Uses enforce_eager=True to skip torch.compile and CUDA graph capture,
    cutting vLLM startup from ~30s to ~5s.
    """
    config = RayServeModelConfig(
        model_identifier=INTEGRATION_TEST_MODEL,
        deployment_config={
            "autoscaling_config": {"min_replicas": 1, "max_replicas": 1},
        },
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 512,
            "enforce_eager": True,
        },
    )

    server = InferenceServer(models=[config], health_check_timeout_s=600)
    server.start()

    yield server
    server.stop()


@pytest.fixture(scope="class")
def inference_gpu_uuids(model_server: InferenceServer) -> set[str]:  # noqa: ARG001
    """Snapshot GPU UUIDs held by the running inference server.

    Captured once per class so residual processes from sibling pipeline
    runs (e.g. Ray Data actors that linger on the non-inference GPU)
    don't pollute the set when a later executor variant runs.
    """
    return gpu_uuids_in_use()


@pytest.mark.gpu
@pytest.mark.usefixtures("model_server")
class TestRayServeIntegration:
    """Full lifecycle tests against a real Ray Serve-backed InferenceServer."""

    def test_is_active_and_queryable(self, model_server: InferenceServer) -> None:
        """Server is active, lists models, and responds to chat completions."""
        from openai import OpenAI

        assert is_inference_server_active()
        assert model_server._started is True

        client = OpenAI(base_url=model_server.endpoint, api_key="na")

        model_ids = [model.id for model in client.models.list()]
        assert INTEGRATION_TEST_MODEL in model_ids

        response = client.chat.completions.create(
            model=INTEGRATION_TEST_MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
            temperature=0.0,
        )
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    def test_second_start_rejected(self, model_server: InferenceServer) -> None:
        """Cannot start a second InferenceServer while one is already active."""
        server2 = InferenceServer(
            models=[
                RayServeModelConfig(
                    model_identifier=INTEGRATION_TEST_MODEL_2,
                    deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 1}},
                    engine_kwargs={"tensor_parallel_size": 1, "max_model_len": 512, "enforce_eager": True},
                )
            ],
            health_check_timeout_s=600,
        )
        with pytest.raises(RuntimeError, match="already active"):
            server2.start()

        from openai import OpenAI

        client = OpenAI(base_url=model_server.endpoint, api_key="na")
        assert INTEGRATION_TEST_MODEL in {model.id for model in client.models.list()}

    @pytest.mark.parametrize(("executor_import", "executor_kwargs"), COEXISTENCE_EXECUTOR_PARAMS)
    def test_pipeline_gpu_stage_uses_different_gpu_than_inference(
        self,
        model_server: InferenceServer,
        inference_gpu_uuids: set[str],
        executor_import: tuple[str, str],
        executor_kwargs: dict[str, Any],
    ) -> None:
        """Pipeline GPU stages never land on the Ray Serve inference GPU.

        Reuses the class-scoped ``model_server`` fixture (no extra
        ``start()`` cost) and runs a 1-stage pipeline with 10 initial
        tasks. ``RayDataExecutor`` and ``RayActorPoolExecutor`` honor Ray's
        GPU accounting and pass. ``XennaExecutor`` is rejected upfront by
        ``Pipeline.run``'s inference-server guard (xfail).
        """
        import importlib

        import pandas as pd

        from nemo_curator.pipeline.pipeline import Pipeline

        module_name, cls_name = executor_import
        executor_cls = getattr(importlib.import_module(module_name), cls_name)

        assert inference_gpu_uuids, "expected the Ray Serve vLLM worker to be visible in gpustat"

        initial_tasks = [
            DocumentBatch(
                task_id=f"gpu-sep-{i}",
                dataset_name="ray-serve-coexistence",
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
        from openai import OpenAI

        client = OpenAI(base_url=model_server.endpoint, api_key="na")
        assert INTEGRATION_TEST_MODEL in {m.id for m in client.models.list()}

    def test_restart_after_stop(self, model_server: InferenceServer) -> None:
        """A new InferenceServer starts cleanly after the previous one is stopped."""
        from openai import OpenAI

        model_server.stop()
        assert not is_inference_server_active()

        config = RayServeModelConfig(
            model_identifier=INTEGRATION_TEST_MODEL,
            deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 1}},
            engine_kwargs={"tensor_parallel_size": 1, "max_model_len": 512, "enforce_eager": True},
        )
        server2 = InferenceServer(models=[config], health_check_timeout_s=600)
        server2.start()

        client = OpenAI(base_url=server2.endpoint, api_key="na")
        assert INTEGRATION_TEST_MODEL in {model.id for model in client.models.list()}

        response = client.chat.completions.create(
            model=INTEGRATION_TEST_MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
            temperature=0.0,
        )
        assert len(response.choices[0].message.content) > 0

        server2.stop()
