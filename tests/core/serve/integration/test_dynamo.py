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

"""GPU integration tests for the Dynamo backend (aggregated + disaggregated)."""

from __future__ import annotations

import pytest

from nemo_curator.core.serve import (
    DynamoRoleConfig,
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
    is_inference_server_active,
)

INTEGRATION_TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # pragma: allowlist secret


def _make_aggregated_server() -> InferenceServer:
    """Build a single-model aggregated Dynamo server used by the GPU integration tests.

    ``enforce_eager=True`` skips torch.compile + CUDA graph capture so the
    worker becomes ready in roughly the same window as the Ray Serve
    integration test.
    """
    model = DynamoVLLMModelConfig(
        model_identifier=INTEGRATION_TEST_MODEL,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 512,
            "enforce_eager": True,
        },
        num_replicas=1,
    )
    return InferenceServer(
        models=[model],
        backend=DynamoServerConfig(),
        health_check_timeout_s=600,
    )


@pytest.fixture(scope="class")
def dynamo_aggregated_server(shared_ray_cluster: str) -> InferenceServer:  # noqa: ARG001
    server = _make_aggregated_server()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.gpu
@pytest.mark.usefixtures("dynamo_aggregated_server")
class TestDynamoAggregatedSingleNode:
    """End-to-end: real Dynamo frontend + etcd + NATS + one vLLM worker."""

    def test_is_active_and_queryable(self, dynamo_aggregated_server: InferenceServer) -> None:
        """Server is active, lists the registered model, and answers chat completions."""
        from openai import OpenAI

        assert is_inference_server_active()
        assert dynamo_aggregated_server._started is True

        client = OpenAI(base_url=dynamo_aggregated_server.endpoint, api_key="na")

        model_ids = {m.id for m in client.models.list()}
        assert INTEGRATION_TEST_MODEL in model_ids

        response = client.chat.completions.create(
            model=INTEGRATION_TEST_MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
            temperature=0.0,
        )
        assert response.choices
        assert response.choices[0].message.content


@pytest.mark.gpu
class TestDynamoRestartAfterStop:
    """Exercises the orphan-PG and orphan-actor sweeps by stopping and restarting.

    This lives in its own class (with its own stop-early-own server) so it does
    not mutate the class-scoped fixture shared by ``TestDynamoAggregatedSingleNode``.
    Test-ordering randomizers (e.g. ``pytest-randomly``) would otherwise leave the
    shared server stopped before other tests run.
    """

    def test_restart_after_stop(self, shared_ray_cluster: str) -> None:
        from openai import OpenAI

        server = _make_aggregated_server()
        server.start()
        server.stop()
        assert not is_inference_server_active()

        server2 = _make_aggregated_server()
        server2.start()
        try:
            client = OpenAI(base_url=server2.endpoint, api_key="na")
            assert INTEGRATION_TEST_MODEL in {m.id for m in client.models.list()}
        finally:
            server2.stop()


def _make_disagg_server() -> InferenceServer:
    """Build a single-model disagg Dynamo server (1 prefill + 1 decode, each TP=1)."""
    engine_kwargs = {
        "tensor_parallel_size": 1,
        "max_model_len": 512,
        "enforce_eager": True,
    }
    model = DynamoVLLMModelConfig(
        model_identifier=INTEGRATION_TEST_MODEL,
        mode="disagg",
        engine_kwargs=engine_kwargs,
        prefill=DynamoRoleConfig(num_replicas=1),
        decode=DynamoRoleConfig(num_replicas=1),
    )
    return InferenceServer(
        models=[model],
        backend=DynamoServerConfig(),
        health_check_timeout_s=600,
    )


@pytest.mark.gpu
class TestDynamoDisaggregated:
    """End-to-end: real Dynamo frontend + etcd + NATS + 1 prefill + 1 decode worker."""

    def test_disagg_serves_chat_completions(self, shared_ray_cluster: str) -> None:
        from openai import OpenAI

        server = _make_disagg_server()
        server.start()
        try:
            assert is_inference_server_active()
            client = OpenAI(base_url=server.endpoint, api_key="na")
            model_ids = {m.id for m in client.models.list()}
            assert INTEGRATION_TEST_MODEL in model_ids

            response = client.chat.completions.create(
                model=INTEGRATION_TEST_MODEL,
                messages=[{"role": "user", "content": "Say hi."}],
                max_tokens=16,
                temperature=0.0,
            )
            assert response.choices
            assert response.choices[0].message.content
        finally:
            server.stop()
        assert not is_inference_server_active()
