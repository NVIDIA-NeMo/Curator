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

from nemo_curator.core.serve import (
    DynamoRoleConfig,
    DynamoRouterConfig,
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
)
from nemo_curator.core.serve.internal.dynamo import DynamoBackend


class TestDynamoConfigResolution:
    def test_resolve_disagg_role_config_merges_role_engine_kwargs(self) -> None:
        model = DynamoVLLMModelConfig(
            model_identifier="m",
            engine_kwargs={"tensor_parallel_size": 2, "max_model_len": 8192},
            mode="disagg",
            prefill=DynamoRoleConfig(num_replicas=4, engine_kwargs={"tensor_parallel_size": 4}),
            decode=DynamoRoleConfig(num_replicas=2),
        )

        (num_prefill, prefill_engine_kwargs), (num_decode, decode_engine_kwargs) = (
            DynamoBackend._resolve_disagg_role_config(model)
        )

        assert num_prefill == 4
        assert num_decode == 2
        assert prefill_engine_kwargs == {"tensor_parallel_size": 4, "max_model_len": 8192}
        assert decode_engine_kwargs == {"tensor_parallel_size": 2, "max_model_len": 8192}

    def test_resolve_num_replicas_reads_typed_model_field(self) -> None:
        model = DynamoVLLMModelConfig(model_identifier="m", num_replicas=3)

        assert DynamoBackend._resolve_num_replicas(model) == 3

    def test_resolve_frontend_router_config_defaults_to_kv_for_disagg(self) -> None:
        server = InferenceServer(
            models=[DynamoVLLMModelConfig(model_identifier="m", mode="disagg")],
            backend=DynamoServerConfig(),
        )

        result = DynamoBackend._resolve_frontend_router_config(server)

        assert result["router_mode"] == "kv"
        assert result["router_kv_events"] is True

    def test_resolve_frontend_router_config_uses_server_router_config(self) -> None:
        server = InferenceServer(
            models=[DynamoVLLMModelConfig(model_identifier="m")],
            backend=DynamoServerConfig(
                router=DynamoRouterConfig(
                    mode="round-robin",
                    kv_events=False,
                    temperature=0.7,
                )
            ),
        )

        result = DynamoBackend._resolve_frontend_router_config(server)

        assert result["router_mode"] == "round-robin"
        assert result["router_kv_events"] is False
        assert result["router_temperature"] == 0.7
