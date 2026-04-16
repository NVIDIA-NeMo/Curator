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

import pytest

from nemo_curator.core.serve import (
    DynamoRoleConfig,
    DynamoRouterConfig,
    DynamoVLLMModelConfig,
)


class TestDynamoRoleConfig:
    def test_post_init(self) -> None:
        # 0 is allowed at the role level, though DynamoVLLMModelConfig rejects 0 on either role when mode="disagg".
        DynamoRoleConfig(num_replicas=0)
        with pytest.raises(ValueError, match="num_replicas must be >= 0"):
            DynamoRoleConfig(num_replicas=-1)


class TestDynamoRouterConfig:
    def test_post_init(self) -> None:
        # Invalid: non-kv mode leaving kv_events=True is contradictory
        with pytest.raises(ValueError, match="kv_events is only meaningful when mode='kv'"):
            DynamoRouterConfig(mode="round_robin", kv_events=True)


class TestDynamoVLLMModelConfig:
    def test_post_init(self) -> None:
        # num_replicas must be >= 1 on the model (unlike DynamoRoleConfig which allows 0)
        with pytest.raises(ValueError, match="num_replicas must be >= 1"):
            DynamoVLLMModelConfig(model_identifier="m", num_replicas=0)

        # prefill/decode belong to disagg only
        with pytest.raises(ValueError, match="prefill/decode are only valid"):
            DynamoVLLMModelConfig(model_identifier="m", mode="aggregated", prefill=DynamoRoleConfig())

        # disagg requires both-or-neither for prefill/decode
        with pytest.raises(ValueError, match="both prefill and decode"):
            DynamoVLLMModelConfig(model_identifier="m", mode="disagg", prefill=DynamoRoleConfig())
        with pytest.raises(ValueError, match="both prefill and decode"):
            DynamoVLLMModelConfig(model_identifier="m", mode="disagg", decode=DynamoRoleConfig())

        # disagg requires each role to actually deploy workers — num_replicas=0 is invalid
        with pytest.raises(ValueError, match="num_replicas >= 1"):
            DynamoVLLMModelConfig(
                model_identifier="m",
                mode="disagg",
                prefill=DynamoRoleConfig(num_replicas=0),
                decode=DynamoRoleConfig(num_replicas=1),
            )
        with pytest.raises(ValueError, match="num_replicas >= 1"):
            DynamoVLLMModelConfig(
                model_identifier="m",
                mode="disagg",
                prefill=DynamoRoleConfig(num_replicas=1),
                decode=DynamoRoleConfig(num_replicas=0),
            )

    def test_curator_managed_fields(self) -> None:
        # kv_events_config / kv_transfer_config are init=False — Curator manages them
        with pytest.raises(TypeError, match="kv_events_config"):
            DynamoVLLMModelConfig(model_identifier="m", kv_events_config={"publisher": "zmq"})
        with pytest.raises(TypeError, match="kv_transfer_config"):
            DynamoVLLMModelConfig(model_identifier="m", kv_transfer_config={"kv_connector": "Nixl"})
