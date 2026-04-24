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

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest

from nemo_curator.core.serve import DynamoVLLMModelConfig
from nemo_curator.core.serve.dynamo import vllm as dynamo_vllm

_SINGLE_NODE_1GPU = [{"node_id": "n1", "num_gpus": 1, "is_head": False}]
_SINGLE_NODE_8GPU = [{"node_id": "n1", "num_gpus": 8, "is_head": False}]
_TWO_NODES_4GPU = [
    {"node_id": "n1", "num_gpus": 4, "is_head": False},
    {"node_id": "n2", "num_gpus": 4, "is_head": False},
]


@pytest.mark.parametrize(
    ("router_mode", "router_kv_events", "expected"),
    [
        ("round_robin", True, False),  # non-kv router never publishes
        ("kv", False, False),  # kv router without events opt-in stays approximate
        ("kv", True, True),  # kv router + events opt-in publishes ZMQ events
    ],
)
def test_aggregated_model_uses_exact_kv_events(router_mode: str, router_kv_events: bool, expected: bool) -> None:
    mc = DynamoVLLMModelConfig(model_identifier="m")
    assert (
        dynamo_vllm.aggregated_model_uses_exact_kv_events(
            mc, router_mode=router_mode, router_kv_events=router_kv_events
        )
        is expected
    )


class TestLaunchReplicas:
    @staticmethod
    def _launch(
        model_config: DynamoVLLMModelConfig,
        *,
        topology: list[dict[str, Any]],
        router_mode: str | None = None,
        router_kv_events: bool = False,
    ) -> None:
        """Run ``launch_replicas`` with real ``plan_replica_bundle_shape`` over
        the given *topology*; mock only the Ray PG + bundle-port plumbing."""
        with (
            mock.patch.object(dynamo_vllm, "build_replica_pg", return_value=object()),
            mock.patch.object(dynamo_vllm, "get_bundle_node_ip", return_value="10.0.0.5"),
            mock.patch.object(dynamo_vllm, "get_free_port_in_bundle", return_value=24567),
        ):
            dynamo_vllm.launch_replicas(
                model_config,
                base_env={"ETCD_ENDPOINTS": "http://10.0.0.5:2379", "NATS_SERVER": "nats://10.0.0.5:4222"},
                namespace="curator",
                request_plane="nats",
                event_plane="nats",
                runtime_dir="/tmp/rt",  # noqa: S108
                actor_name_prefix="dynamo_default_abcd1234",
                router_mode=router_mode,
                router_kv_events=router_kv_events,
                topology=topology,
            )

    def test_single_node_disables_kv_events_by_default(self, captured_spawn: list[dict[str, Any]]) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B", num_replicas=1)
        self._launch(mc, topology=_SINGLE_NODE_1GPU)

        assert len(captured_spawn) == 1
        python_args = captured_spawn[0]["python_args"]
        kv_cfg = json.loads(python_args[python_args.index("--kv-events-config") + 1])
        assert kv_cfg == {"enable_kv_cache_events": False}
        assert "--headless" not in python_args
        assert "--nnodes" not in python_args

    def test_kv_router_enables_exact_kv_events(self, captured_spawn: list[dict[str, Any]]) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B", num_replicas=1)
        self._launch(mc, topology=_SINGLE_NODE_1GPU, router_mode="kv", router_kv_events=True)

        python_args = captured_spawn[0]["python_args"]
        kv_cfg = json.loads(python_args[python_args.index("--kv-events-config") + 1])
        assert kv_cfg == {
            "enable_kv_cache_events": True,
            "endpoint": "tcp://*:24567",
            "publisher": "zmq",
            "topic": "kv-events",
        }

    def test_multi_node_rank0_adds_nnodes_and_master(self, captured_spawn: list[dict[str, Any]]) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            engine_kwargs={"tensor_parallel_size": 8},
            num_replicas=1,
        )
        # Two 4-GPU nodes force the planner to pick STRICT_SPREAD with nnodes=2.
        self._launch(mc, topology=_TWO_NODES_4GPU)
        assert len(captured_spawn) == 2

        rank0 = captured_spawn[0]["python_args"]
        assert rank0[rank0.index("--nnodes") + 1] == "2"
        assert rank0[rank0.index("--node-rank") + 1] == "0"
        assert rank0[rank0.index("--master-addr") + 1] == "10.0.0.5"
        assert "--headless" not in rank0

        # rank >0 runs headless (no scheduler => kv events always off on that rank).
        headless = captured_spawn[1]["python_args"]
        assert "--headless" in headless
        assert headless[headless.index("--node-rank") + 1] == "1"
        assert headless[headless.index("--master-addr") + 1] == "10.0.0.5"
        kv_cfg = json.loads(headless[headless.index("--kv-events-config") + 1])
        assert kv_cfg["enable_kv_cache_events"] is False

    def test_dynamo_kwargs_are_appended_as_cli_flags(self, captured_spawn: list[dict[str, Any]]) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            dynamo_kwargs={"tool_call_parser": "hermes", "reasoning_parser": "deepseek-r1"},
            num_replicas=1,
        )
        self._launch(mc, topology=_SINGLE_NODE_1GPU)

        python_args = captured_spawn[0]["python_args"]
        assert python_args[python_args.index("--tool-call-parser") + 1] == "hermes"
        assert python_args[python_args.index("--reasoning-parser") + 1] == "deepseek-r1"

    def test_num_replicas_fans_out_worker_spawns(self, captured_spawn: list[dict[str, Any]]) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B", num_replicas=3)
        self._launch(mc, topology=_SINGLE_NODE_8GPU)

        assert [c["label"] for c in captured_spawn] == [
            "Dynamo_DP0_Qwen3-0.6B",
            "Dynamo_DP1_Qwen3-0.6B",
            "Dynamo_DP2_Qwen3-0.6B",
        ]
