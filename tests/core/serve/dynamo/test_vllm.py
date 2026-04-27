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
from nemo_curator.core.serve.dynamo.config import DynamoRoleConfig

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


# ---------------------------------------------------------------------------
# Disagg helpers + worker-launch path
# ---------------------------------------------------------------------------


class TestResolveDisaggRoleConfig:
    def test_merges_base_and_role_engine_kwargs(self) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            mode="disagg",
            engine_kwargs={"max_model_len": 4096, "tensor_parallel_size": 2},
            prefill=DynamoRoleConfig(num_replicas=2, engine_kwargs={"tensor_parallel_size": 1}),
            decode=DynamoRoleConfig(num_replicas=3),
        )
        num_p, ek_p = dynamo_vllm.resolve_disagg_role_config(mc, "prefill")
        num_d, ek_d = dynamo_vllm.resolve_disagg_role_config(mc, "decode")
        # Role overrides win for overlapping keys; base kwargs flow through otherwise.
        assert (num_p, ek_p) == (2, {"max_model_len": 4096, "tensor_parallel_size": 1})
        assert (num_d, ek_d) == (3, {"max_model_len": 4096, "tensor_parallel_size": 2})

    def test_rejects_non_disagg(self) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="m", mode="aggregated")
        with pytest.raises(ValueError, match="requires mode='disagg'"):
            dynamo_vllm.resolve_disagg_role_config(mc, "prefill")


class TestPlanDisaggShape:
    """Uses the real ``plan_replica_bundle_shape`` against injected topology."""

    def test_rejects_multi_node_tp(self) -> None:
        # TP=8 can't fit on any node in a 4-GPU cluster → planner picks STRICT_SPREAD.
        with pytest.raises(ValueError, match="multi-node TP"):
            dynamo_vllm.plan_disagg_shape(8, role="prefill", worker_index=0, model_name="m", topology=_TWO_NODES_4GPU)

    def test_accepts_single_node(self) -> None:
        spec = dynamo_vllm.plan_disagg_shape(
            2, role="decode", worker_index=0, model_name="m", topology=_SINGLE_NODE_8GPU
        )
        assert spec.nnodes == 1
        assert spec.per_node_gpus == 2


class TestLaunchDisaggReplicas:
    @staticmethod
    def _launch(
        model_config: DynamoVLLMModelConfig,
        *,
        topology: list[dict[str, Any]],
        worker_index_offset: int = 0,
    ) -> None:
        """Run ``launch_disagg_replicas`` with real ``plan_replica_bundle_shape``;
        mock only the Ray PG + bundle-port plumbing. Ports are seed-stable so the
        test can assert uniqueness directly."""
        with (
            mock.patch.object(dynamo_vllm, "build_replica_pg", return_value=object()),
            mock.patch.object(
                dynamo_vllm,
                "get_free_port_in_bundle",
                side_effect=lambda _pg, _bundle, seed: 31000 + seed,
            ),
        ):
            dynamo_vllm.launch_disagg_replicas(
                model_config,
                base_env={"ETCD_ENDPOINTS": "http://10.0.0.5:2379", "NATS_SERVER": "nats://10.0.0.5:4222"},
                namespace="curator",
                request_plane="nats",
                event_plane="nats",
                runtime_dir="/tmp/rt",  # noqa: S108
                actor_name_prefix="dynamo_default_abcd1234",
                topology=topology,
                worker_index_offset=worker_index_offset,
            )

    def test_decode_and_prefill_workers_launched(self, captured_spawn: list[dict[str, Any]]) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            mode="disagg",
            prefill=DynamoRoleConfig(num_replicas=1),
            decode=DynamoRoleConfig(num_replicas=1),
        )
        self._launch(mc, topology=_SINGLE_NODE_8GPU)

        assert [c["label"] for c in captured_spawn] == [
            "Dynamo_decode_DP0_Qwen3-0.6B",
            "Dynamo_prefill_DP0_Qwen3-0.6B",
        ]

        for call in captured_spawn:
            args = call["python_args"]
            assert "--disaggregation-mode" in args
            assert args[args.index("--kv-transfer-config") + 1] == json.dumps(
                {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
            )
            # NIXL side-channel port is injected into the env for every disagg worker.
            assert "VLLM_NIXL_SIDE_CHANNEL_PORT" in call["subprocess_env"]
            assert call["subprocess_env"]["PYTHONHASHSEED"] == "0"

        # Every disagg worker always receives ``--kv-events-config`` — even
        # decode, which sets ``enable_kv_cache_events=False`` — so Dynamo's
        # args.py does not auto-bind port 20080 and cause per-node
        # collisions between decode workers.
        decode_args = captured_spawn[0]["python_args"]
        prefill_args = captured_spawn[1]["python_args"]

        assert "--kv-events-config" in prefill_args
        prefill_kv = json.loads(prefill_args[prefill_args.index("--kv-events-config") + 1])
        assert prefill_kv["enable_kv_cache_events"] is True

        assert "--kv-events-config" in decode_args
        decode_kv = json.loads(decode_args[decode_args.index("--kv-events-config") + 1])
        assert decode_kv["enable_kv_cache_events"] is False
        assert "endpoint" not in decode_kv

    def test_role_level_engine_kwargs_override_base(self, captured_spawn: list[dict[str, Any]]) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            mode="disagg",
            engine_kwargs={"max_model_len": 4096},
            prefill=DynamoRoleConfig(num_replicas=1, engine_kwargs={"tensor_parallel_size": 1}),
            decode=DynamoRoleConfig(num_replicas=1, engine_kwargs={"max_model_len": 2048}),
        )
        self._launch(mc, topology=_SINGLE_NODE_8GPU)
        decode_args = captured_spawn[0]["python_args"]
        prefill_args = captured_spawn[1]["python_args"]
        # decode override wins; prefill keeps the model-wide default.
        assert decode_args[decode_args.index("--max-model-len") + 1] == "2048"
        assert prefill_args[prefill_args.index("--max-model-len") + 1] == "4096"
        assert prefill_args[prefill_args.index("--tensor-parallel-size") + 1] == "1"

    def test_nixl_ports_are_unique_per_worker(self, captured_spawn: list[dict[str, Any]]) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            mode="disagg",
            prefill=DynamoRoleConfig(num_replicas=2),
            decode=DynamoRoleConfig(num_replicas=2),
        )
        self._launch(mc, topology=_SINGLE_NODE_8GPU)
        nixl_ports = {c["subprocess_env"]["VLLM_NIXL_SIDE_CHANNEL_PORT"] for c in captured_spawn}
        assert len(nixl_ports) == 4

    def test_worker_index_offset_isolates_port_seeds_across_models(self, captured_spawn: list[dict[str, Any]]) -> None:
        """Two disagg models launched with a threaded offset don't share port seeds.

        Without the offset, the first worker of every model lands on the
        same Nixl seed (e.g. both 20097) and same-node placement risks a
        bind race in ``get_free_port_in_bundle``. Simulates what
        ``DynamoBackend`` does in ``start()`` across multiple disagg models.
        """
        mc_a = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            mode="disagg",
            prefill=DynamoRoleConfig(num_replicas=1),
            decode=DynamoRoleConfig(num_replicas=1),
        )
        mc_b = DynamoVLLMModelConfig(
            model_identifier="meta-llama/Llama-3.2-1B",
            mode="disagg",
            prefill=DynamoRoleConfig(num_replicas=1),
            decode=DynamoRoleConfig(num_replicas=1),
        )
        self._launch(mc_a, topology=_SINGLE_NODE_8GPU)
        self._launch(mc_b, topology=_SINGLE_NODE_8GPU, worker_index_offset=len(captured_spawn))

        nixl_ports = [c["subprocess_env"]["VLLM_NIXL_SIDE_CHANNEL_PORT"] for c in captured_spawn]
        assert len(nixl_ports) == 4  # 2 per model
        assert len(set(nixl_ports)) == 4, f"expected unique ports across models, got {nixl_ports}"

    def test_custom_kv_transfer_config_overrides_default(self, captured_spawn: list[dict[str, Any]]) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            mode="disagg",
            prefill=DynamoRoleConfig(num_replicas=1),
            decode=DynamoRoleConfig(num_replicas=1),
        )
        # ``kv_transfer_config`` is ``init=False`` — Curator-managed — but
        # still patchable on the instance. This verifies custom payloads flow
        # through verbatim instead of hardcoding NixlConnector.
        mc.kv_transfer_config = {"kv_connector": "CustomConnector", "kv_role": "kv_producer"}
        self._launch(mc, topology=_SINGLE_NODE_8GPU)

        for call in captured_spawn:
            args = call["python_args"]
            assert args[args.index("--kv-transfer-config") + 1] == json.dumps(
                {"kv_connector": "CustomConnector", "kv_role": "kv_producer"}
            )
