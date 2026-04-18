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

import pytest

from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer
from nemo_curator.core.serve.dynamo import vllm as dynamo_vllm
from nemo_curator.core.serve.dynamo.backend import DynamoBackend
from nemo_curator.core.serve.placement import ReplicaBundleSpec
from nemo_curator.core.serve.subprocess_mgr import ManagedSubprocess

# ---------------------------------------------------------------------------
# Pure helpers in vllm.py
# ---------------------------------------------------------------------------


class TestModelNameToComponent:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Qwen3-0.6B", "qwen3_0_6b"),
            ("Qwen/Qwen3-0.6B", "qwen_qwen3_0_6b"),
            ("meta-llama/Llama-3.1-8B", "meta_llama_llama_3_1_8b"),
            ("simple", "simple"),
        ],
    )
    def test_sanitizes(self, name: str, expected: str) -> None:
        assert dynamo_vllm.model_name_to_component(name) == expected

    def test_rejects_empty_slug(self) -> None:
        with pytest.raises(ValueError, match="empty component slug"):
            dynamo_vllm.model_name_to_component("---")


class TestDynamoEndpoint:
    def test_without_role(self) -> None:
        assert dynamo_vllm.dynamo_endpoint("curator", "qwen") == "dyn://curator.qwen.generate"

    def test_with_role(self) -> None:
        assert dynamo_vllm.dynamo_endpoint("curator", "qwen", role="decode") == "dyn://curator.qwen_decode.generate"


class TestAggregatedModelUsesExactKvEvents:
    def test_non_kv_router_returns_false(self) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="m")
        assert not dynamo_vllm.aggregated_model_uses_exact_kv_events(
            mc, router_mode="round_robin", router_kv_events=True
        )

    def test_kv_router_with_events_returns_true(self) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="m")
        assert dynamo_vllm.aggregated_model_uses_exact_kv_events(mc, router_mode="kv", router_kv_events=True)

    def test_kv_router_without_events_returns_false(self) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="m")
        assert not dynamo_vllm.aggregated_model_uses_exact_kv_events(mc, router_mode="kv", router_kv_events=False)


# ---------------------------------------------------------------------------
# Worker launch args
# ---------------------------------------------------------------------------


def _capture_spawn(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture ``ManagedSubprocess.spawn`` calls without launching Ray actors."""
    calls: list[dict[str, Any]] = []

    @classmethod
    def fake_spawn(_cls, label, pg, bundle_index, **kwargs) -> ManagedSubprocess:  # noqa: ANN001
        calls.append({"label": label, "pg": pg, "bundle_index": bundle_index, **kwargs})
        return ManagedSubprocess(label=label, actor=object())

    monkeypatch.setattr(ManagedSubprocess, "spawn", fake_spawn)
    return calls


def _single_node_spec(num_gpus: int = 1) -> ReplicaBundleSpec:
    return ReplicaBundleSpec(
        bundles=[{"CPU": 1, "GPU": num_gpus}],
        strategy="STRICT_PACK",
        nnodes=1,
        per_node_gpus=num_gpus,
    )


def _multi_node_spec(nnodes: int, per_node_gpus: int) -> ReplicaBundleSpec:
    return ReplicaBundleSpec(
        bundles=[{"CPU": 1, "GPU": per_node_gpus}] * nnodes,
        strategy="STRICT_SPREAD",
        nnodes=nnodes,
        per_node_gpus=per_node_gpus,
    )


class TestAggregatedWorkerLaunchArgs:
    """Verify the Python args passed to ``python -m dynamo.vllm`` for aggregated workers."""

    def _launch_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_config: DynamoVLLMModelConfig,
        *,
        spec: ReplicaBundleSpec,
        router_mode: str | None = None,
        router_kv_events: bool = False,
    ) -> list[dict[str, Any]]:
        calls = _capture_spawn(monkeypatch)
        monkeypatch.setattr(
            dynamo_vllm,
            "plan_replica_bundle_shape",
            lambda _tp, _topology=None: spec,
        )
        monkeypatch.setattr(dynamo_vllm, "build_replica_pg", lambda _spec, **_kw: object())
        monkeypatch.setattr(dynamo_vllm, "get_bundle_node_ip", lambda _pg, _bundle: "10.0.0.5")
        monkeypatch.setattr(
            dynamo_vllm,
            "get_free_port_in_bundle",
            lambda _pg, _bundle, _seed: 24567,
        )

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
            topology=None,
        )
        return calls

    def test_single_node_disables_kv_events_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B", num_replicas=1)
        calls = self._launch_one(monkeypatch, mc, spec=_single_node_spec(1))
        assert len(calls) == 1
        python_args = calls[0]["python_args"]
        idx = python_args.index("--kv-events-config")
        assert json.loads(python_args[idx + 1]) == {"enable_kv_cache_events": False}
        assert "--headless" not in python_args
        assert "--nnodes" not in python_args

    def test_kv_router_enables_exact_kv_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B", num_replicas=1)
        calls = self._launch_one(monkeypatch, mc, spec=_single_node_spec(1), router_mode="kv", router_kv_events=True)
        python_args = calls[0]["python_args"]
        cfg = json.loads(python_args[python_args.index("--kv-events-config") + 1])
        assert cfg["enable_kv_cache_events"] is True
        assert cfg["endpoint"] == "tcp://*:24567"
        assert cfg["publisher"] == "zmq"
        assert cfg["topic"] == "kv-events"

    def test_multi_node_rank0_adds_nnodes_and_master(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            engine_kwargs={"tensor_parallel_size": 8},
            num_replicas=1,
        )
        calls = self._launch_one(monkeypatch, mc, spec=_multi_node_spec(2, 4))
        assert len(calls) == 2

        rank0 = calls[0]["python_args"]
        assert rank0[rank0.index("--nnodes") + 1] == "2"
        assert rank0[rank0.index("--node-rank") + 1] == "0"
        assert rank0[rank0.index("--master-addr") + 1] == "10.0.0.5"
        assert "--headless" not in rank0

        headless = calls[1]["python_args"]
        assert "--headless" in headless
        assert headless[headless.index("--node-rank") + 1] == "1"
        assert headless[headless.index("--master-addr") + 1] == "10.0.0.5"
        cfg = json.loads(headless[headless.index("--kv-events-config") + 1])
        assert cfg["enable_kv_cache_events"] is False

    def test_dynamo_kwargs_are_appended_as_cli_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            dynamo_kwargs={"tool_call_parser": "hermes", "reasoning_parser": "deepseek-r1"},
            num_replicas=1,
        )
        calls = self._launch_one(monkeypatch, mc, spec=_single_node_spec(1))
        python_args = calls[0]["python_args"]
        assert python_args[python_args.index("--tool-call-parser") + 1] == "hermes"
        assert python_args[python_args.index("--reasoning-parser") + 1] == "deepseek-r1"

    def test_multi_replica_spawns_n_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B", num_replicas=3)
        calls = self._launch_one(monkeypatch, mc, spec=_single_node_spec(1))
        assert len(calls) == 3
        labels = [c["label"] for c in calls]
        assert labels == ["Dynamo_DP0_Qwen3-0.6B", "Dynamo_DP1_Qwen3-0.6B", "Dynamo_DP2_Qwen3-0.6B"]


# ---------------------------------------------------------------------------
# Backend-level behavior
# ---------------------------------------------------------------------------


class TestDynamoBackendStartRejectsDisagg:
    def test_start_raises_notimplemented_for_disagg_mode(self) -> None:
        from nemo_curator.core.serve.dynamo.config import DynamoRoleConfig

        server = InferenceServer(
            models=[
                DynamoVLLMModelConfig(
                    model_identifier="Qwen/Qwen3-0.6B",
                    mode="disagg",
                    prefill=DynamoRoleConfig(num_replicas=1),
                    decode=DynamoRoleConfig(num_replicas=1),
                ),
            ],
            backend=DynamoServerConfig(),
        )
        backend = DynamoBackend(server)

        with pytest.raises(NotImplementedError, match="Disaggregated serving"):
            backend.start()


class TestDynamoBackendValidateGpuRequirements:
    def test_rejects_overcommit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_check(gpus_needed: int, *, ignore_head_node: bool = False) -> None:  # noqa: ARG001
            if gpus_needed > 4:
                msg = f"Not enough GPUs: need {gpus_needed}, have 4"
                raise RuntimeError(msg)

        monkeypatch.setattr("nemo_curator.core.serve.dynamo.backend.check_total_gpu_capacity", fake_check)

        server = InferenceServer(
            models=[
                DynamoVLLMModelConfig(
                    model_identifier="Qwen/Qwen3-0.6B",
                    engine_kwargs={"tensor_parallel_size": 4},
                    num_replicas=2,
                ),
            ],
            backend=DynamoServerConfig(),
        )
        with pytest.raises(RuntimeError, match="Not enough GPUs"):
            DynamoBackend._validate_gpu_requirements(server.models)

    def test_accepts_within_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: list[int] = []

        def fake_check(gpus_needed: int, *, ignore_head_node: bool = False) -> None:  # noqa: ARG001
            recorded.append(gpus_needed)

        monkeypatch.setattr("nemo_curator.core.serve.dynamo.backend.check_total_gpu_capacity", fake_check)

        server = InferenceServer(
            models=[
                DynamoVLLMModelConfig(
                    model_identifier="Qwen/Qwen3-0.6B",
                    engine_kwargs={"tensor_parallel_size": 2},
                    num_replicas=2,
                ),
            ],
            backend=DynamoServerConfig(),
        )
        DynamoBackend._validate_gpu_requirements(server.models)
        assert recorded == [4]


class TestDynamoBackendLaunchFrontend:
    def test_router_flags_and_router_kwargs_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_curator.core.serve.dynamo.config import DynamoRouterConfig

        calls = _capture_spawn(monkeypatch)

        backend_cfg = DynamoServerConfig(
            namespace="curator",
            request_plane="nats",
            event_plane="nats",
            router=DynamoRouterConfig(
                mode="kv",
                router_kwargs={"router_temperature": 0.1, "router_ttl_secs": 60},
            ),
        )
        server = InferenceServer(
            models=[DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B")],
            backend=backend_cfg,
        )
        backend = DynamoBackend(server)
        backend._runtime_dir = "/tmp/rt"  # noqa: S108
        backend._actor_name_prefix = "prefix"
        backend._infra_pg = object()

        backend._launch_frontend(port=9999, base_env={"ETCD_ENDPOINTS": "e"}, backend_cfg=backend_cfg)

        python_args = calls[0]["python_args"]
        assert python_args[:2] == ["-m", "dynamo.frontend"]
        assert python_args[python_args.index("--http-port") + 1] == "9999"
        assert python_args[python_args.index("--namespace") + 1] == "curator"
        assert python_args[python_args.index("--router-mode") + 1] == "kv"
        assert python_args[python_args.index("--router-temperature") + 1] == "0.1"
        assert python_args[python_args.index("--router-ttl-secs") + 1] == "60"
        # PYTHONHASHSEED must be pinned when router-mode is set (Dynamo requirement).
        assert calls[0]["subprocess_env"]["PYTHONHASHSEED"] == "0"

    def test_no_router_mode_omits_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _capture_spawn(monkeypatch)

        backend_cfg = DynamoServerConfig()
        server = InferenceServer(
            models=[DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B")],
            backend=backend_cfg,
        )
        backend = DynamoBackend(server)
        backend._runtime_dir = "/tmp/rt"  # noqa: S108
        backend._actor_name_prefix = "prefix"
        backend._infra_pg = object()

        backend._launch_frontend(port=9999, base_env={}, backend_cfg=backend_cfg)

        python_args = calls[0]["python_args"]
        assert "--router-mode" not in python_args
        assert "PYTHONHASHSEED" not in calls[0]["subprocess_env"]
