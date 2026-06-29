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

"""Tests for ``DynamoBackend`` entry points: teardown ordering, validators,
router-mode resolution, and ``_launch_frontend`` router-flag wiring. Worker
CLI-arg construction lives in ``test_vllm.py``; pure helpers in ``test_infra.py``."""

from __future__ import annotations

import contextlib
from typing import Any
from unittest import mock

import pytest

import nemo_curator.core.serve.dynamo.backend as dynamo_backend
from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer
from nemo_curator.core.serve.dynamo.backend import DynamoBackend
from nemo_curator.core.serve.dynamo.config import DynamoRoleConfig, DynamoRouterConfig

# ---------------------------------------------------------------------------
# Backend-level validators
# ---------------------------------------------------------------------------


class TestDynamoBackendValidateGpuRequirementsDisagg:
    def test_rejects_disagg_tp_larger_than_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_curator.core.serve.dynamo.backend.check_total_gpu_capacity", lambda *_, **__: None)
        server = InferenceServer(
            models=[
                DynamoVLLMModelConfig(
                    model_identifier="Qwen/Qwen3-0.6B",
                    mode="disagg",
                    prefill=DynamoRoleConfig(num_replicas=1, engine_kwargs={"tensor_parallel_size": 4}),
                    decode=DynamoRoleConfig(num_replicas=1, engine_kwargs={"tensor_parallel_size": 1}),
                ),
            ],
            backend=DynamoServerConfig(),
        )
        topology = [{"num_gpus": 2}, {"num_gpus": 2}]
        with pytest.raises(ValueError, match="does not support multi-node TP"):
            DynamoBackend._validate_gpu_requirements(server.models, topology=topology)

    def test_accepts_disagg_within_node_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: list[int] = []
        monkeypatch.setattr(
            "nemo_curator.core.serve.dynamo.backend.check_total_gpu_capacity",
            lambda n, **__: recorded.append(n),
        )
        server = InferenceServer(
            models=[
                DynamoVLLMModelConfig(
                    model_identifier="Qwen/Qwen3-0.6B",
                    mode="disagg",
                    prefill=DynamoRoleConfig(num_replicas=2, engine_kwargs={"tensor_parallel_size": 2}),
                    decode=DynamoRoleConfig(num_replicas=1, engine_kwargs={"tensor_parallel_size": 2}),
                ),
            ],
            backend=DynamoServerConfig(),
        )
        topology = [{"num_gpus": 4}, {"num_gpus": 4}]
        DynamoBackend._validate_gpu_requirements(server.models, topology=topology)
        # 2 prefill workers * TP=2 + 1 decode worker * TP=2 = 6 GPUs.
        assert recorded == [6]


class TestDynamoBackendValidateUniqueModelNames:
    def test_accepts_distinct_names(self) -> None:
        models = [
            DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B"),
            DynamoVLLMModelConfig(model_identifier="meta-llama/Llama-3.1-8B"),
        ]
        DynamoBackend._validate_unique_model_names(models)

    def test_rejects_duplicate_names(self) -> None:
        models = [
            DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B"),
            DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B"),
        ]
        with pytest.raises(ValueError, match="Duplicate model name"):
            DynamoBackend._validate_unique_model_names(models)

    def test_rejects_component_slug_collision(self) -> None:
        # 'Qwen.3' and 'Qwen-3' both sanitize to 'qwen_3'
        models = [
            DynamoVLLMModelConfig(model_identifier="org/m1", model_name="Qwen.3"),
            DynamoVLLMModelConfig(model_identifier="org/m2", model_name="Qwen-3"),
        ]
        with pytest.raises(ValueError, match="sanitize to component"):
            DynamoBackend._validate_unique_model_names(models)


class TestResolveEffectiveRouter:
    @staticmethod
    def _disagg_models() -> list[DynamoVLLMModelConfig]:
        return [
            DynamoVLLMModelConfig(
                model_identifier="m",
                mode="disagg",
                prefill=DynamoRoleConfig(num_replicas=1),
                decode=DynamoRoleConfig(num_replicas=1),
            ),
        ]

    def test_explicit_mode_is_honored(self) -> None:
        # Even with a disagg model present, explicit router.mode wins and
        # router.kv_events is passed through untouched (no auto-enable).
        router = DynamoRouterConfig(mode="round_robin")
        mode, kv_events = DynamoBackend._resolve_effective_router(self._disagg_models(), router)
        assert mode == "round_robin"
        assert kv_events is False

    def test_auto_picks_kv_and_auto_enables_kv_events_for_disagg(self) -> None:
        # router.mode=None + any disagg model → auto-pick "kv" AND
        # auto-enable kv_events so the frontend consumes what prefill
        # workers publish.
        router = DynamoRouterConfig()  # mode=None, kv_events=False defaults
        mode, kv_events = DynamoBackend._resolve_effective_router(self._disagg_models(), router)
        assert mode == "kv"
        assert kv_events is True

    def test_auto_disagg_uses_approximate_kv_routing_when_hma_is_explicit(self) -> None:
        models = [
            DynamoVLLMModelConfig(
                model_identifier="m",
                mode="disagg",
                engine_kwargs={"disable_hybrid_kv_cache_manager": False},
                prefill=DynamoRoleConfig(num_replicas=1),
                decode=DynamoRoleConfig(num_replicas=1),
            ),
        ]
        router = DynamoRouterConfig()

        mode, kv_events = DynamoBackend._resolve_effective_router(models, router)

        assert mode == "kv"
        assert kv_events is False

    def test_auto_disagg_with_explicit_kv_events_rejects_hma_publishers(self) -> None:
        models = [
            DynamoVLLMModelConfig(
                model_identifier="m",
                mode="disagg",
                engine_kwargs={"disable_hybrid_kv_cache_manager": False},
                prefill=DynamoRoleConfig(num_replicas=1),
                decode=DynamoRoleConfig(num_replicas=1),
            ),
        ]
        router = DynamoRouterConfig(kv_events=True)

        with pytest.raises(ValueError, match="Hybrid KV cache manager"):
            DynamoBackend._resolve_effective_router(models, router)

    def test_explicit_event_backed_kv_routing_rejects_hma_publishers(self) -> None:
        models = [
            DynamoVLLMModelConfig(
                model_identifier="m",
                engine_kwargs={"disable_hybrid_kv_cache_manager": False},
            ),
        ]
        router = DynamoRouterConfig(mode="kv", kv_events=True)

        with pytest.raises(ValueError, match="Hybrid KV cache manager"):
            DynamoBackend._resolve_effective_router(models, router)

    def test_explicit_kv_with_kv_events_false_is_honored(self) -> None:
        # User set mode="kv" themselves and left kv_events=False (approx
        # tree-based routing). We must NOT auto-flip kv_events just because
        # a disagg model is also present — respect the explicit choice.
        router = DynamoRouterConfig(mode="kv", kv_events=False)
        mode, kv_events = DynamoBackend._resolve_effective_router(self._disagg_models(), router)
        assert mode == "kv"
        assert kv_events is False

    def test_aggregated_only_leaves_mode_unset(self) -> None:
        models = [DynamoVLLMModelConfig(model_identifier="m")]
        router = DynamoRouterConfig()
        mode, kv_events = DynamoBackend._resolve_effective_router(models, router)
        assert mode is None
        assert kv_events is False

    def test_aggregated_without_kv_mode_ignores_kv_events_for_hma_guard(self) -> None:
        models = [
            DynamoVLLMModelConfig(
                model_identifier="m",
                engine_kwargs={"disable_hybrid_kv_cache_manager": False},
            )
        ]
        router = DynamoRouterConfig(kv_events=True)
        mode, kv_events = DynamoBackend._resolve_effective_router(models, router)
        assert mode is None
        assert kv_events is False


# ---------------------------------------------------------------------------
# Backend ``start()`` teardown ordering (locked in by eb6416d8 — the
# killpg-only subprocess_mgr rewrite depends on this order).
# ---------------------------------------------------------------------------


class TestDynamoBackendStart:
    def test_sweeps_orphan_actors_before_removing_placement_groups(self) -> None:
        """``remove_named_pgs_with_prefix`` force-kills actors scheduled into
        the reaped PGs; sweeping named actors first lets ``graceful_stop_actors``
        ``killpg`` each process group cleanly. Also asserts the override
        fan-out runs *before* any worker-spawning step — the actor venv
        install on each worker reads the file from the local filesystem,
        so it must exist by the time the first actor lands."""
        server = InferenceServer(
            models=[DynamoVLLMModelConfig(model_identifier="m")],
            backend=DynamoServerConfig(
                etcd_endpoint="http://127.0.0.1:2379",
                nats_url="nats://127.0.0.1:4222",
            ),
        )
        backend = DynamoBackend(server)
        order: list[str] = []

        mock_ctx = mock.Mock()
        mock_ctx.get_temp_dir.return_value = "/tmp"  # noqa: S108
        mock_ctx.get_session_name.return_value = "session_test"

        with (
            mock.patch.object(dynamo_backend.ray, "init", return_value=contextlib.nullcontext()),
            mock.patch.object(dynamo_backend.ray, "get_runtime_context", return_value=mock_ctx),
            mock.patch.object(dynamo_backend.os, "makedirs"),
            mock.patch.object(
                dynamo_backend,
                "ensure_actor_overrides_on_all_nodes",
                side_effect=lambda **_kw: order.append("overrides"),
            ),
            mock.patch.object(backend, "_sweep_orphan_actors", side_effect=lambda: order.append("actors")),
            mock.patch.object(
                dynamo_backend,
                "remove_named_pgs_with_prefix",
                side_effect=lambda _prefix: order.append("pgs"),
            ),
            mock.patch.object(backend, "_deploy_and_healthcheck", side_effect=lambda *_a: order.append("deploy")),
        ):
            backend.start()

        assert order == ["overrides", "actors", "pgs", "deploy"]


class TestDynamoBackendDeploy:
    def test_subprocess_env_is_merged_into_deploy_base_env(self) -> None:
        backend_cfg = DynamoServerConfig(
            etcd_endpoint="http://system-etcd:2379",
            nats_url="nats://system-nats:4222",
            subprocess_env={
                "DYN_TCP_REQUEST_TIMEOUT": "180",
                "ETCD_ENDPOINTS": "user-etcd-should-not-win",
                "NATS_SERVER": "user-nats-should-not-win",
            },
        )
        server = InferenceServer(
            models=[DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B")],
            backend=backend_cfg,
        )
        backend = DynamoBackend(server)
        backend._runtime_dir = "/tmp/rt"  # noqa: S108
        backend._actor_name_prefix = "prefix"

        captured: dict[str, dict[str, str]] = {}

        def fake_launch_replicas(  # noqa: PLR0913
            _model_config: DynamoVLLMModelConfig,
            *,
            base_env: dict[str, str],
            namespace: str,  # noqa: ARG001
            request_plane: str,  # noqa: ARG001
            event_plane: str,  # noqa: ARG001
            runtime_dir: str,  # noqa: ARG001
            actor_name_prefix: str,  # noqa: ARG001
            router_mode: str | None,  # noqa: ARG001
            router_kv_events: bool,  # noqa: ARG001
            topology: list[dict[str, object]],  # noqa: ARG001
        ) -> tuple[list[object], list[object], list[dict[str, object]]]:
            captured["worker"] = base_env
            return [], [], []

        def fake_launch_frontend(
            _port: int,
            base_env: dict[str, str],
            *,
            backend_cfg: DynamoServerConfig,  # noqa: ARG001
            effective_router_mode: str | None = None,  # noqa: ARG001
            effective_router_kv_events: bool = False,  # noqa: ARG001
            runtime_env: dict[str, object] | None = None,  # noqa: ARG001
        ) -> object:
            captured["frontend"] = base_env
            return object()

        with (
            mock.patch.object(backend, "_validate_unique_model_names"),
            mock.patch.object(backend, "_validate_gpu_requirements"),
            mock.patch.object(dynamo_backend, "_get_gpu_topology", return_value=[{"num_gpus": 1}]),
            mock.patch.object(dynamo_backend, "build_infra_pg", return_value=object()),
            mock.patch.object(dynamo_backend, "get_bundle_node_ip", return_value="10.0.0.5"),
            mock.patch.object(dynamo_backend, "get_free_port_in_bundle", return_value=9999),
            mock.patch.object(dynamo_backend, "launch_replicas", side_effect=fake_launch_replicas),
            mock.patch.object(backend, "_launch_frontend", side_effect=fake_launch_frontend),
            mock.patch.object(backend, "_wait_for_models"),
            mock.patch.object(backend, "_write_manifest"),
        ):
            backend._deploy_and_healthcheck(server, backend_cfg)

        expected = {
            "DYN_TCP_REQUEST_TIMEOUT": "180",
            "ETCD_ENDPOINTS": "http://system-etcd:2379",
            "NATS_SERVER": "nats://system-nats:4222",
        }
        assert captured == {"worker": expected, "frontend": expected}


# ---------------------------------------------------------------------------
# Frontend CLI-arg wiring (router mode + router_kwargs translation)
# ---------------------------------------------------------------------------


class TestDynamoBackendLaunchFrontend:
    @staticmethod
    def _make_backend(backend_cfg: DynamoServerConfig) -> DynamoBackend:
        server = InferenceServer(
            models=[DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B")],
            backend=backend_cfg,
        )
        backend = DynamoBackend(server)
        backend._runtime_dir = "/tmp/rt"  # noqa: S108
        backend._actor_name_prefix = "prefix"
        backend._infra_pg = object()
        return backend

    @staticmethod
    def _make_multimodal_backend(backend_cfg: DynamoServerConfig) -> DynamoBackend:
        server = InferenceServer(
            models=[
                DynamoVLLMModelConfig(
                    model_identifier="nvidia/NVIDIA-Nemotron-Parse-v1.2",
                    dynamo_kwargs={"enable_multimodal": True},
                )
            ],
            backend=backend_cfg,
        )
        backend = DynamoBackend(server)
        backend._runtime_dir = "/tmp/rt"  # noqa: S108
        backend._actor_name_prefix = "prefix"
        backend._infra_pg = object()
        return backend

    def test_multimodal_without_explicit_processor_uses_native(self, captured_spawn: list[dict[str, Any]]) -> None:
        """router_kwargs are passed through verbatim — no auto-injection for multimodal models."""
        backend_cfg = DynamoServerConfig()
        backend = self._make_multimodal_backend(backend_cfg)

        backend._launch_frontend(port=9999, base_env={}, backend_cfg=backend_cfg)

        assert "--dyn-chat-processor" not in captured_spawn[0]["python_args"]

    def test_explicit_vllm_chat_processor_is_forwarded(self, captured_spawn: list[dict[str, Any]]) -> None:
        """When dyn_chat_processor is set explicitly it flows through as a CLI flag."""
        backend_cfg = DynamoServerConfig(
            router=DynamoRouterConfig(
                router_kwargs={
                    "dyn_chat_processor": "vllm",
                }
            ),
        )
        backend = self._make_multimodal_backend(backend_cfg)

        backend._launch_frontend(port=9999, base_env={}, backend_cfg=backend_cfg)

        python_args = captured_spawn[0]["python_args"]
        assert python_args[python_args.index("--dyn-chat-processor") + 1] == "vllm"

    def test_supported_false_router_kwargs_emit_no_flags(self, captured_spawn: list[dict[str, Any]]) -> None:
        backend_cfg = DynamoServerConfig(
            router=DynamoRouterConfig(router_kwargs={"exclude_tools_when_tool_choice_none": False}),
        )
        backend = self._make_backend(backend_cfg)

        backend._launch_frontend(port=9999, base_env={}, backend_cfg=backend_cfg)

        python_args = captured_spawn[0]["python_args"]
        assert "--no-exclude-tools-when-tool-choice-none" in python_args

    def test_text_only_model_keeps_native_chat_processor(self, captured_spawn: list[dict[str, Any]]) -> None:
        backend_cfg = DynamoServerConfig()
        backend = self._make_backend(backend_cfg)

        backend._launch_frontend(port=9999, base_env={}, backend_cfg=backend_cfg)

        assert "--dyn-chat-processor" not in captured_spawn[0]["python_args"]

    def test_router_flags_and_router_kwargs_passthrough(self, captured_spawn: list[dict[str, Any]]) -> None:
        """``PYTHONHASHSEED=0`` is pinned when ``router-mode`` is set: Dynamo KV
        routing relies on a stable prefix-hash across frontend + worker processes."""
        backend_cfg = DynamoServerConfig(
            router=DynamoRouterConfig(
                mode="kv",
                router_kwargs={"router_temperature": 0.1, "router_ttl_secs": 60},
            ),
        )
        backend = self._make_backend(backend_cfg)

        backend._launch_frontend(port=9999, base_env={"ETCD_ENDPOINTS": "e"}, backend_cfg=backend_cfg)

        assert captured_spawn[0]["python_args"] == [
            "-m",
            "dynamo.frontend",
            "--http-port",
            "9999",
            "--namespace",
            "curator",
            "--discovery-backend",
            "etcd",
            "--request-plane",
            "nats",
            "--event-plane",
            "nats",
            "--router-mode",
            "kv",
            "--no-router-kv-events",
            "--router-temperature",
            "0.1",
            "--router-ttl-secs",
            "60",
        ]
        assert captured_spawn[0]["subprocess_env"] == {"ETCD_ENDPOINTS": "e", "PYTHONHASHSEED": "0"}

    def test_no_router_mode_omits_flag_and_hashseed(self, captured_spawn: list[dict[str, Any]]) -> None:
        backend_cfg = DynamoServerConfig()
        backend = self._make_backend(backend_cfg)

        backend._launch_frontend(port=9999, base_env={}, backend_cfg=backend_cfg)

        assert captured_spawn[0]["python_args"] == [
            "-m",
            "dynamo.frontend",
            "--http-port",
            "9999",
            "--namespace",
            "curator",
            "--discovery-backend",
            "etcd",
            "--request-plane",
            "nats",
            "--event-plane",
            "nats",
        ]
        assert captured_spawn[0]["subprocess_env"] == {}

    def test_tcp_request_plane_and_timeout_env_passthrough(self, captured_spawn: list[dict[str, Any]]) -> None:
        backend_cfg = DynamoServerConfig(request_plane="tcp")
        backend = self._make_backend(backend_cfg)

        backend._launch_frontend(
            port=9999,
            base_env={"ETCD_ENDPOINTS": "e", "DYN_TCP_REQUEST_TIMEOUT": "180"},
            backend_cfg=backend_cfg,
        )

        python_args = captured_spawn[0]["python_args"]
        assert python_args[python_args.index("--request-plane") + 1] == "tcp"
        assert captured_spawn[0]["subprocess_env"] == {
            "ETCD_ENDPOINTS": "e",
            "DYN_TCP_REQUEST_TIMEOUT": "180",
        }

    def test_kv_mode_without_events_emits_no_router_kv_events(self, captured_spawn: list[dict[str, Any]]) -> None:
        backend_cfg = DynamoServerConfig(router=DynamoRouterConfig(mode="kv", kv_events=False))
        backend = self._make_backend(backend_cfg)

        backend._launch_frontend(port=9999, base_env={}, backend_cfg=backend_cfg)

        python_args = captured_spawn[0]["python_args"]
        assert "--no-router-kv-events" in python_args
        assert "--router-kv-events" not in python_args

    def test_kv_mode_with_events_emits_router_kv_events(self, captured_spawn: list[dict[str, Any]]) -> None:
        backend_cfg = DynamoServerConfig(router=DynamoRouterConfig(mode="kv", kv_events=True))
        backend = self._make_backend(backend_cfg)

        backend._launch_frontend(port=9999, base_env={}, backend_cfg=backend_cfg)

        python_args = captured_spawn[0]["python_args"]
        assert "--router-kv-events" in python_args
        assert "--no-router-kv-events" not in python_args

    def test_round_robin_mode_omits_kv_events_flag(self, captured_spawn: list[dict[str, Any]]) -> None:
        backend_cfg = DynamoServerConfig(router=DynamoRouterConfig(mode="round_robin"))
        backend = self._make_backend(backend_cfg)

        backend._launch_frontend(port=9999, base_env={}, backend_cfg=backend_cfg)

        python_args = captured_spawn[0]["python_args"]
        assert "--router-mode" in python_args
        assert "--router-kv-events" not in python_args
        assert "--no-router-kv-events" not in python_args

    def test_effective_router_overrides_are_honored(self, captured_spawn: list[dict[str, Any]]) -> None:
        """Auto-resolve path: ``_deploy_and_healthcheck`` passes both resolved
        values in. The CLI must use them, regardless of what ``router.mode``
        / ``router.kv_events`` were in the config (typical case: config says
        ``mode=None, kv_events=False`` defaults and the resolver picked
        ``"kv"`` + ``True`` because a disagg model is present).
        """
        backend_cfg = DynamoServerConfig(router=DynamoRouterConfig(mode=None, kv_events=False))
        backend = self._make_backend(backend_cfg)

        backend._launch_frontend(
            port=9999,
            base_env={},
            backend_cfg=backend_cfg,
            effective_router_mode="kv",
            effective_router_kv_events=True,
        )

        python_args = captured_spawn[0]["python_args"]
        assert python_args[python_args.index("--router-mode") + 1] == "kv"
        assert "--router-kv-events" in python_args
        assert "--no-router-kv-events" not in python_args
