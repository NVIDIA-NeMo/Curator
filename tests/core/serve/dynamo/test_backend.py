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

"""Tests for ``DynamoBackend`` entry points: disagg rejection, ``start()``
teardown ordering, and ``_launch_frontend`` router-flag wiring."""

from __future__ import annotations

import contextlib
from typing import Any
from unittest import mock

import pytest

import nemo_curator.core.serve.dynamo.backend as dynamo_backend
from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer
from nemo_curator.core.serve.dynamo.backend import DynamoBackend
from nemo_curator.core.serve.dynamo.config import DynamoRoleConfig, DynamoRouterConfig


class TestDynamoBackendStart:
    def test_rejects_disagg_mode(self) -> None:
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
        with pytest.raises(NotImplementedError, match="Disaggregated serving"):
            DynamoBackend(server).start()

    def test_sweeps_orphan_actors_before_removing_placement_groups(self) -> None:
        """``remove_named_pgs_with_prefix`` force-kills actors scheduled into
        the reaped PGs; sweeping named actors first lets ``graceful_stop_actors``
        ``killpg`` each process group cleanly."""
        server = InferenceServer(
            models=[DynamoVLLMModelConfig(model_identifier="m")],
            backend=DynamoServerConfig(
                etcd_endpoint="http://127.0.0.1:2379",
                nats_url="nats://127.0.0.1:4222",
            ),
        )
        backend = DynamoBackend(server)
        order: list[str] = []

        with (
            mock.patch.object(dynamo_backend.ray, "init", return_value=contextlib.nullcontext()),
            mock.patch.object(dynamo_backend.tempfile, "mkdtemp", return_value="/tmp/dynamo-test-runtime"),  # noqa: S108
            mock.patch.object(backend, "_sweep_orphan_actors", side_effect=lambda: order.append("actors")),
            mock.patch.object(
                dynamo_backend,
                "remove_named_pgs_with_prefix",
                side_effect=lambda _prefix: order.append("pgs"),
            ),
            mock.patch.object(backend, "_deploy_and_healthcheck", side_effect=lambda *_a: order.append("deploy")),
        ):
            backend.start()

        assert order == ["actors", "pgs", "deploy"]


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
