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

import os
from pathlib import Path

import pytest

import nemo_curator.core.utils as core_utils
from nemo_curator.core.utils import ignore_ray_head_node

_ALL_INTERFACES = "0.0.0.0"  # noqa: S104


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> None:
    """Override the repository-wide Ray fixture for these pure utility tests."""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
        *[(v, True) for v in ("1", "true", "TRUE", "yes", " 1 ")],
    ],
)
def test_ignore_ray_head_node_env_parsing(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool) -> None:
    if value is None:
        monkeypatch.delenv("CURATOR_IGNORE_RAY_HEAD_NODE", raising=False)
    else:
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", value)
    assert ignore_ray_head_node() is expected


def _patch_cluster_start_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[list[str]], list[tuple[int, bool, str]]]:
    popen_commands: list[list[str]] = []
    free_port_calls: list[tuple[int, bool, str]] = []
    free_ports = {
        core_utils.DEFAULT_RAY_DASHBOARD_METRIC_PORT: 8266,
        core_utils.DEFAULT_RAY_AUTOSCALER_METRIC_PORT: 8267,
        core_utils.DEFAULT_RAY_SERVE_HAPROXY_METRICS_PORT: 9111,
        core_utils.DEFAULT_RAY_SERVE_HAPROXY_STATS_PORT: 8414,
    }

    def fake_popen(command: list[str], **_: object) -> object:
        popen_commands.append(command)
        return object()

    def fake_get_free_port(start_port: int, get_next_free_port: bool = True, bind_host: str = "localhost") -> int:
        free_port_calls.append((start_port, get_next_free_port, bind_host))
        return free_ports[start_port]

    monkeypatch.setattr(core_utils.ray.util, "register_serializer", lambda *_, **__: None)
    monkeypatch.setattr(core_utils.socket, "gethostbyname", lambda _hostname: "127.0.0.1")
    monkeypatch.setattr(core_utils.socket, "gethostname", lambda: "localhost")
    monkeypatch.setattr(core_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(core_utils, "get_free_port", fake_get_free_port)
    return popen_commands, free_port_calls


def test_ray_serve_haproxy_source_prefers_packaged_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAY_SERVE_HAPROXY_BINARY_PATH", raising=False)
    monkeypatch.setattr(
        core_utils.importlib.util, "find_spec", lambda name: object() if name == "ray_haproxy" else None
    )

    assert core_utils._ray_serve_haproxy_source() == "ray-haproxy package"


def test_init_cluster_enables_ray_serve_haproxy_env_vars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAY_SERVE_HAPROXY_BINARY_PATH", "/opt/ray-haproxy")
    popen_calls, free_port_calls = _patch_cluster_start_side_effects(monkeypatch)

    core_utils.init_cluster(
        ray_port=6379,
        ray_temp_dir=str(tmp_path),
        ray_dashboard_port=8265,
        ray_metrics_port=8080,
        ray_client_server_port=10001,
        ray_dashboard_host="127.0.0.1",
    )

    assert popen_calls
    assert (core_utils.DEFAULT_RAY_SERVE_HAPROXY_METRICS_PORT, True, _ALL_INTERFACES) in free_port_calls
    assert (core_utils.DEFAULT_RAY_SERVE_HAPROXY_STATS_PORT, True, _ALL_INTERFACES) in free_port_calls
    assert os.environ["RAY_SERVE_ENABLE_HA_PROXY"] == "1"
    assert os.environ["RAY_SERVE_EXPERIMENTAL_PIP_HAPROXY"] == "1"
    assert os.environ["RAY_SERVE_HAPROXY_METRICS_PORT"] == "9111"
    assert os.environ["RAY_SERVE_HAPROXY_STATS_PORT"] == "8414"
