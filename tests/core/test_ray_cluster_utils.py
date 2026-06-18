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

from typing import TYPE_CHECKING

from nemo_curator.core import utils as core_utils

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _mock_init_cluster_dependencies(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    popen_calls: list[list[str]] = []

    class FakePopen:
        def __init__(self, cmd: list[str], **_kwargs: object) -> None:
            popen_calls.append(cmd)

    monkeypatch.setattr(core_utils.ray.util, "register_serializer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(core_utils, "get_free_port", lambda port, **_kwargs: port)
    monkeypatch.setattr(core_utils.shutil, "which", lambda _cmd: None)
    monkeypatch.setattr(core_utils.subprocess, "Popen", FakePopen)
    return popen_calls


def test_init_cluster_sets_xenna_gpu_env_before_ray_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    popen_calls = _mock_init_cluster_dependencies(monkeypatch)
    monkeypatch.delenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", raising=False)

    core_utils.init_cluster(
        ray_port=6379,
        ray_temp_dir=str(tmp_path),
        ray_dashboard_port=8265,
        ray_metrics_port=8080,
        ray_client_server_port=10001,
    )

    assert popen_calls
    assert "--head" in popen_calls[0]
    assert "--block" in popen_calls[0]
    assert core_utils.os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"


def test_init_cluster_preserves_existing_xenna_gpu_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_init_cluster_dependencies(monkeypatch)
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "0")

    core_utils.init_cluster(
        ray_port=6379,
        ray_temp_dir=str(tmp_path),
        ray_dashboard_port=8265,
        ray_metrics_port=8080,
        ray_client_server_port=10001,
    )

    assert core_utils.os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "0"
