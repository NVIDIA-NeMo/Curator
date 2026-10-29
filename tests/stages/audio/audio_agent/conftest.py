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

"""Isolated agent unit tests and explicitly requested Ray integration tests."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import pytest
import ray

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> None:
    """Agent unit tests do not require the repository-wide cluster fixture."""


@pytest.fixture(autouse=True)
def isolated_ray_context(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep another test's cluster out of unit preflight and resource planning."""
    if "agent_ray_cluster" in request.fixturenames:
        monkeypatch.setenv("RAY_ADDRESS", request.getfixturevalue("agent_ray_cluster"))
        return

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "shutdown", lambda: None)

    def unexpected_ray_init(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Real Ray execution requires the agent_ray_cluster fixture")

    monkeypatch.setattr(ray, "init", unexpected_ray_init)


@pytest.fixture(scope="module")
def agent_ray_cluster() -> Iterator[str]:
    """Borrow an explicit cluster or own a bounded CPU cluster for this module."""
    from nemo_curator.core.client import RayClient
    from nemo_curator.stages.audio.audio_agent._ray import (
        _connected_address,
        _interpreter_ray_on_path,
        _ray_temp_root,
    )

    address = os.environ.get("CURATOR_TEST_RAY_ADDRESS") or os.environ.get("RAY_ADDRESS")
    if not address and ray.is_initialized():
        address = _connected_address(ray)
    if address:
        yield address
        return

    with tempfile.TemporaryDirectory(prefix="agent_ray_", dir=_ray_temp_root()) as directory:
        client = RayClient(
            num_cpus=4,
            num_gpus=0,
            object_store_memory=512 * 1024**2,
            ray_temp_dir=directory,
            include_dashboard=False,
        )
        previous_address = os.environ.get("RAY_ADDRESS")
        try:
            with _interpreter_ray_on_path():
                client.start()
            address = os.environ["RAY_ADDRESS"]
            # Export only during integration cases, not the intervening unit tests.
            os.environ.pop("RAY_ADDRESS", None)
            yield address
        finally:
            client.stop()
            if previous_address is not None:
                os.environ["RAY_ADDRESS"] = previous_address
            else:
                os.environ.pop("RAY_ADDRESS", None)
