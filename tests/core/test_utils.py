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

import socket

import pytest

from nemo_curator.core.utils import get_free_port, ignore_ray_head_node


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


def test_get_free_port_checks_highest_valid_port(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSocket:
        def __enter__(self) -> "FakeSocket":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def setsockopt(self, *args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            _, port = address
            if port != 65535:
                raise OSError

    def fake_socket(*args: object, **kwargs: object) -> FakeSocket:
        _ = args, kwargs
        return FakeSocket()

    monkeypatch.setattr(socket, "socket", fake_socket)

    assert get_free_port(65535) == 65535
