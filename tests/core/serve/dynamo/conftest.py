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

from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest

from nemo_curator.core.serve.subprocess_mgr import ManagedSubprocess

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def captured_spawn() -> Iterator[list[dict[str, Any]]]:
    """Patch ``ManagedSubprocess.spawn`` to record calls without launching Ray actors.

    Yields the ``calls`` list — each entry is the kwargs the test code passed
    (``label``, ``pg``, ``bundle_index``, plus every keyword arg).
    """
    calls: list[dict[str, Any]] = []

    def fake_spawn(label, pg, bundle_index, **kwargs) -> ManagedSubprocess:  # noqa: ANN001
        calls.append({"label": label, "pg": pg, "bundle_index": bundle_index, **kwargs})
        return ManagedSubprocess(label=label, actor=object())

    with mock.patch.object(ManagedSubprocess, "spawn", side_effect=fake_spawn):
        yield calls
