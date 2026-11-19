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

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.utils import get_gpu_stats


def test_get_gpu_stats_returns_empty_without_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    def new_query() -> None:
        msg = "NVML Shared Library Not Found"
        raise RuntimeError(msg)

    monkeypatch.setitem(sys.modules, "gpustat", SimpleNamespace(new_query=new_query))

    assert get_gpu_stats() == {}
