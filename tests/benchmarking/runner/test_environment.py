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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.environment import DEFAULT_SUBPROCESS_ENVIRONMENT, merge_subprocess_environment


def test_default_ray_data_diagnostics_enabled(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_CURATOR_RAY_DATA_DIAGNOSTICS", raising=False)

    env = merge_subprocess_environment({})

    assert (
        env["NEMO_CURATOR_RAY_DATA_DIAGNOSTICS"] == DEFAULT_SUBPROCESS_ENVIRONMENT["NEMO_CURATOR_RAY_DATA_DIAGNOSTICS"]
    )


def test_parent_environment_overrides_subprocess_defaults(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_CURATOR_RAY_DATA_DIAGNOSTICS", "0")

    env = merge_subprocess_environment({})

    assert env["NEMO_CURATOR_RAY_DATA_DIAGNOSTICS"] == "0"


def test_configured_environment_overrides_parent_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_CURATOR_RAY_DATA_DIAGNOSTICS", "0")

    env = merge_subprocess_environment({"NEMO_CURATOR_RAY_DATA_DIAGNOSTICS": "1"})

    assert env["NEMO_CURATOR_RAY_DATA_DIAGNOSTICS"] == "1"
