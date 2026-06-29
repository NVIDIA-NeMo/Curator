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
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.session import Session

_RESULTS_PATH = str(Path(__file__).resolve().parent)


def _config(entries: list[dict], **overrides: object) -> dict:
    return {
        "paths": [{"name": "results_path", "host_path": _RESULTS_PATH}],
        "entries": entries,
        **overrides,
    }


def test_session_defaults_max_timeout_s() -> None:
    session = Session.from_dict(_config([{"name": "entry_a", "script": "benchmark.py"}]))

    assert session.max_timeout_s == 14340
    assert session.entries[0].timeout_s == 7200


def test_session_rejects_timeout_above_max_timeout_s() -> None:
    with pytest.raises(ValueError, match=r"entry_a.*timeout_s=101.*max_timeout_s=100"):
        Session.from_dict(
            _config(
                [{"name": "entry_a", "script": "benchmark.py", "timeout_s": 101}],
                max_timeout_s=100,
            )
        )


def test_session_accepts_timeout_equal_to_max_timeout_s() -> None:
    session = Session.from_dict(
        _config(
            [{"name": "entry_a", "script": "benchmark.py", "timeout_s": 100}],
            max_timeout_s=100,
        )
    )

    assert session.entries[0].timeout_s == 100


@pytest.mark.parametrize("bad_max_timeout_s", [0, -1, True, 1.5])
def test_session_rejects_invalid_max_timeout_s(bad_max_timeout_s: object) -> None:
    with pytest.raises(ValueError, match="Invalid max_timeout_s"):
        Session.from_dict(
            _config(
                [{"name": "entry_a", "script": "benchmark.py"}],
                max_timeout_s=bad_max_timeout_s,
            )
        )


def test_session_applies_max_timeout_s_after_default_timeout_s() -> None:
    with pytest.raises(ValueError, match=r"entry_a.*timeout_s=120.*max_timeout_s=100"):
        Session.from_dict(
            _config(
                [{"name": "entry_a", "script": "benchmark.py"}],
                default_timeout_s=120,
                max_timeout_s=100,
            )
        )


def _generate_job(monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict]:
    ruamel_module = types.ModuleType("ruamel")
    yaml_module = types.ModuleType("ruamel.yaml")
    scalarstring_module = types.ModuleType("ruamel.yaml.scalarstring")

    class YAML:
        default_flow_style = False
        preserve_quotes = False

    yaml_module.YAML = YAML
    scalarstring_module.DoubleQuotedScalarString = str
    ruamel_module.yaml = yaml_module
    monkeypatch.setitem(sys.modules, "ruamel", ruamel_module)
    monkeypatch.setitem(sys.modules, "ruamel.yaml", yaml_module)
    monkeypatch.setitem(sys.modules, "ruamel.yaml.scalarstring", scalarstring_module)
    sys.modules.pop("tools.generate_ci_tests", None)

    from tools.generate_ci_tests import generate_job

    return generate_job


def test_generate_job_rejects_timeout_above_max_timeout_s(monkeypatch: pytest.MonkeyPatch) -> None:
    generate_job = _generate_job(monkeypatch)

    with pytest.raises(ValueError, match=r"entry_a.*timeout_s=101.*max_timeout_s=100"):
        generate_job(
            {"name": "entry_a", "timeout_s": 101},
            "nightly",
            default_timeout_s=120,
            cleanup_timeout_s=60,
            min_timeout_s=600,
            max_timeout_s=100,
        )


def test_generate_job_accepts_timeout_equal_to_max_timeout_s(monkeypatch: pytest.MonkeyPatch) -> None:
    generate_job = _generate_job(monkeypatch)

    job = generate_job(
        {"name": "entry_a", "timeout_s": 100},
        "nightly",
        default_timeout_s=120,
        cleanup_timeout_s=60,
        min_timeout_s=0,
        max_timeout_s=100,
    )

    assert job["variables"]["ENTRY_NAME"] == "entry_a"
    assert job["variables"]["TIME"] == "00:02:40"


def test_generate_job_applies_max_timeout_s_after_default_timeout_s(monkeypatch: pytest.MonkeyPatch) -> None:
    generate_job = _generate_job(monkeypatch)

    with pytest.raises(ValueError, match=r"entry_a.*timeout_s=120.*max_timeout_s=100"):
        generate_job(
            {"name": "entry_a"},
            "nightly",
            default_timeout_s=120,
            cleanup_timeout_s=60,
            min_timeout_s=600,
            max_timeout_s=100,
        )
