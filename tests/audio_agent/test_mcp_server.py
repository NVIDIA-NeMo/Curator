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

"""Contract tests for the optional MCP adapter.

The real ``mcp`` package is deliberately not a test dependency. A tiny fake
captures the decorated callables so their public schemas and SDK forwarding can
be checked directly.
"""

from __future__ import annotations

import inspect
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from nemo_curator import audio_agent as aa
from nemo_curator.audio_agent import mcp_server

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


class _FakeFastMCP:
    def __init__(self, name: str):
        self.name = name
        self.tools: dict[str, Any] = {}

    def tool(self):  # noqa: ANN202
        def register(function):  # noqa: ANN001, ANN202
            self.tools[function.__name__] = function
            return function

        return register


def _build_fake_server(monkeypatch) -> _FakeFastMCP:  # noqa: ANN001
    mcp_module = ModuleType("mcp")
    mcp_module.__path__ = []  # type: ignore[attr-defined]
    server_module = ModuleType("mcp.server")
    server_module.__path__ = []  # type: ignore[attr-defined]
    fastmcp_module = ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP  # type: ignore[attr-defined]
    mcp_module.server = server_module  # type: ignore[attr-defined]
    server_module.fastmcp = fastmcp_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)
    return mcp_server.build_server()


def test_mcp_exposes_reuse_provenance_and_health_tools(monkeypatch) -> None:  # noqa: ANN001
    server = _build_fake_server(monkeypatch)

    assert {
        "runs",
        "reuse_scan",
        "reindex",
        "plan_continuation",
        "delta_run",
        "plan_checkpoint",
        "doctor",
        "diagnose",
    } <= server.tools.keys()


def test_mcp_parameter_contract_matches_public_verb_surface(monkeypatch) -> None:  # noqa: ANN001
    tools = _build_fake_server(monkeypatch).tools

    expected_parameters = {
        "context": [
            "goal",
            "data",
            "stages",
            "roles",
            "planning_preference",
        ],
        "report": ["output", "recipe", "data"],
        "runs": ["run_id", "data", "stage", "since", "limit", "goal"],
        "reuse_scan": ["recipe", "data", "limit"],
        "run": [
            "recipe",
            "confirm",
            "data",
            "output_dir",
            "checkpoint_path",
            "bootstrap_ray",
            "smoke_token",
            "calibration",
            "goal",
        ],
        "diagnose": [
            "error",
            "recipe",
            "operation",
            "phase",
            "attempted_actions",
            "execution_target",
        ],
        "plan_continuation": [
            "recipe",
            "parent_run_id",
            "data",
            "execute",
            "choice",
            "confirm",
            "output_dir",
            "checkpoint_path",
            "bootstrap_ray",
            "smoke_token",
            "calibration",
            "goal",
        ],
        "delta_run": [
            "recipe",
            "from_run",
            "data",
            "confirm",
            "bootstrap_ray",
            "smoke_token",
            "calibration",
            "goal",
        ],
        "plan_checkpoint": [
            "recipe",
            "from_run",
            "data",
            "output_path",
            "decision_stage",
            "decision_value",
            "decision_conditions",
            "choice",
            "retention_sec",
            "owner",
        ],
    }
    for tool_name, parameter_names in expected_parameters.items():
        assert list(inspect.signature(tools[tool_name]).parameters) == parameter_names

    continuation_signature = inspect.signature(tools["plan_continuation"])
    assert continuation_signature.parameters["parent_run_id"].default is None


def test_mcp_context_forwards_planning_preference(
    monkeypatch: MonkeyPatch,
) -> None:
    tool = _build_fake_server(monkeypatch).tools["context"]
    context_mock = Mock(return_value={"planning_preference": {"curation_mode": "fast_first"}})
    monkeypatch.setattr(aa, "context", context_mock)
    preference = {
        "schema_version": 1,
        "curation_mode": "fast_first",
        "source": "explicit_user_choice",
    }

    result = tool(
        goal={"task": "quality_filter"},
        planning_preference=preference,
    )

    assert result["planning_preference"]["curation_mode"] == "fast_first"
    context_mock.assert_called_once_with(
        {"task": "quality_filter"},
        data=None,
        stages=None,
        roles=None,
        planning_preference=preference,
    )


def test_mcp_checkpoint_planner_rejects_non_scalar_decision_value(
    monkeypatch,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    tool = _build_fake_server(monkeypatch).tools["plan_checkpoint"]
    recipe = {
        "stages": [
            {
                "ref": "ManifestReader",
                "params": {"manifest_path": str(tmp_path / "input.jsonl")},
            },
            {"ref": "GetAudioDurationStage", "params": {"duration_key": "duration"}},
            {
                "ref": "PreserveByValueStage",
                "params": {
                    "input_value_key": "duration",
                    "target_value": 5.0,
                    "operator": "ge",
                },
            },
        ]
    }

    result = tool(recipe=recipe, decision_value={"threshold": 7})

    assert result["status"] == "refused"
    assert "JSON scalar" in result["reason"]


def test_mcp_checkpoint_planner_fails_closed_for_compound_scalar_feedback(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _build_fake_server(monkeypatch).tools["plan_checkpoint"]
    recipe = {
        "stages": [
            {
                "ref": "ManifestReader",
                "params": {"manifest_path": str(tmp_path / "input.jsonl")},
            },
            {
                "ref": "SIGMOSFilterStage",
                "params": {
                    "action": "annotate",
                    "mode": "task",
                    "input_residency": "file",
                },
            },
            {
                "ref": "PreserveByValueConditionsStage",
                "params": {
                    "conditions": [
                        {
                            "input_value_key": "sigmos_noise",
                            "target_value": 4.0,
                            "operator": "ge",
                        },
                        {
                            "input_value_key": "sigmos_ovrl",
                            "target_value": 3.5,
                            "operator": "ge",
                        },
                    ],
                    "missing_value_policy": "drop",
                },
            },
        ]
    }

    result = tool(
        recipe=recipe,
        decision_stage="SIGMOSFilterStage",
        decision_value=4.1,
    )

    assert result["status"] == "no_candidate"
    assert "compound decisions cannot be tuned" in result["rejected"][0]["reason"]


def test_mcp_checkpoint_planner_accepts_complete_compound_conditions(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _build_fake_server(monkeypatch).tools["plan_checkpoint"]
    recipe = {
        "stages": [
            {
                "ref": "ManifestReader",
                "params": {"manifest_path": str(tmp_path / "input.jsonl")},
            },
            {
                "ref": "SIGMOSFilterStage",
                "params": {
                    "action": "annotate",
                    "mode": "task",
                    "input_residency": "file",
                },
            },
            {
                "ref": "PreserveByValueConditionsStage",
                "params": {
                    "conditions": [
                        {
                            "input_value_key": "sigmos_noise",
                            "target_value": 4.0,
                            "operator": "ge",
                        },
                        {
                            "input_value_key": "sigmos_ovrl",
                            "target_value": 3.5,
                            "operator": "ge",
                        },
                    ],
                    "missing_value_policy": "drop",
                },
            },
        ]
    }

    result = tool(
        recipe=recipe,
        decision_stage="SIGMOSFilterStage",
        decision_conditions={"sigmos_ovrl": 3.8, "sigmos_sig": 3.6},
    )

    assert result["status"] == "candidates"
    assert result["candidates"][0]["conditions"] == [
        {"input_value_key": "sigmos_ovrl", "target_value": 3.8, "operator": "ge"},
        {"input_value_key": "sigmos_sig", "target_value": 3.6, "operator": "ge"},
    ]


def test_mcp_forwards_run_and_report_arguments_without_reinterpretation(monkeypatch) -> None:  # noqa: ANN001
    tools = _build_fake_server(monkeypatch).tools
    run_mock = Mock(return_value={"status": "completed"})
    report_mock = Mock(return_value={"status": "ok"})
    monkeypatch.setattr(aa, "run", run_mock)
    monkeypatch.setattr(aa, "report", report_mock)

    recipe = {"stages": [{"ref": "ManifestReaderStage"}]}
    calibration = {"calibration": {"SomeStage": {"gpu_mem_gb": 2.5}}}
    goal = {"task": "quality_filter"}
    assert tools["run"](
        recipe,
        confirm="approved-hash",
        data="/data/input.jsonl",
        output_dir="/data/out",
        checkpoint_path="/data/checkpoint",
        bootstrap_ray=True,
        smoke_token="smoke-proof",  # noqa: S106
        calibration=calibration,
        goal=goal,
    ) == {"status": "completed"}
    run_mock.assert_called_once_with(
        recipe,
        confirm="approved-hash",
        data="/data/input.jsonl",
        output_dir="/data/out",
        checkpoint_path="/data/checkpoint",
        bootstrap_ray=True,
        smoke_token="smoke-proof",  # noqa: S106
        calibration=calibration,
        goal=goal,
    )

    assert tools["report"]("/data/out.jsonl", recipe=recipe, data="/data/input.jsonl") == {"status": "ok"}
    report_mock.assert_called_once_with(
        "/data/out.jsonl",
        recipe=recipe,
        data="/data/input.jsonl",
    )


def test_mcp_forwards_diagnosis_context_without_applying_a_fix(monkeypatch) -> None:  # noqa: ANN001
    tools = _build_fake_server(monkeypatch).tools
    diagnose_mock = Mock(return_value={"status": "action_required"})
    monkeypatch.setattr(aa, "diagnose", diagnose_mock)
    recipe = {"stages": [{"ref": "ManifestReader"}]}

    assert tools["diagnose"](
        "CUDA error 222",
        recipe=recipe,
        operation="smoke",
        phase="pipeline_execution",
        attempted_actions=["upgrade_nvidia_driver"],
        execution_target="local",
    ) == {"status": "action_required"}
    diagnose_mock.assert_called_once_with(
        "CUDA error 222",
        recipe=recipe,
        operation="smoke",
        phase="pipeline_execution",
        attempted_actions=["upgrade_nvidia_driver"],
        execution_target="local",
    )


def test_mcp_forwards_provenance_and_reuse_arguments(monkeypatch) -> None:  # noqa: ANN001
    tools = _build_fake_server(monkeypatch).tools
    runs_mock = Mock(return_value={"runs": []})
    scan_mock = Mock(return_value={"decision": "fresh"})
    reindex_mock = Mock(return_value={"status": "ok"})
    continuation_mock = Mock(return_value={"mode": "incremental"})
    doctor_mock = Mock(return_value={"status": "healthy"})
    monkeypatch.setattr(aa, "runs", runs_mock)
    monkeypatch.setattr(aa, "reuse_scan", scan_mock)
    monkeypatch.setattr(aa, "reindex", reindex_mock)
    monkeypatch.setattr(aa, "plan_continuation", continuation_mock)
    monkeypatch.setattr(aa, "doctor", doctor_mock)

    recipe = {"stages": [{"ref": "ManifestReaderStage"}]}
    assert tools["runs"](
        run_id=None,
        data="content:dataset-key",
        stage="SomeStage",
        since="2026-07-01T00:00:00Z",
        limit=7,
        goal="16 kHz mono with a quality filter",
    ) == {"runs": []}
    runs_mock.assert_called_once_with(
        run_id=None,
        data="content:dataset-key",
        stage="SomeStage",
        since="2026-07-01T00:00:00Z",
        limit=7,
        goal="16 kHz mono with a quality filter",
    )

    assert tools["reuse_scan"](recipe, data="/data/input.jsonl", limit=3) == {"decision": "fresh"}
    scan_mock.assert_called_once_with(recipe, data="/data/input.jsonl", limit=3)
    assert tools["reindex"]() == {"status": "ok"}
    reindex_mock.assert_called_once_with()
    assert tools["doctor"]() == {"status": "healthy"}
    doctor_mock.assert_called_once_with()

    calibration = {"calibration": {"SomeStage": {"gpu_mem_gb": 2.5}}}
    goal = {"task": "transcribe"}
    assert tools["plan_continuation"](
        recipe,
        data="/data/input.jsonl",
        execute=True,
        choice="extend",
        confirm="approved-hash",
        output_dir="/data/out",
        checkpoint_path="/data/checkpoint",
        bootstrap_ray=True,
        smoke_token="smoke-proof",  # noqa: S106
        calibration=calibration,
        goal=goal,
    ) == {"mode": "incremental"}
    continuation_mock.assert_called_once_with(
        recipe,
        None,
        data="/data/input.jsonl",
        execute=True,
        choice="extend",
        confirm="approved-hash",
        output_dir="/data/out",
        checkpoint_path="/data/checkpoint",
        bootstrap_ray=True,
        smoke_token="smoke-proof",  # noqa: S106
        calibration=calibration,
        goal=goal,
    )


def test_calibrate_result_is_forwarded_as_an_accepted_wrapper(monkeypatch) -> None:  # noqa: ANN001
    tools = _build_fake_server(monkeypatch).tools
    wrapper = {"calibration": {"SomeStage": {"gpu_mem_gb": 2.5}}}
    calibrate_mock = Mock(return_value=wrapper)
    smoke_mock = Mock(return_value={"status": "completed"})
    monkeypatch.setattr(aa, "calibrate", calibrate_mock)
    monkeypatch.setattr(aa, "smoke", smoke_mock)

    smoke_report = {"per_stage_metrics": {}}
    measured = tools["calibrate"](smoke_report)
    assert measured is wrapper
    calibrate_mock.assert_called_once_with(smoke_report)

    recipe = {"stages": [{"ref": "ManifestReaderStage"}]}
    tools["smoke"](recipe, calibration=measured)
    smoke_mock.assert_called_once_with(
        recipe,
        sample=10,
        data=None,
        output_dir=None,
        bootstrap_ray=False,
        calibration=wrapper,
    )
    calibrate_docs = inspect.getdoc(tools["calibrate"])
    assert "Pass that complete" in calibrate_docs
    assert "result unchanged" in calibrate_docs
