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

"""Discovery requires an explicit contract on each agent-ready stage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_curator.stages import audio, base
from nemo_curator.stages.audio import agent
from nemo_curator.stages.audio._agent import _catalog
from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    ConditionalWrite,
    IOSpec,
    StageContract,
)
from nemo_curator.stages.audio._agent._catalog import unavailable_modules
from nemo_curator.stages.audio._agent._conformance import produced_roles
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


class _DeclaredStage(AgentReady):
    def describe(self) -> StageContract:
        return StageContract()


class _InheritedStage(_DeclaredStage):
    pass


class _ReviewedSubclass(_DeclaredStage):
    def describe(self) -> StageContract:
        return super().describe()


class _UnmarkedStage:
    def describe(self) -> StageContract:
        return StageContract()


def test_discovery_requires_a_contract_on_the_concrete_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base,
        "_STAGE_REGISTRY",
        {
            "Declared": _DeclaredStage,
            "Inherited": _InheritedStage,
            "Reviewed": _ReviewedSubclass,
            "Unmarked": _UnmarkedStage,
        },
    )
    monkeypatch.setattr(_catalog, "_IMPORTED", True)

    assert _catalog.list_agent_ready_stages() == ["Declared", "Reviewed"]
    assert _catalog.get_agent_ready_stage_class("Declared") is _DeclaredStage
    assert _catalog.get_agent_ready_stage_class("Reviewed") is _ReviewedSubclass
    for name in ("Inherited", "Unmarked"):
        with pytest.raises(KeyError, match="not a registered agent-ready audio stage"):
            _catalog.get_agent_ready_stage_class(name)
        with pytest.raises(KeyError):
            _catalog.describe_stage(name)


def test_inherited_stage_remains_usable_outside_agent_discovery() -> None:
    assert isinstance(_InheritedStage().describe(), StageContract)


class _ConfiguredContractStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    def __init__(self, contract: StageContract) -> None:
        self.contract = contract

    def describe(self) -> StageContract:
        return self.contract

    def process(self, task: AudioTask) -> AudioTask:
        return task


def test_conditional_roles_are_discoverable_but_not_planner_guaranteed() -> None:
    producer_contract = StageContract(
        conditional_writes=[
            ConditionalWrite(
                writes=IOSpec(data_keys=["potential_metrics"]),
                condition="valid runtime data causes metric assignment",
            )
        ],
        key_roles={"potential_metrics": "metrics"},
    )
    assert produced_roles(producer_contract) == {"metrics"}

    consumer_contract = StageContract(
        reads=IOSpec(data_keys=["potential_metrics"]),
        key_roles={"potential_metrics": "metrics"},
    )
    report = validate_pipeline(
        [_ConfiguredContractStage(producer_contract), _ConfiguredContractStage(consumer_contract)],
        initial_roles=set(),
        initial_keys=set(),
    )

    # Conditional outputs are discoverable and let the consumer compose, but only as a
    # ``conditional_read`` warning: the key is never a guaranteed planner output.
    assert report.ok
    assert any(issue.code == "conditional_read" and issue.stage_index == 1 for issue in report.issues)
    assert (
        "potential_metrics"
        not in validate_pipeline(
            [_ConfiguredContractStage(producer_contract)], initial_roles=set(), initial_keys=set()
        ).produced_keys
    )


def test_public_facade_exposes_unavailable_modules_and_folder_source() -> None:
    """The documented public layer must expose foundation discovery features."""
    from nemo_curator.stages.audio import CreateInitialManifestAudioFolderStage
    from nemo_curator.stages.audio.common import CreateInitialManifestAudioFolderStage as FolderSource

    assert agent.unavailable_modules is unavailable_modules
    assert CreateInitialManifestAudioFolderStage is FolderSource
    assert "CreateInitialManifestAudioFolderStage" in audio.__all__


def test_public_discovery_reports_an_optional_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial install exposes skipped modules through the public facade."""
    missing_module = "nemo_curator.stages.audio.optional_missing"
    monkeypatch.setattr(_catalog, "_IMPORTED", False)
    monkeypatch.setattr(_catalog, "_SKIPPED", [])
    monkeypatch.setattr(
        _catalog.pkgutil,
        "walk_packages",
        lambda *_args, **_kwargs: [SimpleNamespace(name=missing_module)],
    )

    def fail_optional_import(name: str) -> None:
        assert name == missing_module
        message = "optional dependency is not installed"
        raise ModuleNotFoundError(message)

    monkeypatch.setattr(_catalog.importlib, "import_module", fail_optional_import)
    with pytest.warns(UserWarning, match="optional_missing"):
        missing = agent.unavailable_modules()

    assert missing == [
        {
            "module": missing_module,
            "error": "ModuleNotFoundError: optional dependency is not installed",
        }
    ]
