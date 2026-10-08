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

import pytest

from nemo_curator.stages import base
from nemo_curator.stages.audio._agent import _catalog
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, StageContract


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
