# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Generic generated-source protocol, independent of any simulation toolkit."""

from pathlib import Path

import pytest

from nemo_curator.stages.audio.audio_agent import _capabilities
from nemo_curator.stages.audio.audio_agent._verb_smoke_inputs import _bound_recipe
from nemo_curator.stages.audio.audio_agent.contracts import SmokeReport
from nemo_curator.stages.audio.audio_agent.input_identity import resolve_dataset_binding
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = tmp_path / "generator.cfg"
    config.write_text("records=99")
    spec = {"kind": "config_file", "path_param": "configuration", "limit_param": "row_cap"}
    monkeypatch.setattr(_capabilities, "io_for", lambda _ref: {"source": spec})
    monkeypatch.setattr(_capabilities, "configured_values", lambda _ref, params: {"row_cap": -1, **params})
    return Recipe(stages=[StageRef(ref="FutureGenerator", params={"configuration": str(config)})])


def test_config_identity_is_generated_and_not_profiled(source: Recipe, tmp_path: Path):
    binding = resolve_dataset_binding(source)
    assert binding.status == "resolved"
    assert binding.generated
    assert binding.profile_source is None
    assert resolve_dataset_binding(source, str(tmp_path / "other.cfg")).status == "mismatch"
    assert not binding.selected_manifest_files


@pytest.mark.parametrize("value", [-1, 1, 9])
def test_smoke_caps_output_without_rewriting_configuration(source: Recipe, value: object):
    source.stages[0].params["row_cap"] = value
    bound = _bound_recipe(source, 2, SmokeReport(), resolve_dataset_binding(source))
    assert bound.recipe.stages[0].params["row_cap"] == (1 if value == 1 else 2)
    assert source.stages[0].params["row_cap"] == value


@pytest.mark.parametrize("value", [0, -2, True, None, "9"])
def test_invalid_cap_fails_closed(source: Recipe, value: object):
    source.stages[0].params["row_cap"] = value
    bound = _bound_recipe(source, 2, SmokeReport(), resolve_dataset_binding(source))
    assert bound.recipe is None


def test_config_protocol_rejects_directories_and_selector_lists(source: Recipe, tmp_path: Path):
    for value in (str(tmp_path), [source.stages[0].params["configuration"]]):
        source.stages[0].params["configuration"] = value
        assert resolve_dataset_binding(source).status != "resolved"


def test_config_card_requires_real_bound_parameter():
    card = {"io": {"source": {"kind": "config_file", "path_param": "configuration"}}, "verified": {"io": "mechanical"}}
    assert _capabilities.io_violations("Generator", card, {"configuration"})
    card["io"]["source"]["limit_param"] = "row_cap"
    assert not _capabilities.io_violations("Generator", card, {"configuration", "row_cap"})
