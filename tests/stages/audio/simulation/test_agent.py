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

"""Public recipe regression tests for generated meeting sources."""

import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.audio_agent import validate
from nemo_curator.stages.audio.audio_agent._verb_smoke_inputs import _bound_recipe, _isolate_smoke_outputs
from nemo_curator.stages.audio.audio_agent.contracts import SmokeReport
from nemo_curator.stages.audio.audio_agent.input_identity import resolve_dataset_binding
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef
from nemo_curator.stages.audio.simulation import FastMSS, FastMSSManifestReaderStage, FastMSSSimulationStage
from nemo_curator.tasks import EmptyTask, FileGroupTask


def recipe(tmp_path: Path, **params: object) -> Recipe:
    config = tmp_path / "meetings.yaml"
    config.write_text("n_meetings: 99\n")
    return Recipe(
        stages=[
            StageRef(
                ref="FastMSS",
                params={
                    "config_paths": str(config),
                    "output_dir": str(tmp_path / "production"),
                    "isolate_work_dirs": True,
                    **params,
                },
            )
        ]
    )


def test_public_validation_requires_safe_wrapping_and_guarantees_outputs(tmp_path: Path):
    rec = recipe(tmp_path)
    expected = ["audio_filepath", "duration", "text", "rttm_filepath"]
    verdict = validate(rec, expected_outputs=expected)
    assert verdict["runnable"], verdict
    assert not validate(recipe(tmp_path, isolate_work_dirs=False))["runnable"]
    renamed = recipe(tmp_path, duration_key="meeting_seconds", text_key="reference", rttm_filepath_key="rttm")
    verdict = validate(renamed, expected_outputs=["duration", "text", "rttm"])
    assert verdict["runnable"], verdict


def test_binding_is_one_readonly_generated_config_without_audio_profile(tmp_path: Path):
    rec = recipe(tmp_path)
    binding = resolve_dataset_binding(rec)
    assert binding.status == "resolved"
    assert binding.generated
    assert binding.profile_source is None
    assert binding.primary_path == str(tmp_path / "meetings.yaml")
    assert resolve_dataset_binding(rec, data=str(tmp_path / "other.yaml")).status == "mismatch"
    rec.stages[0].params["config_paths"] = [str(tmp_path / "meetings.yaml")]
    assert resolve_dataset_binding(rec).status != "resolved"


def test_smoke_caps_generator_and_isolates_output_without_editing_config(tmp_path: Path):
    rec = recipe(tmp_path)
    before = (tmp_path / "meetings.yaml").read_bytes()
    report = SmokeReport()
    bound = _bound_recipe(rec, 2, report, resolve_dataset_binding(rec))
    assert bound.recipe.stages[0].params["max_meetings"] == 2
    isolated = _isolate_smoke_outputs(bound, report)
    assert isolated.recipe.stages[0].params["output_dir"].startswith(isolated.output_root)
    assert rec.stages[0].params["output_dir"] == str(tmp_path / "production")
    assert "max_meetings" not in rec.stages[0].params
    assert (tmp_path / "meetings.yaml").read_bytes() == before
    import shutil

    shutil.rmtree(isolated.output_root)


@pytest.mark.parametrize(
    ("field", "value"), [("text", ""), ("duration", -1), ("duration", float("nan")), ("rttm_filepath", "/missing")]
)
def test_reader_enforces_guarantees(tmp_path: Path, field: str, value: object):
    audio = tmp_path / "meeting.wav"
    rttm = tmp_path / "meeting.rttm"
    audio.touch()
    rttm.touch()
    row = {
        "audio_filepath": str(audio),
        "duration": 1,
        "text": "overlapping turns",
        "rttm_filepath": str(rttm),
        field: value,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="FastMSS meeting"):
        FastMSSManifestReaderStage().process(FileGroupTask(dataset_name="test", data=[str(manifest)]))


def test_yml_is_rejected_even_with_same_stem_yaml(tmp_path: Path):
    yaml = tmp_path / "meetings.yaml"
    yml = tmp_path / "meetings.yml"
    yaml.write_text("n_meetings: 1\n")
    yml.write_text("n_meetings: 99\n")
    with pytest.raises(ValueError, match=r"\.yml"):
        FastMSS(config_paths=str(yml), output_dir=str(tmp_path / "out"))
    stage = FastMSSSimulationStage(output_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError, match=r"\.yaml"):
        stage.process(FileGroupTask(dataset_name="test", data=[str(yml)]))
    composite = FastMSS(config_paths=str(tmp_path), output_dir=str(tmp_path / "out"))
    partition = composite.decompose()[0]
    selected = [path for group in partition.process(EmptyTask()) for path in group.data]
    assert selected == [str(yaml)]


def test_reader_contract_renames_required_fields_and_preserves_extras(tmp_path: Path) -> None:
    from nemo_curator.stages.audio._agent._conformance import assert_agent_ready

    audio = tmp_path / "meeting.wav"
    rttm = tmp_path / "meeting.rttm"
    audio.touch()
    rttm.write_text("SPEAKER meeting 1 0 1 <NA> <NA> speaker <NA> <NA>\n")
    row = {
        "audio_filepath": str(audio),
        "duration": 1,
        "text": "first overlap",
        "rttm_filepath": str(rttm),
        "custom": {"retained": True},
    }
    manifest = tmp_path / "meetings.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    stage = FastMSSManifestReaderStage(
        audio_filepath_key="mixture", duration_key="seconds", text_key="reference", rttm_filepath_key="rttm"
    )

    def fixture() -> FileGroupTask:
        return FileGroupTask(dataset_name="test", data=[str(manifest)])

    assert_agent_ready(stage, fixture_factory=fixture, expected_cardinality="1:N fan-out")
    data = stage.process(fixture())[0].data
    assert data == {
        "mixture": str(audio),
        "seconds": 1,
        "reference": "first overlap",
        "rttm": str(rttm),
        "custom": {"retained": True},
    }


def test_every_simulation_stage_is_discoverable_and_has_a_valid_card() -> None:
    from nemo_curator.stages.audio import agent
    from nemo_curator.stages.audio.audio_agent.card_conformance import audit, check_card
    from nemo_curator.stages.audio.audio_agent.index import get_index

    names = {"FastMSS", "FastMSSSimulationStage", "FastMSSManifestReaderStage"}
    assert names <= set(get_index().stage_names())
    for name in names:
        assert not check_card(name, get_index().card(name))
        assert agent.describe_stage(name) is not None
    # Other modules may register temporary fixture classes during collection.
    # The shipped-card suite owns global coverage; this regression owns FastMSS.
    assert names.isdisjoint(audit()["uncarded_stages"])
