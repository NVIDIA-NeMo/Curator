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

from nemo_curator.stages.audio.simulation import FastMSS, FastMSSManifestReaderStage, FastMSSSimulationStage
from nemo_curator.tasks import EmptyTask, FileGroupTask


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
    result = stage.process(fixture())[0]
    from nemo_curator.stages.audio.common import GetAudioDurationStage

    assert GetAudioDurationStage(audio_filepath_key="mixture").validate_input(result)
    data = result.data
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

    names = {
        "FastMSS",
        "FastMSSSimulationStage",
        "FastMSSManifestReaderStage",
        "FastMSSRequests",
        "FastMSSRequestStage",
    }
    assert names <= set(get_index().stage_names())
    for name in names:
        assert not check_card(name, get_index().card(name))
        assert agent.describe_stage(name) is not None
    # Other modules may register temporary fixture classes during collection.
    # The shipped-card suite owns global coverage; this regression owns FastMSS.
    assert names.isdisjoint(audit()["uncarded_stages"])
