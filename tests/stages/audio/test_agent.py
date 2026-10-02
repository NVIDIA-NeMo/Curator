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

"""Public foundation discovery, contract serialization, and composition checks."""

import json
from pathlib import Path

from nemo_curator.stages.audio import agent
from nemo_curator.stages.audio.common import (
    CreateInitialManifestAudioFolderStage,
    GetAudioDurationStage,
    ManifestReaderStage,
    ManifestWriterStage,
)


def test_public_discovery_round_trips_contracts_and_resolves_roles() -> None:
    catalog = json.loads(agent.catalog_as_json(include_dynamic_defaults=True))
    names = {entry["name"] for entry in catalog}
    assert {"ManifestWriterStage", "CreateInitialManifestAudioFolderStage", "GetAudioDurationStage"} <= names
    for entry in catalog:
        contract = agent.describe_stage(entry["name"])
        agent.assert_contract_wellformed(agent.get_agent_ready_stage_class(entry["name"]))
        assert json.loads(json.dumps(contract.to_dict())) == entry["contract"]
        assert agent.get_agent_ready_stage_class(entry["name"]).__name__ == entry["name"]
    assert "GetAudioDurationStage" in agent.find_producers("duration")
    assert "GetAudioDurationStage" in agent.find_consumers("audio_filepath")
    assert isinstance(agent.unavailable_modules(), list)


def test_public_validation_accepts_folder_manifest_and_rejects_wrong_task_type(tmp_path: Path) -> None:
    source = CreateInitialManifestAudioFolderStage(str(tmp_path))
    report = agent.validate_pipeline(
        [source, GetAudioDurationStage(), ManifestWriterStage(str(tmp_path / "manifest.jsonl"))],
        initial_keys=set(),
        initial_roles=set(),
    )
    assert report.ok, report.summary()
    invalid = agent.validate_pipeline([source, ManifestReaderStage()], initial_keys=set(), initial_roles=set())
    assert not invalid.ok
    assert any(issue.code == "task_type_mismatch" for issue in invalid.errors)
