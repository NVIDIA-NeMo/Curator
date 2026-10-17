# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from nemo_curator.stages.audio.audio_agent.profiler import profile_data
from nemo_curator.stages.audio.audio_agent.semantic_review import _data_profile_summary


def test_folder_profile_does_not_claim_sibling_transcripts_absent(tmp_path: Path) -> None:
    sidecar = tmp_path / "labels.jsonl"
    sidecar.write_text('{"text": "a supplied transcript"}\n')
    before = sidecar.read_bytes()
    profile = profile_data(str(tmp_path), max_probe=0)
    assert profile.has_transcripts is False
    assert any("transcript_presence=not_inspected" in note for note in profile.notes)
    summary = _data_profile_summary(profile.to_dict())
    assert summary["notes"] == profile.notes
    assert sidecar.read_bytes() == before


def test_manifest_profile_still_reports_observed_transcript_values(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text('{"text": "a supplied transcript"}\n')
    profile = profile_data(str(manifest), max_probe=0)
    assert profile.has_transcripts is True
    assert not any("transcript_presence=not_inspected" in note for note in profile.notes)
