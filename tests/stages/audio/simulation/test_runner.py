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

"""Verify transcript assembly and unsupported upstream configurations."""

import gzip
import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.simulation._runner import enrich_manifest, validate_config


@pytest.mark.parametrize(
    "config",
    [
        {"min_max_spk": [1, 1]},
        {"min_max_spk": [1, 4]},
        {"min_max_spk": [3, 2]},
        {"save_anechoic": True, "save_spk": True, "reverberate": True},
        {"save_anechoic": True, "save_spk": True, "reverberate": True, "reverb_prob": 0},
        {"save_anechoic": True, "reverberate": False, "reverb_prob": 1},
    ],
)
def test_unsafe_engine_settings_are_rejected(config: dict) -> None:
    with pytest.raises(ValueError, match="FastMSS"):
        validate_config(config)


@pytest.mark.parametrize(
    "config",
    [
        {"min_max_spk": [2, 2]},
        {"save_anechoic": False, "reverberate": False},
        {"save_anechoic": True, "save_spk": True, "reverberate": True, "reverb_prob": 1},
    ],
)
def test_supported_settings_are_accepted(config: dict) -> None:
    validate_config(config)


@pytest.mark.parametrize("missing_text", [False, True])
def test_transcripts_use_recording_identity_and_start_order(tmp_path: Path, missing_text: bool) -> None:
    manifest = tmp_path / "nemo_manifest.json"
    rows = [
        {"audio_filepath": "/audio/renamed.wav", "duration": 4, "custom": "preserved"},
        {"audio_filepath": "/audio/other.wav", "text": "existing transcript"},
    ]
    original = "".join(json.dumps(row) + "\n" for row in rows)
    manifest.write_text(original)
    directory = tmp_path / "manifests"
    directory.mkdir()
    with gzip.open(directory / "synth-source-train-recordings.jsonl.gz", "wt") as output:
        output.write(json.dumps({"id": "meeting", "sources": [{"source": "/audio/renamed.wav"}]}))
    segments = [
        {"recording_id": "meeting", "id": "c", "start": 1, "duration": 2, "text": "later"},
        {"recording_id": "meeting", "id": "b", "start": 0, "duration": 2, "text": "overlap"},
        {"recording_id": "unrelated", "id": "x", "start": 0, "text": "exclude"},
        {"recording_id": "meeting", "id": "a", "start": 0, "duration": 2, "text": " привет "},
    ]
    if missing_text:
        segments[0]["text"] = None
    with gzip.open(directory / "synth-source-train-supervisions.jsonl.gz", "wt", encoding="utf-8") as output:
        output.write("".join(json.dumps(segment) + "\n" for segment in segments))
    if missing_text:
        with pytest.raises(ValueError, match="missing supervision transcripts"):
            enrich_manifest(tmp_path)
        assert manifest.read_text() == original
    else:
        enrich_manifest(tmp_path)
        actual = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        assert actual == [{**rows[0], "text": "привет overlap later"}, rows[1]]


def test_missing_supervisions_fail_instead_of_emitting_unlabelled_meetings(tmp_path: Path) -> None:
    (tmp_path / "nemo_manifest.json").write_text(json.dumps({"audio_filepath": "/audio/meeting.wav"}))
    with pytest.raises(ValueError, match="missing supervision transcripts"):
        enrich_manifest(tmp_path)
