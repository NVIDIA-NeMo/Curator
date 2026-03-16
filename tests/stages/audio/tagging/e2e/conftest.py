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

"""Shared fixtures for audio tagging e2e tests."""

import json
from pathlib import Path

import pytest

from tests import FIXTURES_DIR

AUDIO_FIXTURES_DIR = FIXTURES_DIR / "audio" / "tagging"
REFERENCE_DIR = AUDIO_FIXTURES_DIR / "reference"
CONFIGS_DIR = Path(__file__).parent / "configs"


@pytest.fixture
def get_input_manifest(tmp_path: Path):
    """Create a JSONL manifest from the test audio fixtures.

    Writes entries for audio_1.opus and audio_2.opus,
    matching the format expected by the tagging pipeline.
    """
    audio_dir = AUDIO_FIXTURES_DIR / "audios"
    manifest_path = tmp_path / "manifest.jsonl"

    target_files = {"audio_1.opus", "audio_2.opus"}

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for audio_file in sorted(audio_dir.iterdir()):
            if audio_file.name not in target_files:
                continue
            entry = {
                "audio_filepath": str(audio_file),
                "audio_item_id": audio_file.stem,
            }
            fout.write(json.dumps(entry) + "\n")

    return str(manifest_path)
