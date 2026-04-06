# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Shared fixtures for all audio stage tests (tagging + inference + ALM)."""

import json
import subprocess

import pytest

from tests import FIXTURES_DIR

OPUS_FIXTURE = FIXTURES_DIR / "audio/tagging/audios/audio_1.opus"


@pytest.fixture
def sample_entries() -> list[dict]:
    """Load sample entries from ALM fixture file."""
    fixture_path = FIXTURES_DIR / "audio" / "alm" / "sample_input.jsonl"
    entries = []
    with open(fixture_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return entries


@pytest.fixture(scope="session")
def wav_filepath(tmp_path_factory: pytest.TempPathFactory):
    """Convert the opus fixture to 16 kHz mono WAV once per test session."""
    wav = tmp_path_factory.mktemp("audio") / "audio_1.wav"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(OPUS_FIXTURE),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603
    return wav


@pytest.fixture
def audio_filepath():
    return OPUS_FIXTURE
