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

"""Shared fixtures for all audio stage tests (tagging + inference)."""

import pytest

from tests import FIXTURES_DIR


@pytest.fixture
def wav_filepath():
    return FIXTURES_DIR / "audio/tagging/audios/audio_1.wav"


@pytest.fixture
def audio_filepath():
    return FIXTURES_DIR / "audio/tagging/audios/audio_1.opus"
