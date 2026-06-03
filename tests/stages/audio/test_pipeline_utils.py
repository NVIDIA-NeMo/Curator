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

from nemo_curator.stages.audio.pipeline_utils import set_note


def test_set_note_writes_structured_stage_notes() -> None:
    data: dict = {}

    set_note(data, "StageA", "applied")

    assert data["additional_notes"] == {"StageA": "applied"}


def test_set_note_preserves_legacy_string_note() -> None:
    data = {"additional_notes": "Recovered:ASR"}

    set_note(data, "QwenASR", "skipped unsupported language")

    assert data["additional_notes"] == {
        "_legacy": "Recovered:ASR",
        "QwenASR": "skipped unsupported language",
    }
