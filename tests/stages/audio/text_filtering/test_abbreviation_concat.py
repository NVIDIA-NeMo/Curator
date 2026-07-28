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

import pytest

from nemo_curator.stages.audio.text_filtering.abbreviation_concat import (
    AbbreviationConcatStage,
    concat_abbreviations,
)
from nemo_curator.tasks import AudioTask


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the A P I uses G P U acceleration", "the API uses GPU acceleration"),
        ("the U K's policy", "the UK's policy"),
        ("I I think this stays", "I I think this stays"),
        ("a cat sat nearby", "a cat sat nearby"),
    ],
)
def test_concat_abbreviations(text: str, expected: str) -> None:
    result, _ = concat_abbreviations(text)

    assert result == expected


def test_stage_records_changed_abbreviations() -> None:
    task = AudioTask(data={"text": "N V I D I A", "source_lang": "en"})

    AbbreviationConcatStage().process(task)

    assert task.data["text"] == "NVIDIA"
    assert "NVIDIA" in task.data["additional_notes"]["AbbreviationConcat"]


def test_stage_preserves_skipped_rows() -> None:
    task = AudioTask(data={"text": "A P I", "_skipme": "read_error"})

    AbbreviationConcatStage().process(task)

    assert task.data["text"] == "A P I"
