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

from dataclasses import dataclass

import pytest

from nemo_curator.stages.audio.tagging.text.chinese_conversion import ChineseConversionStage
from nemo_curator.stages.audio.tagging.text.itn import InverseTextNormalizationStage
from nemo_curator.stages.audio.text_filtering import AbbreviationConcatStage, RegexSubstitutionStage
from nemo_curator.stages.audio.text_filtering.text_transform import RowTextTransform, SegmentTextTransform
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class _Upper(RowTextTransform, ProcessingStage[AudioTask, AudioTask]):
    text_key: str = "text"
    output_text_key: str = "out"
    skip_me_key: str = "_skipme"

    def transform_row(self, task: AudioTask, text: str) -> AudioTask:
        task.data[self.output_text_key] = text.upper()
        return task


@dataclass
class _SegmentUpper(SegmentTextTransform, ProcessingStage[AudioTask, AudioTask]):
    text_key: str = "text"
    segments_key: str = "segments"
    output_suffix: str = "_up"

    def accepts_text(self, text: str) -> bool:
        return bool(text)

    def transform_text(self, text: str) -> str:
        return text.upper()


def _task(**data: object) -> AudioTask:
    return AudioTask(dataset_name="d", data=dict(data))


@pytest.mark.parametrize(
    ("data", "expected_out"),
    [
        ({"text": "abc"}, "ABC"),
        ({"text": "abc", "_skipme": "earlier"}, ""),
        ({"text": 7}, ""),
        ({"text": None, "out": "kept"}, "kept"),
        ({}, ""),
    ],
)
def test_row_transform_guards(data: dict, expected_out: str) -> None:
    assert _Upper().process(_task(**data)).data["out"] == expected_out


def test_segment_transform_writes_suffix_key_only_for_accepted_text() -> None:
    task = _task(segments=[{"text": "a"}, {"text": ""}, {"other": "b"}])

    assert _SegmentUpper().process(task).data["segments"] == [
        {"text": "a", "text_up": "A"},
        {"text": ""},
        {"other": "b"},
    ]


@pytest.mark.parametrize(
    ("stage_cls", "base"),
    [
        (RegexSubstitutionStage, RowTextTransform),
        (AbbreviationConcatStage, RowTextTransform),
        (InverseTextNormalizationStage, SegmentTextTransform),
        (ChineseConversionStage, SegmentTextTransform),
    ],
)
def test_text_transform_stages_share_the_base(stage_cls: type, base: type) -> None:
    assert issubclass(stage_cls, base)
    assert stage_cls.process is base.process
