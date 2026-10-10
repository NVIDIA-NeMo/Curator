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

from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    ManifestReader,
    ManifestReaderStage,
)


@dataclass
class _DurationSubclass(GetAudioDurationStage):
    marker: str = "DEFAULT"


@dataclass
class _ReaderStageSubclass(ManifestReaderStage):
    marker: str = "DEFAULT"


@dataclass
class _ReaderSubclass(ManifestReader):
    marker: str = "DEFAULT"


def test_legacy_subclass_positionals_keep_their_meaning() -> None:
    duration = _DurationSubclass("duration", "audio_filepath", "duration", "LEGACY")
    reader_stage = _ReaderStageSubclass("reader", "LEGACY")
    reader = _ReaderSubclass("input.jsonl", "reader", 1, None, [".jsonl"], None, "LEGACY")
    assert duration.marker == reader_stage.marker == reader.marker == "LEGACY"
    assert duration.waveform_key == "waveform"
    assert reader_stage.include_files is None
    assert reader.include_files is None
