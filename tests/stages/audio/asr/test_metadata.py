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

from nemo_curator.stages.audio.asr.metadata import ASRMetadata


def test_asr_metadata_includes_optional_speaker_and_verbatim_fields() -> None:
    metadata = ASRMetadata(
        audio_filepath="sample.wav",
        text="normalized text",
        duration=1.0,
        lang="en",
        split_type="train",
        source="UnitTest",
        extra={
            "gender": "extra_gender",
            "speaker_id": "extra_speaker",
            "age": "extra_age",
            "text_verbatim": "extra verbatim",
            "custom_key": "custom_value",
        },
    )

    row = metadata.to_dict()

    assert row["gender"] is None
    assert row["speaker_id"] is None
    assert row["age"] is None
    assert row["text_verbatim"] is None
    assert row["custom_key"] == "custom_value"


def test_asr_metadata_round_trips_optional_speaker_and_verbatim_fields() -> None:
    metadata = ASRMetadata.from_dict(
        {
            "audio_filepath": "sample.wav",
            "text": "normalized text",
            "duration": 1.0,
            "lang": "en",
            "split_type": "train",
            "source": "UnitTest",
            "gender": "female",
            "speaker_id": "speaker_1",
            "age": "adult",
            "text_verbatim": "Original Text",
        }
    )

    assert metadata.gender == "female"
    assert metadata.speaker_id == "speaker_1"
    assert metadata.age == "adult"
    assert metadata.text_verbatim == "Original Text"
