# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_curator.stages.audio.text_filtering.finalize_fields import FinalizeFieldsStage
from nemo_curator.tasks import AudioTask


def test_happy_path() -> None:
    stage = FinalizeFieldsStage()
    task = AudioTask(
        data={
            "granary_v1_prediction": "original text",
            "cleaned_text": "cleaned version",
            "pred_text": "pred",
            "pnc": "pnc",
            "itn": "noitn",
            "timestamp": "notimestamp",
            "audio_filepath": "/a.wav",
            "duration": 3.5,
        }
    )
    result = stage.process(task)
    assert result.data["cleaned_text"] == "cleaned version"
    assert result.data["granary_v1_prediction"] == "original text"
    assert "pnc" not in result.data
    assert "itn" not in result.data
    assert "timestamp" not in result.data
    assert result.data["audio_filepath"] == "/a.wav"
    assert result.data["duration"] == 3.5


def test_missing_drop_keys_are_ignored() -> None:
    stage = FinalizeFieldsStage()
    task = AudioTask(data={"cleaned_text": "c"})
    result = stage.process(task)
    assert result.data["cleaned_text"] == "c"


def test_custom_drop_keys() -> None:
    stage = FinalizeFieldsStage(drop_keys=["custom_field", "another"])
    task = AudioTask(data={"cleaned_text": "c", "custom_field": "drop_me", "another": "also_drop"})
    result = stage.process(task)
    assert "custom_field" not in result.data
    assert "another" not in result.data
    assert result.data["cleaned_text"] == "c"


def test_other_fields_preserved() -> None:
    stage = FinalizeFieldsStage()
    task = AudioTask(
        data={
            "cleaned_text": "c",
            "pred_text": "raw",
            "skip_me": "",
            "shard_id": 42,
            "granary_v1_prediction": "orig",
        }
    )
    result = stage.process(task)
    assert result.data["pred_text"] == "raw"
    assert result.data["skip_me"] == ""
    assert result.data["shard_id"] == 42
    assert result.data["granary_v1_prediction"] == "orig"


def test_cleaned_text_kept() -> None:
    stage = FinalizeFieldsStage()
    task = AudioTask(data={"cleaned_text": "clean"})
    result = stage.process(task)
    assert result.data["cleaned_text"] == "clean"
