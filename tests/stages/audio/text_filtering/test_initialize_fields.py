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

from nemo_curator.stages.audio.text_filtering.initialize_fields import InitializeFieldsStage
from nemo_curator.tasks import AudioTask


def test_renames_text_and_sets_skip_me() -> None:
    stage = InitializeFieldsStage()
    task = AudioTask(data={"text": "original", "pred_text": "hello world"})
    result = stage.process(task)
    assert result.data["granary_v1_prediction"] == "original"
    assert "text" not in result.data
    assert result.data["skip_me"] == ""


def test_skip_me_always_reset_to_empty() -> None:
    stage = InitializeFieldsStage()
    task = AudioTask(data={"text": "t", "skip_me": "stale reason"})
    result = stage.process(task)
    assert result.data["skip_me"] == ""


def test_drops_default_keys() -> None:
    stage = InitializeFieldsStage()
    task = AudioTask(
        data={
            "text": "t",
            "answer": "Okay.",
            "source_lang": "en",
            "target_lang": "en",
            "decodercontext": "",
            "emotion": "<|emo:undefined|>",
            "diarize": "nodiarize",
            "duration": 3.5,
        }
    )
    result = stage.process(task)
    for key in ("answer", "target_lang", "decodercontext", "emotion", "diarize"):
        assert key not in result.data
    assert result.data["source_lang"] == "en"
    assert result.data["duration"] == 3.5
    assert result.data["granary_v1_prediction"] == "t"


def test_no_text_field_skips_rename() -> None:
    stage = InitializeFieldsStage()
    task = AudioTask(data={"pred_text": "hello"})
    result = stage.process(task)
    assert "granary_v1_prediction" not in result.data
    assert result.data["skip_me"] == ""


def test_custom_keys() -> None:
    stage = InitializeFieldsStage(
        original_text_key="src",
        granary_v1_key="v1",
        skip_me_key="drop",
        drop_keys=["extra"],
    )
    task = AudioTask(data={"src": "old text", "extra": "remove_me", "keep": "yes"})
    result = stage.process(task)
    assert result.data["v1"] == "old text"
    assert result.data["drop"] == ""
    assert "src" not in result.data
    assert "extra" not in result.data
    assert result.data["keep"] == "yes"


def test_missing_drop_keys_are_ignored() -> None:
    stage = InitializeFieldsStage()
    task = AudioTask(data={"text": "t"})
    result = stage.process(task)
    assert result.data["granary_v1_prediction"] == "t"


def test_pred_text_untouched() -> None:
    stage = InitializeFieldsStage()
    task = AudioTask(data={"text": "orig", "pred_text": "prediction"})
    result = stage.process(task)
    assert result.data["pred_text"] == "prediction"
