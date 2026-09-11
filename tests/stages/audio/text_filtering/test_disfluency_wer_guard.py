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

# ruff: noqa: INP001

import pytest

from nemo_curator.stages.audio.text_filtering import DisfluencyWerGuardStage
from nemo_curator.tasks import AudioTask

_REF_KEY = "qwen3_prediction_s1"
_HYP_KEY = "qwen3_prediction_s2"
_RATE_KEY = "disfluency_wer"


def _task(reference: str, hypothesis: str, language: str | None = None) -> AudioTask:
    data = {_REF_KEY: reference, _HYP_KEY: hypothesis, "_skipme": ""}
    if language is not None:
        data["source_lang"] = language
    return AudioTask(data=data)


@pytest.mark.parametrize("language", ["ja", "zh", "th", "zh_CN", "Japanese"])
def test_no_space_languages_use_cer_and_keep_near_match(language: str) -> None:
    task = _task("这是一个测试句子用来检查协议", "这是一个测试句子用来检查协义", language)

    DisfluencyWerGuardStage(max_wer_pct=20.0).process(task)

    assert 0.0 < task.data[_RATE_KEY] < 20.0
    assert task.data[_HYP_KEY] == "这是一个测试句子用来检查协义"


def test_no_space_genuine_disagreement_restores_original() -> None:
    task = _task("あなたのおすすめの映画は何ですか", "今日はとても良い天気ですね", "ja")

    DisfluencyWerGuardStage(max_wer_pct=20.0).process(task)

    assert task.data[_RATE_KEY] > 20.0
    assert task.data[_HYP_KEY] == task.data[_REF_KEY]


def test_space_separated_language_keeps_wer_behavior() -> None:
    task = _task("one two three four", "one two three", "en")

    DisfluencyWerGuardStage(max_wer_pct=20.0).process(task)

    assert task.data[_RATE_KEY] == 25.0
    assert task.data[_HYP_KEY] == task.data[_REF_KEY]


def test_missing_language_defaults_to_wer() -> None:
    task = _task("hello world", "hello")

    DisfluencyWerGuardStage(max_wer_pct=75.0).process(task)

    assert task.data[_RATE_KEY] == 50.0
    assert task.data[_HYP_KEY] == "hello"


def test_custom_language_and_text_keys_are_supported() -> None:
    task = AudioTask(data={"before": "日本語の文字列", "after": "日本語の文宇列", "lang": "Japanese"})
    stage = DisfluencyWerGuardStage(
        ref_text_key="before",
        hyp_text_key="after",
        language_key="lang",
        wer_key="error_rate",
        max_wer_pct=20.0,
    )

    stage.process(task)

    assert 0.0 < task.data["error_rate"] < 20.0
    assert task.data["after"] == "日本語の文宇列"


@pytest.mark.parametrize(
    "data",
    [
        {_REF_KEY: "", _HYP_KEY: "text"},
        {_REF_KEY: "text", _HYP_KEY: ""},
        {_REF_KEY: "before", _HYP_KEY: "after", "_skipme": "other filter"},
    ],
)
def test_unscorable_or_previously_skipped_rows_record_sentinel(data: dict[str, str]) -> None:
    task = AudioTask(data=data)

    DisfluencyWerGuardStage().process(task)

    assert task.data[_RATE_KEY] == -1.0


def test_existing_sentinel_is_not_overwritten_on_skipped_row() -> None:
    task = AudioTask(data={_REF_KEY: "before", _HYP_KEY: "after", "_skipme": "other filter", _RATE_KEY: 12.5})

    DisfluencyWerGuardStage().process(task)

    assert task.data[_RATE_KEY] == 12.5


def test_declares_reference_compatible_io_contract() -> None:
    stage = DisfluencyWerGuardStage()

    assert stage.inputs() == ([], [_REF_KEY, _HYP_KEY])
    assert stage.outputs() == ([], [_HYP_KEY, _RATE_KEY])
