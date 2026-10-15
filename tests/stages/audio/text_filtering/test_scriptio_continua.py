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

from collections.abc import Callable

import pytest

from nemo_curator.stages.audio.text_filtering.scriptio_continua import is_scriptio_continua
from nemo_curator.stages.audio.text_filtering.text_metrics import (
    character_error_rate_percent,
    word_error_rate_percent,
)


@pytest.mark.parametrize(
    "language",
    ["ja", "zh", "zh-CN", "zh_tw", "zh-Hans", "zh-hant", "yue", "th", "lo", "km", "my", "bo", "dz"],
)
def test_reference_no_space_codes_are_recognized(language: str) -> None:
    assert is_scriptio_continua(language)


@pytest.mark.parametrize("language", ["Japanese", "CHINESE", "Thai"])
def test_full_language_names_are_recognized(language: str) -> None:
    assert is_scriptio_continua(language)


@pytest.mark.parametrize("language", [None, "", "en", "ko", "vi", "ja-JP", "Cantonese", "Lao", "Khmer", "unknown"])
def test_other_or_missing_languages_are_not_recognized(language: str | None) -> None:
    assert not is_scriptio_continua(language)


def test_word_and_character_metrics_diverge_for_unspaced_near_match() -> None:
    reference = "这是一个测试句子用来检查协议"
    hypothesis = "这是一个测试句子用来检查协义"

    assert word_error_rate_percent(reference, hypothesis) == 100.0
    assert character_error_rate_percent(reference, hypothesis) == 7.14


@pytest.mark.parametrize(
    ("metric", "reference", "hypothesis", "expected"),
    [
        (word_error_rate_percent, "", "", 0.0),
        (word_error_rate_percent, "", "text", 100.0),
        (character_error_rate_percent, "", "", 0.0),
        (character_error_rate_percent, "", "字", 100.0),
    ],
)
def test_empty_reference_metric_contract(
    metric: Callable[[str, str], float], reference: str, hypothesis: str, expected: float
) -> None:
    assert metric(reference, hypothesis) == expected
