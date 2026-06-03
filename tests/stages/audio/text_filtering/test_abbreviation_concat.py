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

from nemo_curator.stages.audio.text_filtering.abbreviation_concat import (
    AbbreviationConcatStage,
    concat_abbreviations,
)
from nemo_curator.tasks import AudioTask


def test_basic_concat() -> None:
    result, found = concat_abbreviations("the A P I uses A D X format")
    assert result == "the API uses ADX format"
    assert found == ["API", "ADX"]


def test_possessive_preserved() -> None:
    result, found = concat_abbreviations("at the U K's major conference")
    assert result == "at the UK's major conference"
    assert "UK" in found


def test_no_concat_for_single_letter() -> None:
    result, _ = concat_abbreviations("I went home")
    assert result == "I went home"


def test_double_i_not_joined() -> None:
    result, _ = concat_abbreviations("I I think so")
    assert "I I" in result


def test_plural_abbreviation() -> None:
    result, found = concat_abbreviations("the C D s are here")
    assert "CDs" in result or "CD" in "".join(found)


def test_stage_process_basic() -> None:
    stage = AbbreviationConcatStage(text_key="text", output_text_key="abbr_text")
    task = AudioTask(data={"text": "the A P I is great", "source_lang": "en", "_skip_me": ""})
    result = stage.process(task)
    assert result.data["abbr_text"] == "the API is great"
    assert "API" in result.data["abbreviations"]


def test_stage_skips_flagged_task() -> None:
    stage = AbbreviationConcatStage(text_key="text", output_text_key="abbr_text")
    task = AudioTask(data={"text": "A P I", "source_lang": "en", "_skip_me": "Hallucination"})
    result = stage.process(task)
    assert result.data["abbr_text"] == ""
    assert result.data["abbreviations"] == []


def test_stage_handles_empty_text() -> None:
    stage = AbbreviationConcatStage(text_key="text", output_text_key="abbr_text")
    task = AudioTask(data={"text": "", "source_lang": "en", "_skip_me": ""})
    result = stage.process(task)
    assert result.data["abbr_text"] == ""


def test_stage_handles_non_string_text() -> None:
    stage = AbbreviationConcatStage(text_key="text", output_text_key="abbr_text")
    task = AudioTask(data={"text": None, "source_lang": "en", "_skip_me": ""})
    result = stage.process(task)
    assert result.data["abbr_text"] == ""


def test_process_batch() -> None:
    stage = AbbreviationConcatStage(text_key="text", output_text_key="abbr_text")
    tasks = [
        AudioTask(data={"text": "the U S A is large", "source_lang": "en", "_skip_me": ""}),
        AudioTask(data={"text": "no abbrevs here", "source_lang": "en", "_skip_me": ""}),
    ]
    results = stage.process_batch(tasks)
    assert "USA" in results[0].data["abbr_text"]
    assert results[1].data["abbr_text"] == "no abbrevs here"


def test_empty_batch() -> None:
    stage = AbbreviationConcatStage()
    assert stage.process_batch([]) == []


def test_german_characters() -> None:
    result, found = concat_abbreviations("die Ä B C Methode", language="de")
    assert "ÄBC" in result or len(found) > 0


def test_contraction_trailing_i() -> None:
    result, _ = concat_abbreviations("the A P I'm sure")
    assert "I'm" in result or "I" in result
