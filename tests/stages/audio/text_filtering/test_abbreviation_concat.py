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

from __future__ import annotations

import pytest

from nemo_curator.stages.audio.text_filtering.abbreviation_concat import concat_abbreviations


class TestConcatAbbreviations:
    """Tests for the concat_abbreviations function."""

    def test_basic_abbreviation(self):
        text, abbrevs = concat_abbreviations("the A P I uses A D X format")
        assert text == "the API uses ADX format"
        assert abbrevs == ["API", "ADX"]

    def test_possessive(self):
        text, abbrevs = concat_abbreviations("at the U K's major conference")
        assert text == "at the UK's major conference"
        assert abbrevs == ["UK"]

    def test_no_change(self):
        text, abbrevs = concat_abbreviations("nothing to do here")
        assert text == "nothing to do here"
        assert abbrevs == []


class TestUppercaseOnlyMatching:
    """Only uppercase letter sequences should be concatenated."""

    @pytest.mark.parametrize(
        "input_text",
        [
            "a B",
            "a b",
            "B a",
            "b a",
            "say a B word",
            "that b a thing",
        ],
    )
    def test_mixed_or_lowercase_not_concatenated(self, input_text):
        text, abbrevs = concat_abbreviations(input_text, language="en")
        assert text == input_text
        assert abbrevs == []

    def test_uppercase_pair_concatenates(self):
        text, abbrevs = concat_abbreviations("the A B format")
        assert text == "the AB format"
        assert abbrevs == ["AB"]

    def test_lowercase_letter_adjacent_to_uppercase_run(self):
        text, abbrevs = concat_abbreviations("the a P I thing")
        assert text == "the a PI thing"
        assert abbrevs == ["PI"]

    @pytest.mark.parametrize(
        ("input_text", "lang"),
        [
            ("e B", "it"),
            ("B e", "it"),
            ("a B", "it"),
            ("e B", "pt"),
            ("B e", "pt"),
            ("a B", "pt"),
            ("a B", "es"),
            ("B a", "es"),
        ],
    )
    def test_mixed_case_pairs_not_concatenated_any_language(self, input_text, lang):
        text, abbrevs = concat_abbreviations(input_text, language=lang)
        assert text == input_text
        assert abbrevs == []

    def test_uppercase_pair_concatenates_other_language(self):
        text, abbrevs = concat_abbreviations("the A B thing", language="de")
        assert text == "the AB thing"
        assert abbrevs == ["AB"]
