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

from nemo_curator.stages.text.filters.heuristic.string import (
    BulletsFilter,
    EllipsisFilter,
    LongWordFilter,
    MeanWordLengthFilter,
    PunctuationFilter,
    SymbolsToWordsFilter,
    WordsWithoutAlphabetsFilter,
)


# Inputs that previously caused crashes (ZeroDivisionError / ValueError)
EMPTY_INPUTS = [
    "",           # empty string
    " ",          # single space
    "\n\n",       # newlines only
    "\xa0",       # non-breaking space
    "  \t  \n  ", # mixed whitespace
]


class TestSymbolsToWordsFilter:
    @pytest.mark.parametrize("text", EMPTY_INPUTS)
    def test_empty_input_does_not_crash(self, text):
        filt = SymbolsToWordsFilter()
        score = filt.score_document(text)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Empty text should get worst score (1.0) to be filtered out
        assert score == 1.0

    def test_normal_text_unchanged(self):
        filt = SymbolsToWordsFilter()
        # Normal text should not raise and produce a valid score
        score = filt.score_document("Hello world, this is a normal text.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestBulletsFilter:
    @pytest.mark.parametrize("text", EMPTY_INPUTS)
    def test_empty_input_does_not_crash(self, text):
        filt = BulletsFilter()
        score = filt.score_document(text)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == 1.0

    def test_normal_text_unchanged(self):
        filt = BulletsFilter()
        score = filt.score_document("First sentence. Second sentence. Third sentence.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestLongWordFilter:
    @pytest.mark.parametrize("text", EMPTY_INPUTS)
    def test_empty_input_does_not_crash(self, text):
        filt = LongWordFilter()
        score = filt.score_document(text)
        assert isinstance(score, (int, float))
        assert score == 0

    def test_normal_text_unchanged(self):
        filt = LongWordFilter()
        score = filt.score_document("Hello world.")
        assert isinstance(score, (int, float))
        assert score >= 0


class TestMeanWordLengthFilter:
    @pytest.mark.parametrize("text", EMPTY_INPUTS)
    def test_empty_input_does_not_crash(self, text):
        filt = MeanWordLengthFilter()
        score = filt.score_document(text)
        assert isinstance(score, float)
        assert score == 0.0

    def test_normal_text_unchanged(self):
        filt = MeanWordLengthFilter()
        score = filt.score_document("Hello world this is a test")
        assert isinstance(score, float)
        assert score > 0


class TestPunctuationFilter:
    @pytest.mark.parametrize("text", EMPTY_INPUTS)
    def test_empty_input_does_not_crash(self, text):
        filt = PunctuationFilter()
        score = filt.score_document(text)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == 1.0

    def test_normal_text_unchanged(self):
        filt = PunctuationFilter()
        score = filt.score_document("Hello world. How are you? I am fine!")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestEllipsisFilter:
    @pytest.mark.parametrize("text", EMPTY_INPUTS)
    def test_empty_input_does_not_crash(self, text):
        filt = EllipsisFilter()
        score = filt.score_document(text)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == 1.0

    def test_normal_text_unchanged(self):
        filt = EllipsisFilter()
        score = filt.score_document("Hello world. How are you?")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestWordsWithoutAlphabetsFilter:
    @pytest.mark.parametrize("text", EMPTY_INPUTS)
    def test_empty_input_does_not_crash(self, text):
        filt = WordsWithoutAlphabetsFilter()
        score = filt.score_document(text)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Higher = better for this filter; empty text = 0.0 (worst)
        assert score == 0.0

    def test_normal_text_unchanged(self):
        filt = WordsWithoutAlphabetsFilter()
        score = filt.score_document("Hello world 123 test")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0
