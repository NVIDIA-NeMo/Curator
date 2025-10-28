# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latex_augment.sentences import split_sentences


def test_basic_sentence_splitting():
    text = " This is sentence one.  This is sentence two.\n "
    sentences = list(split_sentences(text))
    assert len(sentences) == 2
    assert text[sentences[0]] == "This is sentence one."
    assert text[sentences[1]] == "This is sentence two."


def test_abbreviations():
    text = (
        "Dr. Smith and Prof. Jones wrote ref. 1 and Fig. 2 in Sec. 3. "
        "Next sentence. "
        "See Doe et al. (in prep.) for references."
    )
    sentences = list(split_sentences(text))
    assert len(sentences) == 3
    assert (
        text[sentences[0]]
        == "Dr. Smith and Prof. Jones wrote ref. 1 and Fig. 2 in Sec. 3."
    )
    assert text[sentences[1]] == "Next sentence."
    assert text[sentences[2]] == "See Doe et al. (in prep.) for references."


def test_whitespace():
    text = "First sentence.\n\nSecond sentence."
    sentences = list(split_sentences(text))
    assert len(sentences) == 2
    assert text[sentences[0]] == "First sentence."
    assert text[sentences[1]] == "Second sentence."


def test_whitespace2():
    text = "                         or                     or~                "
    sentences = list(split_sentences(text))
    assert len(sentences) == 1
    assert text[sentences[0]] == "or                     or~"


def test_leading_whitespace():
    text = "\n\n  First sentence. Second sentence."
    sentences = list(split_sentences(text))
    assert len(sentences) == 2
    assert text[sentences[0]] == "First sentence."
    assert text[sentences[1]] == "Second sentence."


def test_only_whitespace():
    text = "   \n\n   "
    sentences = list(split_sentences(text))
    assert len(sentences) == 0


def test_unicode():
    text = "Hello, 世界! This is unicode."
    sentences = list(split_sentences(text))
    assert len(sentences) == 2
    assert text[sentences[0]] == "Hello, 世界!"
    assert text[sentences[1]] == "This is unicode."


def test_single_sentence():
    text = "Just one sentence without period"
    sentences = list(split_sentences(text))
    assert len(sentences) == 1
    assert text[sentences[0]] == text


def test_empty_input():
    text = ""
    sentences = list(split_sentences(text))
    assert len(sentences) == 0
