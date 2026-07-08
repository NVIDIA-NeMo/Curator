# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from detect_glued_oov_heuristic import (
    best_segmentation,
    detect_glued,
    is_candidate_token,
    score_split,
    write_mapping,
)

DICT = {
    "abandoned", "and", "abby", "thanks", "ability", "to",
    "leather", "worker", "abba", "dele", "region", "the",
    "aardvark", "have",
}


def test_is_candidate_token():
    assert is_candidate_token("abandonedand")
    assert not is_candidate_token("'a")  # apostrophe -> not plain alpha
    assert not is_candidate_token("abc")  # too short
    assert not is_candidate_token("lee-enfield")  # hyphen -> explicit boundary


def test_glue_word_split_high_confidence():
    parts = best_segmentation("abandonedand", DICT)
    assert parts == ["abandoned", "and"]
    confidence, pattern = score_split(parts)
    assert pattern == "glue_word"
    assert confidence >= 0.95


def test_two_content_words_requires_long_parts():
    parts = best_segmentation("leatherworker", DICT)
    assert parts == ["leather", "worker"]
    confidence, pattern = score_split(parts)
    assert pattern == "two_content_words"
    assert confidence >= 0.90


def test_short_leading_article_is_downgraded():
    # "a" + 4-letter fragment is a coincidental name split, not high confidence.
    parts = best_segmentation("aabba", DICT)
    assert parts == ["a", "abba"]
    confidence, pattern = score_split(parts)
    assert pattern == "glue_word_short"
    assert confidence < 0.90


def test_real_word_not_glued():
    assert best_segmentation("aardvark", DICT) is None


def test_detect_and_mapping_roundtrip(tmp_path):
    rows = detect_glued(
        ["abandonedand", "leatherworker", "aabba", "aardvark", "regionthe"],
        DICT,
    )
    words = {word for word, *_ in rows}
    assert "abandonedand" in words
    assert "leatherworker" in words
    assert "aardvark" not in words

    out = tmp_path / "map.tsv"
    count = write_mapping(out, rows, min_confidence=0.90)
    text = out.read_text(encoding="utf-8")
    assert "abandonedand\tabandoned and\n" in text
    assert "aabba" not in text  # downgraded below threshold
    assert count >= 2
