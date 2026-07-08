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

from detect_glued_oov_llm import (
    build_prompt,
    is_junk_oov,
    parse_llm_json,
    prefilter_candidates,
    write_repairs_tsv,
)


def test_is_junk_oov():
    assert is_junk_oov("'a")
    assert is_junk_oov("spn")
    assert not is_junk_oov("abandonedand")


def test_prefilter_candidates_boundary():
    dictionary = {"abandoned", "and", "abby", "thanks", "aardvark"}
    candidates = prefilter_candidates(
        ["abandonedand", "abbythanks", "aardvark", "'a"],
        dictionary,
        mode="unglue_and_boundary",
    )
    words = {word for word, _ in candidates}
    assert "abandonedand" in words
    assert "abbythanks" in words
    assert "aardvark" not in words
    assert "'a" not in words


def test_parse_llm_json():
    raw = '{"results":[{"word":"abandonedand","glued":true,"replacement":"abandoned and","confidence":0.98,"reason":"two words"}]}'
    rows = parse_llm_json(raw)
    assert len(rows) == 1
    assert rows[0]["word"] == "abandonedand"


def test_write_repairs_tsv_filters_confidence(tmp_path):
    rows = [
        {
            "word": "abandonedand",
            "replacement": "abandoned and",
            "confidence": 0.98,
            "glued": True,
            "reason": "ok",
            "heuristic": "abandoned and",
        },
        {
            "word": "maybe",
            "replacement": "may be",
            "confidence": 0.70,
            "glued": True,
            "reason": "low",
            "heuristic": "",
        },
    ]
    out = tmp_path / "repairs.tsv"
    count = write_repairs_tsv(out, rows, min_confidence=0.95)
    assert count == 1
    assert out.read_text(encoding="utf-8") == "abandonedand\tabandoned and\n"


def test_build_prompt_includes_heuristic():
    prompt = build_prompt([("abandonedand", "abandoned and")])
    assert "abandonedand" in prompt
    assert "abandoned and" in prompt
