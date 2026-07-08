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

from collections import Counter

from david_ai_glued_words import (
    apply_repair_map_to_text,
    build_unglue_repair_map,
    repair_normalized_text_two_step,
    repair_punctuation_only,
    separate_gluing_punctuation,
    try_unglue_word,
)


def test_separate_gluing_punctuation():
    assert separate_gluing_punctuation("about...but") == "about . . . but"
    assert separate_gluing_punctuation("talking—right") == "talking — right"
    assert separate_gluing_punctuation("don't-stop") == "don't-stop"
    assert separate_gluing_punctuation("well-known,don't") == "well-known , don't"


def test_try_unglue_word_splits_glued_tokens():
    dictionary = {"about", "but", "chaos", "chef's", "christmas", "seventy", "sand"}
    assert try_unglue_word("aboutbut", dictionary) == ["about", "but"]
    assert try_unglue_word("aboutchaos", dictionary) == ["about", "chaos"]
    assert try_unglue_word("seventysand", dictionary) == ["seventy", "sand"]
    assert try_unglue_word("about", dictionary) is None


def test_build_unglue_repair_map_uses_frequency_threshold():
    dictionary = {"about", "but", "chaos"}
    word_freq = Counter({"aboutbut": 2, "aboutchaos": 10})
    repairs = build_unglue_repair_map(word_freq, dictionary, max_freq=5)
    assert repairs == {"aboutbut": "about but"}
    assert "aboutchaos" not in repairs


def test_apply_repair_map_to_text():
    repair_map = {"aboutbut": "about but"}
    assert apply_repair_map_to_text("i was aboutbut sure", repair_map) == "i was about but sure"


def test_repair_punctuation_only_leaves_unglued_tokens():
    repaired, changed = repair_punctuation_only(
        "about...but",
        "aboutbut",
        num2words_lang="en",
    )
    assert changed
    assert "about" in repaired
    assert "but" in repaired


def test_resolve_unglue_repairs_path_prefers_explicit(tmp_path):
    from david_ai_glued_words import load_lexicon_unglue_repairs, resolve_unglue_repairs_path

    heuristic = tmp_path / "unglue_repairs_heuristic.tsv"
    heuristic.write_text("aboutbut\tabout but\n", encoding="utf-8")
    default = tmp_path / "unglue_repairs.tsv"
    default.write_text("other\tother word\n", encoding="utf-8")

    assert resolve_unglue_repairs_path(tmp_path).name == "unglue_repairs.tsv"
    assert resolve_unglue_repairs_path(tmp_path, explicit=heuristic).name == "unglue_repairs_heuristic.tsv"

    repairs, path = load_lexicon_unglue_repairs(tmp_path)
    assert path == default.resolve()
    assert repairs == {"other": "other word"}

    repairs, path = load_lexicon_unglue_repairs(tmp_path, explicit=heuristic)
    assert repairs == {"aboutbut": "about but"}


def test_repair_normalized_text_two_step_unglue_only():
    repaired, punctuation_changed, unglue_changed = repair_normalized_text_two_step(
        "",
        "i was aboutbut sure",
        repair_map={"aboutbut": "about but"},
    )
    assert repaired == "i was about but sure"
    assert not punctuation_changed
    assert unglue_changed
