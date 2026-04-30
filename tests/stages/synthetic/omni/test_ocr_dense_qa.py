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

"""Unit tests for nemo_curator.stages.synthetic.omni.ocr_dense_qa."""

import random
from pathlib import Path

import pytest

from nemo_curator.stages.synthetic.omni.ocr_dense_qa import (
    MAX_QA_PAIRS,
    QA_TYPE_BBOX_TO_TEXT,
    QA_TYPE_POINT_TO_TEXT,
    QA_TYPE_TEXT_TO_BBOX,
    QA_TYPE_TEXT_TO_POINT,
    _balanced_sample_qa,
    _bbox_center,
    _bbox_dist_from_center,
    _escape_text_for_prompt,
    _fmt_box,
    build_conversation,
    build_dense_conversation,
    build_qa_tagged,
)
from nemo_curator.stages.synthetic.omni.utils.conversation import ImageMedia
from nemo_curator.tasks.ocr import OCRData, OCRDenseWord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_word(
    bbox: list[int],
    text: str,
    *,
    valid: bool = True,
    bbox_match: int | None = None,
    text_errors: int | None = None,
) -> OCRDenseWord:
    return OCRDenseWord(
        bbox_2d=bbox,
        text_content=text,
        valid=valid,
        bbox_match=bbox_match,
        text_errors=text_errors,
    )


def _make_ocr_data(words: list[OCRDenseWord]) -> OCRData:
    return OCRData(
        image_path=Path("test.jpg"),
        image_id="test_id",
        ocr_dense=words,
    )


def _fixed_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)  # noqa: S311


# ---------------------------------------------------------------------------
# _fmt_box
# ---------------------------------------------------------------------------


class TestFmtBox:
    def test_list_input(self):
        assert _fmt_box([10, 20, 30, 40]) == "[10, 20, 30, 40]"

    def test_tuple_input(self):
        assert _fmt_box((0, 0, 100, 200)) == "[0, 0, 100, 200]"

    def test_zero_box(self):
        assert _fmt_box([0, 0, 0, 0]) == "[0, 0, 0, 0]"


# ---------------------------------------------------------------------------
# _bbox_center
# ---------------------------------------------------------------------------


class TestBboxCenter:
    def test_simple(self):
        cx, cy = _bbox_center([0, 0, 100, 200])
        assert cx == 50
        assert cy == 100

    def test_already_centered(self):
        cx, cy = _bbox_center([490, 490, 510, 510])
        assert cx == 500
        assert cy == 500

    def test_integer_truncation(self):
        # (10 + 11) // 2 == 10  (integer division, not rounding)
        cx, cy = _bbox_center([10, 10, 11, 11])
        assert cx == 10
        assert cy == 10


# ---------------------------------------------------------------------------
# _bbox_dist_from_center
# ---------------------------------------------------------------------------


class TestBboxDistFromCenter:
    def test_center_of_image(self):
        # bbox centered exactly at (500, 500)
        dist = _bbox_dist_from_center([490, 490, 510, 510])
        assert dist == pytest.approx(0.0, abs=1.0)

    def test_corner(self):
        dist = _bbox_dist_from_center([0, 0, 0, 0])
        # center of [0,0,0,0] is (0,0); dist from (500,500) = sqrt(500²+500²)
        expected = (500**2 + 500**2) ** 0.5
        assert dist == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# _escape_text_for_prompt
# ---------------------------------------------------------------------------


class TestEscapeTextForPrompt:
    def test_double_quotes_in_text_uses_single_quotes(self):
        rng = random.Random(0)  # noqa: S311
        result = _escape_text_for_prompt('say "hello"', rng)
        assert result.startswith("'")
        assert result.endswith("'")

    def test_single_quotes_in_text_uses_double_quotes(self):
        rng = random.Random(0)  # noqa: S311
        result = _escape_text_for_prompt("it's fine", rng)
        assert result.startswith('"')
        assert result.endswith('"')

    def test_normal_text_wrapped_in_quotes(self):
        rng = random.Random(0)  # noqa: S311
        result = _escape_text_for_prompt("hello", rng)
        assert (result.startswith("'") and result.endswith("'")) or (result.startswith('"') and result.endswith('"'))
        assert "hello" in result

    def test_uppercase_alpha_may_pass_through(self):
        # All-uppercase alpha text has 50% chance of returning raw.
        # Use a large sample to verify both outcomes are possible.
        results = {_escape_text_for_prompt("ABC", random.Random(i)) for i in range(50)}  # noqa: S311
        # At least one result should be unquoted
        assert "ABC" in results
        # At least one result should be quoted
        assert any(r != "ABC" for r in results)


# ---------------------------------------------------------------------------
# _balanced_sample_qa
# ---------------------------------------------------------------------------


class TestBalancedSampleQa:
    def _make_tagged(self, types_and_counts: dict[str, int]) -> list[tuple[str, str, str]]:
        tagged = []
        for typ, count in types_and_counts.items():
            for i in range(count):
                tagged.append((typ, f"q_{typ}_{i}", f"a_{typ}_{i}"))
        return tagged

    def test_returns_all_when_under_limit(self):
        tagged = self._make_tagged({QA_TYPE_BBOX_TO_TEXT: 3, QA_TYPE_POINT_TO_TEXT: 2})
        result = _balanced_sample_qa(tagged, max_pairs=10, rng=_fixed_rng())
        assert len(result) == 5

    def test_caps_at_max_pairs(self):
        tagged = self._make_tagged(
            {
                QA_TYPE_BBOX_TO_TEXT: 50,
                QA_TYPE_POINT_TO_TEXT: 50,
                QA_TYPE_TEXT_TO_BBOX: 50,
                QA_TYPE_TEXT_TO_POINT: 50,
            }
        )
        result = _balanced_sample_qa(tagged, max_pairs=MAX_QA_PAIRS, rng=_fixed_rng())
        assert len(result) == MAX_QA_PAIRS

    def test_balanced_across_types(self):
        tagged = self._make_tagged(
            {
                QA_TYPE_BBOX_TO_TEXT: 100,
                QA_TYPE_POINT_TO_TEXT: 100,
            }
        )
        result = _balanced_sample_qa(tagged, max_pairs=10, rng=_fixed_rng())
        assert len(result) == 10
        # With 2 equal-size buckets and 10 total, each gets ~5
        q_set = {q for q, _ in result}
        assert any("bbox_to_text" in q for q in q_set)
        assert any("point_to_text" in q for q in q_set)

    def test_fills_from_leftover_when_one_bucket_is_small(self):
        # 1 item in type A, 100 in type B; request 10 → should still return 10
        tagged = self._make_tagged({QA_TYPE_BBOX_TO_TEXT: 1, QA_TYPE_POINT_TO_TEXT: 100})
        result = _balanced_sample_qa(tagged, max_pairs=10, rng=_fixed_rng())
        assert len(result) == 10

    def test_empty_tagged_returns_empty(self):
        result = _balanced_sample_qa([], max_pairs=10, rng=_fixed_rng())
        assert result == []

    def test_deterministic_with_same_seed(self):
        tagged = self._make_tagged({QA_TYPE_BBOX_TO_TEXT: 20, QA_TYPE_POINT_TO_TEXT: 20})
        r1 = _balanced_sample_qa(tagged, max_pairs=5, rng=random.Random(99))  # noqa: S311
        r2 = _balanced_sample_qa(tagged, max_pairs=5, rng=random.Random(99))  # noqa: S311
        assert r1 == r2


# ---------------------------------------------------------------------------
# build_qa_tagged
# ---------------------------------------------------------------------------


class TestBuildQaTagged:
    def test_empty_words_returns_empty(self):
        data = _make_ocr_data([])
        qa_tagged, _ = build_qa_tagged(data, "task_0")
        assert qa_tagged == []

    def test_all_invalid_words_returns_empty(self):
        words = [_make_word([0, 0, 100, 100], "hello", valid=False)]
        data = _make_ocr_data(words)
        qa_tagged, _ = build_qa_tagged(data, "task_0")
        assert qa_tagged == []

    def test_valid_words_produce_tagged_pairs(self):
        words = [
            _make_word([10, 10, 100, 50], "HELLO"),
            _make_word([200, 10, 400, 50], "WORLD"),
        ]
        data = _make_ocr_data(words)
        qa_tagged, _ = build_qa_tagged(data, "task_1")
        assert len(qa_tagged) > 0
        # Each entry is (type, question, answer)
        for typ, q, a in qa_tagged:
            assert typ in {
                QA_TYPE_BBOX_TO_TEXT,
                QA_TYPE_POINT_TO_TEXT,
                QA_TYPE_TEXT_TO_BBOX,
                QA_TYPE_TEXT_TO_POINT,
            }
            assert isinstance(q, str)
            assert q
            assert isinstance(a, str)
            assert a

    def test_many_invalids_disables_text_to_bbox(self):
        # 5 invalid → allow_text_to_bbox=False (< 5 threshold is strict)
        words = [_make_word([i * 10, 0, (i + 1) * 10, 50], f"w{i}", valid=False) for i in range(5)]
        words += [_make_word([500, 0, 600, 50], "TEXT", valid=True)]
        data = _make_ocr_data(words)
        qa_tagged, _ = build_qa_tagged(data, "task_2")
        types_used = {typ for typ, _, _ in qa_tagged}
        assert QA_TYPE_TEXT_TO_BBOX not in types_used
        assert QA_TYPE_TEXT_TO_POINT not in types_used

    def test_reproducible_with_same_task_id(self):
        words = [
            _make_word([10, 10, 100, 50], "HELLO"),
            _make_word([200, 10, 400, 50], "WORLD"),
        ]
        data = _make_ocr_data(words)
        tagged1, _ = build_qa_tagged(data, "same_task")
        tagged2, _ = build_qa_tagged(data, "same_task")
        assert tagged1 == tagged2

    def test_different_task_ids_may_differ(self):
        words = [_make_word([i * 100, 0, (i + 1) * 100, 50], f"word{i}") for i in range(10)]
        data = _make_ocr_data(words)
        tagged_a, _ = build_qa_tagged(data, "task_a")
        tagged_b, _ = build_qa_tagged(data, "task_b")
        # Different seeds → different type selections (with high probability)
        types_a = [t for t, _, _ in tagged_a]
        types_b = [t for t, _, _ in tagged_b]
        # They may or may not differ, but the RNG is seeded differently
        assert types_a != types_b or tagged_a[0][1] != tagged_b[0][1]


# ---------------------------------------------------------------------------
# build_conversation
# ---------------------------------------------------------------------------


class TestBuildConversation:
    def test_returns_none_for_empty_pairs(self):
        result = build_conversation([], _fixed_rng(), "img.jpg")
        assert result is None

    def test_first_user_message_has_image_media(self):
        qa_tagged = [(QA_TYPE_BBOX_TO_TEXT, "question?", "answer")]
        conv = build_conversation(qa_tagged, _fixed_rng(), "my_image.jpg")
        first_msg = conv.conversation[0]
        assert first_msg.sender == "user"
        assert any(isinstance(frag, ImageMedia) for frag in first_msg.fragments)
        img_frag = next(f for f in first_msg.fragments if isinstance(f, ImageMedia))
        assert img_frag.value == "my_image.jpg"

    def test_alternating_user_assistant_pattern(self):
        qa_tagged = [(QA_TYPE_BBOX_TO_TEXT, f"q{i}", f"a{i}") for i in range(5)]
        conv = build_conversation(qa_tagged, _fixed_rng(), "img.jpg")
        for i, msg in enumerate(conv.conversation):
            expected_sender = "user" if i % 2 == 0 else "assistant"
            assert msg.sender == expected_sender, f"Message {i} has wrong sender"

    def test_caps_at_max_qa_pairs(self):
        qa_tagged = [(QA_TYPE_BBOX_TO_TEXT, f"q{i}", f"a{i}") for i in range(200)]
        conv = build_conversation(qa_tagged, _fixed_rng(), "img.jpg")
        # Each QA pair → 2 messages; total messages = 2 * actual_pairs
        actual_pairs = len(conv.conversation) // 2
        assert actual_pairs <= MAX_QA_PAIRS

    def test_serializable_to_dict(self):
        qa_tagged = [(QA_TYPE_BBOX_TO_TEXT, "What text?", "HELLO")]
        conv = build_conversation(qa_tagged, _fixed_rng(), "img.jpg")
        d = conv.to_dict()
        assert "conversation" in d
        assert isinstance(d["conversation"], list)
        assert d["conversation"][0]["sender"] == "user"


# ---------------------------------------------------------------------------
# build_dense_conversation
# ---------------------------------------------------------------------------


class TestBuildDenseConversation:
    def test_single_turn(self):
        words = [_make_word([10, 10, 100, 50], "TEXT")]
        conv = build_dense_conversation(words, _fixed_rng(), "img.jpg")
        assert len(conv.conversation) == 2
        assert conv.conversation[0].sender == "user"
        assert conv.conversation[1].sender == "assistant"

    def test_first_message_has_image_media(self):
        words = [_make_word([10, 10, 100, 50], "TEXT")]
        conv = build_dense_conversation(words, _fixed_rng(), "my_img.jpg")
        first_msg = conv.conversation[0]
        assert any(isinstance(f, ImageMedia) for f in first_msg.fragments)
        img_frag = next(f for f in first_msg.fragments if isinstance(f, ImageMedia))
        assert img_frag.value == "my_img.jpg"

    def test_answer_contains_word_text(self):
        words = [_make_word([10, 10, 100, 50], "UNIQUEWORD123")]
        conv = build_dense_conversation(words, _fixed_rng(), "img.jpg")
        answer = conv.conversation[1].fragments[0]
        assert isinstance(answer, str)
        assert "UNIQUEWORD123" in answer
