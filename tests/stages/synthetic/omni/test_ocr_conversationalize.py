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

"""Unit tests for nemo_curator.stages.synthetic.omni.ocr_conversationalize."""

import json
from pathlib import Path

import pytest

from nemo_curator.stages.synthetic.omni.ocr_conversationalize import (
    SDG_PROMPT_VARIATIONS,
    WORD_OUTPUT_FORMATS,
    OCRConversationData,
    _fmt_json_markdown,
    _fmt_json_plain,
    _fmt_markdown_table,
    _fmt_text_bracket,
    _fmt_text_per_line,
    _fmt_text_tuple,
    _fmt_tsv,
)
from nemo_curator.stages.synthetic.omni.utils.conversation import (
    ConversationSample,
    ImageMedia,
    Message,
)
from nemo_curator.tasks.ocr import OCRDenseItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_word(bbox: list[int], text: str) -> OCRDenseItem:
    return OCRDenseItem(bbox_2d=bbox, text_content=text)


_WORD = _make_word([10, 20, 100, 50], "HELLO")
_WORDS = [_make_word([10, 20, 100, 50], "HELLO"), _make_word([200, 20, 300, 50], "WORLD")]


# ---------------------------------------------------------------------------
# SDG_PROMPT_VARIATIONS
# ---------------------------------------------------------------------------


class TestSdgPromptVariations:
    def test_non_empty(self):
        assert len(SDG_PROMPT_VARIATIONS) > 0

    def test_all_strings(self):
        for v in SDG_PROMPT_VARIATIONS:
            assert isinstance(v, str)
            assert v


# ---------------------------------------------------------------------------
# WORD_OUTPUT_FORMATS — structural contract for every format function
# ---------------------------------------------------------------------------


class TestWordOutputFormats:
    def test_all_functions_registered(self):
        assert len(WORD_OUTPUT_FORMATS) == 11

    @pytest.mark.parametrize("fmt_fn", WORD_OUTPUT_FORMATS)
    def test_returns_two_strings(self, fmt_fn: object) -> None:
        suffix, answer = fmt_fn([_WORD])  # type: ignore[operator]
        assert isinstance(suffix, str)
        assert suffix
        assert isinstance(answer, str)
        assert answer

    @pytest.mark.parametrize("fmt_fn", WORD_OUTPUT_FORMATS)
    def test_answer_contains_word_text(self, fmt_fn: object) -> None:
        _, answer = fmt_fn([_WORD])  # type: ignore[operator]
        assert "HELLO" in answer

    @pytest.mark.parametrize("fmt_fn", WORD_OUTPUT_FORMATS)
    def test_answer_contains_bbox_coords(self, fmt_fn: object) -> None:
        _, answer = fmt_fn([_WORD])  # type: ignore[operator]
        # bbox is [10, 20, 100, 50] — at least one coord must appear
        assert any(str(c) in answer for c in [10, 20, 100, 50])

    @pytest.mark.parametrize("fmt_fn", WORD_OUTPUT_FORMATS)
    def test_multiple_words_all_present(self, fmt_fn: object) -> None:
        _, answer = fmt_fn(_WORDS)  # type: ignore[operator]
        assert "HELLO" in answer
        assert "WORLD" in answer


# ---------------------------------------------------------------------------
# Individual format functions — format-specific contracts
# ---------------------------------------------------------------------------


class TestFmtJsonPlain:
    def test_answer_is_valid_json(self):
        _, answer = _fmt_json_plain([_WORD])
        parsed = json.loads(answer)
        assert isinstance(parsed, list)
        assert parsed[0]["text_content"] == "HELLO"
        assert parsed[0]["bbox_2d"] == [10, 20, 100, 50]

    def test_no_markdown_fences(self):
        _, answer = _fmt_json_plain([_WORD])
        assert "```" not in answer


class TestFmtJsonMarkdown:
    def test_answer_wrapped_in_code_fence(self):
        _, answer = _fmt_json_markdown([_WORD])
        assert answer.startswith("```json")
        assert answer.endswith("```")

    def test_inner_content_is_valid_json(self):
        _, answer = _fmt_json_markdown([_WORD])
        inner = answer.removeprefix("```json\n").removesuffix("\n```")
        parsed = json.loads(inner)
        assert isinstance(parsed, list)


class TestFmtTextPerLine:
    def test_one_line_per_word(self):
        _, answer = _fmt_text_per_line(_WORDS)
        lines = answer.strip().splitlines()
        assert len(lines) == 2

    def test_line_format_text_then_bbox(self):
        _, answer = _fmt_text_per_line([_WORD])
        assert answer.startswith("HELLO")
        assert "[10, 20, 100, 50]" in answer


class TestFmtTextBracket:
    def test_bbox_before_text(self):
        _, answer = _fmt_text_bracket([_WORD])
        assert answer.startswith("[10, 20, 100, 50]")
        assert "HELLO" in answer


class TestFmtTextTuple:
    def test_tuple_notation_used(self):
        _, answer = _fmt_text_tuple([_WORD])
        assert "(10, 20, 100, 50)" in answer


class TestFmtMarkdownTable:
    def test_has_table_header(self):
        _, answer = _fmt_markdown_table([_WORD])
        assert "| text |" in answer
        assert "| bbox |" in answer

    def test_word_in_table_body(self):
        _, answer = _fmt_markdown_table([_WORD])
        assert "'HELLO'" in answer or "HELLO" in answer


class TestFmtTsv:
    def test_tab_separated(self):
        _, answer = _fmt_tsv([_WORD])
        parts = answer.split("\t")
        assert len(parts) == 5  # text x1 y1 x2 y2
        assert parts[0] == "HELLO"
        assert parts[1] == "10"


# ---------------------------------------------------------------------------
# OCRConversationData.to_dict
# ---------------------------------------------------------------------------


class TestOCRConversationDataToDict:
    def _make_data(self, *, conversation: ConversationSample | None = None) -> OCRConversationData:
        return OCRConversationData(
            image_path=Path("test.jpg"),
            image_id="id0",
            conversation=conversation,
        )

    def test_conversation_none_serializes_as_none(self):
        d = self._make_data().to_dict()
        assert d["conversation"] is None

    def test_image_fields_present(self):
        d = self._make_data().to_dict()
        assert "image_path" in d
        assert "image_id" in d

    def test_conversation_uses_to_dict_not_asdict(self):
        # dataclasses.asdict would lose the "t" type discriminator on media
        # fragments; to_dict() overrides with conversation.to_dict() to preserve it.
        conv = ConversationSample(
            conversation=[Message(sender="user", fragments=[ImageMedia(value="img.jpg")])]
        )
        d = self._make_data(conversation=conv).to_dict()

        # The override produces {"conversation": [...]}, not the raw asdict result
        assert isinstance(d["conversation"], dict)
        assert "conversation" in d["conversation"]

        frag = d["conversation"]["conversation"][0]["fragments"][0]
        assert frag.get("t") == "image"   # "t" discriminator preserved
        assert frag.get("value") == "img.jpg"

    def test_ocr_dense_serialized(self):
        data = OCRConversationData(
            image_path=Path("test.jpg"),
            image_id="id0",
            ocr_dense=[_WORD],
        )
        d = data.to_dict()
        assert isinstance(d["ocr_dense"], list)
        assert d["ocr_dense"][0]["text_content"] == "HELLO"


# ---------------------------------------------------------------------------
# OCRConversationData.from_dict
# ---------------------------------------------------------------------------


class TestOCRConversationDataFromDict:
    def _base_record(self) -> dict:
        return {
            "image_path": "test.jpg",
            "image_id": "id0",
            "is_valid": True,
            "error": None,
            "ocr_is_word_level": True,
            "ocr_dense_prompt": None,
            "ocr_dense": None,
            "ocr_scoring_prompt": None,
            "ocr_scoring_model": None,
            "ocr_scoring_response_raw": None,
            "ocr_scoring_mode": None,
            "ocr_scoring_missing": None,
        }

    def test_no_conversation_key_gives_none(self):
        data = OCRConversationData.from_dict(self._base_record())
        assert data.conversation is None

    def test_explicit_none_conversation_gives_none(self):
        rec = self._base_record()
        rec["conversation"] = None
        data = OCRConversationData.from_dict(rec)
        assert data.conversation is None

    def test_base_fields_preserved(self):
        data = OCRConversationData.from_dict(self._base_record())
        assert data.image_id == "id0"
        assert data.is_valid is True
        assert data.ocr_is_word_level is True

    def test_ocr_dense_deserialized(self):
        rec = self._base_record()
        rec["ocr_dense"] = [{"bbox_2d": [10, 20, 100, 50], "text_content": "HELLO"}]
        data = OCRConversationData.from_dict(rec)
        assert data.ocr_dense is not None
        assert len(data.ocr_dense) == 1
        assert data.ocr_dense[0].text_content == "HELLO"

    def test_round_trip_with_conversation(self):
        conv = ConversationSample(
            conversation=[Message(sender="user", fragments=["Describe the image"])]
        )
        original = OCRConversationData(
            image_path=Path("test.jpg"),
            image_id="id0",
            conversation=conv,
        )
        data = OCRConversationData.from_dict(original.to_dict())
        assert data.conversation is not None
        assert len(data.conversation.conversation) == 1
        assert data.conversation.conversation[0].sender == "user"

    def test_round_trip_preserves_image_media_discriminator(self):
        conv = ConversationSample(
            conversation=[Message(sender="user", fragments=[ImageMedia(value="img.jpg")])]
        )
        original = OCRConversationData(
            image_path=Path("img.jpg"),
            image_id="id1",
            conversation=conv,
        )
        data = OCRConversationData.from_dict(original.to_dict())
        assert data.conversation is not None
        frag = data.conversation.conversation[0].fragments[0]
        assert isinstance(frag, ImageMedia)
        assert frag.value == "img.jpg"
