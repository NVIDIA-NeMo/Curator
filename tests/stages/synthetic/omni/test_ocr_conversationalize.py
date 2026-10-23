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

"""Unit tests for nemo_curator.stages.synthetic.omni.ocr_conversationalize.

Covers two things:
  - The shape contract of every word-output format (each must emit text +
    bbox coords and round-trip through json without crashing) — this is what
    keeps the diverse generation safe.
  - The OCRConversationData dataclass round-trip (to_dict/from_dict).
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from nemo_curator.stages.synthetic.omni.ocr_conversationalize import (
    SDG_PROMPT_VARIATIONS,
    WORD_OUTPUT_FORMATS,
    OCRConversationData,
)
from nemo_curator.stages.synthetic.omni.utils.conversation import (
    ConversationSample,
    ImageMedia,
    Message,
)
from nemo_curator.tasks.ocr import OCRDenseItem

WordFormatter = Callable[[list], tuple[str, str]]

_WORDS = [
    OCRDenseItem(bbox_2d=[10, 20, 100, 50], text_content="HELLO"),
    OCRDenseItem(bbox_2d=[120, 25, 200, 55], text_content="WORLD"),
]


def _make_conversation() -> ConversationSample:
    return ConversationSample(
        conversation=[
            Message(sender="user", fragments=[ImageMedia(value="img.jpg"), "Q1"]),
            Message(sender="assistant", fragments=["A1"]),
        ]
    )


def _make_data(**kwargs: object) -> OCRConversationData:
    defaults: dict[str, object] = {
        "image_path": Path("test.jpg"),
        "image_id": "img_0",
        "ocr_dense": [_WORDS[0]],
        "conversation": _make_conversation(),
    }
    defaults.update(kwargs)
    return OCRConversationData(**defaults)  # type: ignore[arg-type]


class TestWordOutputFormats:
    """Every format in WORD_OUTPUT_FORMATS must satisfy a shared contract.

    Parametrized across all 11 formatters so additions/removals are picked up
    automatically — no per-formatter test boilerplate.
    """

    def test_prompt_variations_are_non_empty(self) -> None:
        # The pool the dense-dump turn samples from — must stay populated.
        assert len(SDG_PROMPT_VARIATIONS) > 0
        for prompt in SDG_PROMPT_VARIATIONS:
            assert prompt.strip(), "empty prompt in SDG_PROMPT_VARIATIONS"

    @pytest.mark.parametrize("fmt_fn", WORD_OUTPUT_FORMATS)
    def test_format_returns_non_empty_suffix_and_answer(self, fmt_fn: WordFormatter) -> None:
        suffix, answer = fmt_fn(_WORDS)
        assert isinstance(suffix, str)
        assert suffix.strip(), "format instruction must be non-empty"
        assert isinstance(answer, str)
        assert answer.strip(), "answer must be non-empty"

    @pytest.mark.parametrize("fmt_fn", WORD_OUTPUT_FORMATS)
    def test_format_includes_text_content(self, fmt_fn: WordFormatter) -> None:
        _suffix, answer = fmt_fn(_WORDS)
        # Every word's text must appear somewhere in the answer.
        assert "HELLO" in answer
        assert "WORLD" in answer

    @pytest.mark.parametrize("fmt_fn", WORD_OUTPUT_FORMATS)
    def test_format_includes_all_bbox_coordinates(self, fmt_fn: WordFormatter) -> None:
        _suffix, answer = fmt_fn(_WORDS)
        for word in _WORDS:
            for coord in word.bbox_2d:
                assert str(coord) in answer, f"coord {coord} missing in {fmt_fn.__name__} output"

    @pytest.mark.parametrize("fmt_fn", WORD_OUTPUT_FORMATS)
    def test_format_handles_empty_word_list(self, fmt_fn: WordFormatter) -> None:
        # Empty input must not crash any formatter.
        suffix, answer = fmt_fn([])
        assert isinstance(suffix, str)
        assert isinstance(answer, str)

    def test_distinct_formats_produce_distinct_outputs(self) -> None:
        # Diversity check: not all 11 formatters collapse to the same string.
        outputs = {fmt(_WORDS)[1] for fmt in WORD_OUTPUT_FORMATS}
        assert len(outputs) >= 5, (
            f"WORD_OUTPUT_FORMATS expected to produce diverse strings, got {len(outputs)} distinct"
        )


class TestOCRConversationData:
    def test_from_dict_roundtrip(self) -> None:
        original = _make_data()
        recovered = OCRConversationData.from_dict(original.to_dict())
        assert recovered.image_id == original.image_id
        assert recovered.conversation is not None
        assert len(recovered.conversation.conversation) == 2
        assert recovered.conversation.conversation[1].fragments == ["A1"]

    def test_from_dict_with_no_conversation_key(self) -> None:
        d = _make_data(conversation=None).to_dict()
        d.pop("conversation", None)
        recovered = OCRConversationData.from_dict(d)
        assert recovered.conversation is None

    def test_inherits_ocr_data_fields(self) -> None:
        d = _make_data(
            ocr_scoring_model="nemotron",
            ocr_scoring_mode="word",
        ).to_dict()
        recovered = OCRConversationData.from_dict(d)
        assert recovered.ocr_scoring_model == "nemotron"
        assert recovered.ocr_scoring_mode == "word"


def test_at_least_one_format_emits_valid_json() -> None:
    """The JSON-output formatters must actually emit json that parses — this
    catches regressions where a quote or comma is dropped."""
    for fmt in WORD_OUTPUT_FORMATS:
        suffix, answer = fmt(_WORDS)
        # Heuristically pick formatters with 'JSON' or 'json' in their suffix.
        if "JSON" in suffix or "json" in suffix:
            # Strip code-fence wrappers if present.
            stripped = answer.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as e:
                msg = f"formatter {fmt.__name__} emitted invalid JSON: {e}"
                raise AssertionError(msg) from e
            # The parsed payload must still mention both words.
            assert "HELLO" in json.dumps(parsed)
            assert "WORLD" in json.dumps(parsed)
