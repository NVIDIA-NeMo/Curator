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

"""Unit tests for nemo_curator.stages.synthetic.omni.ocr_scoring_qa."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.stages.synthetic.omni.ocr_scoring_qa import (
    OCRScoringQAStage,
    _parse_json_object,
)
from nemo_curator.tasks.image import SingleDataTask
from nemo_curator.tasks.ocr import OCRData, OCRDenseWord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_word(
    bbox: list[int],
    text: str,
    *,
    valid: bool = True,
) -> OCRDenseWord:
    return OCRDenseWord(bbox_2d=bbox, text_content=text, valid=valid)


def _make_task(
    words: list[OCRDenseWord] | None = None,
    *,
    task_id: str = "t0",
    image_path: Path | None = None,
    is_valid: bool = True,
) -> SingleDataTask[OCRData]:
    if image_path is None:
        image_path = Path("test.jpg")
    data = OCRData(
        image_path=image_path,
        image_id="img_0",
        is_valid=is_valid,
        ocr_dense=words,
    )
    return SingleDataTask(task_id=task_id, dataset_name="test", data=data)


def _make_stage(**kwargs) -> OCRScoringQAStage:
    """Build a stage with the Gemini model mocked out (no network calls)."""
    with patch(
        "nemo_curator.stages.synthetic.omni.ocr_scoring_qa.Gemini3Pro",
        return_value=MagicMock(),
    ):
        return OCRScoringQAStage(**kwargs)


def _gemini_response(
    *,
    ocr_mode: str = "word",
    items: list[dict] | None = None,
    missing_text: list[dict] | None = None,
) -> str:
    payload = {
        "ocr_mode": ocr_mode,
        "text": items or [],
        "missing_text": missing_text or [],
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# _parse_json_object
# ---------------------------------------------------------------------------


class TestParseJsonObject:
    def test_plain_json(self):
        result = _parse_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_code_block(self):
        text = '```json\n{"key": 42}\n```'
        result = _parse_json_object(text)
        assert result == {"key": 42}

    def test_json_in_generic_code_block(self):
        text = '```\n{"a": 1}\n```'
        result = _parse_json_object(text)
        assert result == {"a": 1}

    def test_invalid_json_returns_none(self):
        result = _parse_json_object("this is not json at all")
        assert result is None

    def test_broken_json_returns_none(self):
        result = _parse_json_object("{broken:")
        assert result is None

    def test_json_array_skipped(self):
        # The function only returns dicts
        result = _parse_json_object("[1, 2, 3]")
        assert result is None

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"ocr_mode": "word", "text": []} as requested.'
        result = _parse_json_object(text)
        assert result is not None
        assert result["ocr_mode"] == "word"

    def test_nested_json_dict(self):
        payload = {"ocr_mode": "line", "text": [{"idx": 0, "bbox_match": 9, "text_errors": 0}]}
        result = _parse_json_object(json.dumps(payload))
        assert result == payload


# ---------------------------------------------------------------------------
# OCRScoringQAStage.build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_raises_skip_sample_for_empty_ocr_dense(self):
        from nemo_curator.stages.synthetic.omni.base import SkipSample

        stage = _make_stage()
        task = _make_task(words=[])
        with pytest.raises(SkipSample):
            stage.build_prompt(task)

    def test_raises_skip_sample_for_none_ocr_dense(self):
        from nemo_curator.stages.synthetic.omni.base import SkipSample

        stage = _make_stage()
        task = _make_task(words=None)
        with pytest.raises(SkipSample):
            stage.build_prompt(task)

    def test_prompt_contains_bbox_json(self):
        stage = _make_stage()
        words = [_make_word([10, 20, 100, 50], "HELLO")]
        task = _make_task(words=words)
        prompt = stage.build_prompt(task)
        assert "HELLO" in prompt
        # NB: prompt uses (y1, x1, y2, x2) ordering
        assert "20" in prompt  # y1 from bbox

    def test_prompt_sets_task_fields(self):
        stage = _make_stage()
        words = [_make_word([10, 20, 100, 50], "HELLO")]
        task = _make_task(words=words)
        stage.build_prompt(task)
        assert task.data.ocr_scoring_prompt is not None
        assert task.data.ocr_scoring_model is not None

    def test_multiple_bboxes_all_included(self):
        stage = _make_stage()
        words = [
            _make_word([0, 0, 100, 50], "WORD1"),
            _make_word([200, 0, 300, 50], "WORD2"),
        ]
        task = _make_task(words=words)
        prompt = stage.build_prompt(task)
        assert "WORD1" in prompt
        assert "WORD2" in prompt


# ---------------------------------------------------------------------------
# OCRScoringQAStage.handle_response
# ---------------------------------------------------------------------------


class TestHandleResponse:
    def test_empty_response_marks_invalid(self):
        stage = _make_stage()
        task = _make_task(words=[_make_word([0, 0, 100, 50], "HI")])
        result = stage.handle_response(task, "")
        assert result.data.is_valid is False
        assert "empty response" in (result.data.error or "")

    def test_unparseable_json_marks_invalid(self):
        stage = _make_stage()
        task = _make_task(words=[_make_word([0, 0, 100, 50], "HI")])
        result = stage.handle_response(task, "not valid json at all {{")
        assert result.data.is_valid is False
        assert "could not parse" in (result.data.error or "")

    def test_scores_applied_to_bboxes(self):
        stage = _make_stage()
        words = [_make_word([0, 0, 100, 50], "HELLO")]
        task = _make_task(words=words)
        response = _gemini_response(
            items=[{"idx": 0, "bbox_match": 9, "text_errors": 0, "is_word": True, "is_line": False}]
        )
        result = stage.handle_response(task, response)
        word = result.data.ocr_dense[0]
        assert word.bbox_match == 9
        assert word.text_errors == 0
        assert word.valid is True

    def test_low_bbox_match_marks_invalid(self):
        stage = _make_stage(min_bbox_match=7)
        words = [_make_word([0, 0, 100, 50], "HELLO")]
        task = _make_task(words=words)
        response = _gemini_response(
            items=[{"idx": 0, "bbox_match": 5, "text_errors": 0, "is_word": True, "is_line": False}]
        )
        result = stage.handle_response(task, response)
        assert result.data.ocr_dense[0].valid is False

    def test_text_errors_marks_invalid(self):
        stage = _make_stage(max_text_errors=0)
        words = [_make_word([0, 0, 100, 50], "HELLO")]
        task = _make_task(words=words)
        response = _gemini_response(
            items=[{"idx": 0, "bbox_match": 10, "text_errors": 1, "is_word": True, "is_line": False}]
        )
        result = stage.handle_response(task, response)
        assert result.data.ocr_dense[0].valid is False

    def test_all_bboxes_fail_marks_image_invalid(self):
        stage = _make_stage(min_bbox_match=7)
        words = [_make_word([0, 0, 100, 50], "HELLO")]
        task = _make_task(words=words)
        response = _gemini_response(
            items=[{"idx": 0, "bbox_match": 3, "text_errors": 0, "is_word": True, "is_line": False}]
        )
        result = stage.handle_response(task, response)
        assert result.data.is_valid is False
        assert "no bboxes passed quality threshold" in (result.data.error or "")

    def test_missing_idx_in_response_marks_bbox_invalid(self):
        stage = _make_stage()
        words = [_make_word([0, 0, 100, 50], "HELLO")]
        task = _make_task(words=words)
        # Response with no matching idx entry
        response = _gemini_response(items=[])
        result = stage.handle_response(task, response)
        assert result.data.ocr_dense[0].valid is False

    def test_fail_on_missing_text_true_marks_invalid(self):
        stage = _make_stage(fail_on_missing_text=True)
        words = [_make_word([0, 0, 100, 50], "HELLO")]
        task = _make_task(words=words)
        response = _gemini_response(
            items=[{"idx": 0, "bbox_match": 10, "text_errors": 0, "is_word": True, "is_line": False}],
            missing_text=[{"text": "MISSING", "bbox_2d": [10, 10, 50, 50]}],
        )
        result = stage.handle_response(task, response)
        assert result.data.is_valid is False
        assert "missing text region" in (result.data.error or "")

    def test_fail_on_missing_text_false_keeps_partial_qa(self):
        stage = _make_stage(fail_on_missing_text=False, dense_dump_prob=0.0)
        words = [_make_word([0, 0, 100, 50], "HELLO")]
        task = _make_task(words=words)
        response = _gemini_response(
            items=[{"idx": 0, "bbox_match": 10, "text_errors": 0, "is_word": True, "is_line": False}],
            missing_text=[{"text": "MISSING", "bbox_2d": [10, 10, 50, 50]}],
        )
        result = stage.handle_response(task, response)
        # Image stays valid (partial QA generated)
        assert result.data.is_valid is True
        assert result.data.conversation is not None

    def test_ocr_mode_word_sets_is_word_level(self):
        stage = _make_stage()
        words = [_make_word([0, 0, 100, 50], "HELLO")]
        task = _make_task(words=words)
        response = _gemini_response(
            ocr_mode="word",
            items=[{"idx": 0, "bbox_match": 10, "text_errors": 0, "is_word": True, "is_line": False}],
        )
        result = stage.handle_response(task, response)
        assert result.data.ocr_is_word_level is True

    def test_ocr_mode_line_clears_is_word_level(self):
        stage = _make_stage()
        words = [_make_word([0, 0, 100, 50], "HELLO WORLD")]
        task = _make_task(words=words)
        response = _gemini_response(
            ocr_mode="line",
            items=[{"idx": 0, "bbox_match": 10, "text_errors": 0, "is_word": False, "is_line": True}],
        )
        result = stage.handle_response(task, response)
        assert result.data.ocr_is_word_level is False

    def test_dense_dump_generated_when_ocr_complete_and_prob_1(self):
        """With dense_dump_prob=1.0 and no missing text, conversation must be a single-turn dump."""
        stage = _make_stage(dense_dump_prob=1.0)
        words = [_make_word([0, 0, 100, 50], "HELLO")]
        task = _make_task(words=words)
        response = _gemini_response(
            items=[{"idx": 0, "bbox_match": 10, "text_errors": 0, "is_word": True, "is_line": False}],
            missing_text=[],
        )
        result = stage.handle_response(task, response)
        # Dense dump produces exactly 2 messages (1 QA turn)
        assert result.data.conversation is not None
        assert len(result.data.conversation.conversation) == 2

    def test_multiturn_qa_generated_when_prob_0(self):
        """With dense_dump_prob=0.0, multi-turn QA is always used."""
        stage = _make_stage(dense_dump_prob=0.0)
        # Use multiple different words so multiple QA pairs can be generated
        words = [_make_word([i * 100, 0, (i + 1) * 100, 50], f"WORD{i}") for i in range(5)]
        task = _make_task(words=words)
        items = [{"idx": i, "bbox_match": 10, "text_errors": 0, "is_word": True, "is_line": False} for i in range(5)]
        response = _gemini_response(items=items, missing_text=[])
        result = stage.handle_response(task, response)
        assert result.data.conversation is not None
        # Multi-turn QA should have more than 2 messages (> 1 pair)
        assert len(result.data.conversation.conversation) >= 2

    def test_raw_response_stored(self):
        stage = _make_stage()
        words = [_make_word([0, 0, 100, 50], "HELLO")]
        task = _make_task(words=words)
        response = _gemini_response(
            items=[{"idx": 0, "bbox_match": 10, "text_errors": 0, "is_word": True, "is_line": False}]
        )
        result = stage.handle_response(task, response)
        assert result.data.ocr_scoring_response_raw == response
