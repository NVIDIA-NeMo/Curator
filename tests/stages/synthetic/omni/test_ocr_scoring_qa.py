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

"""Unit tests for OCRScoringQAStage.

The verifier model is mocked (it lives in models/omni and is tested there);
these tests cover prompt construction, the JSON extractor, and response
handling — including how scoring thresholds map to per-bbox / per-image
validity and which conversation shape is produced.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.stages.synthetic.omni.base import SkipSample
from nemo_curator.stages.synthetic.omni.ocr_scoring_qa import OCRScoringQAStage, _parse_json_object
from nemo_curator.tasks.image import ImageSampleTask
from nemo_curator.tasks.ocr import OCRData, OCRDenseItem


def _make_word(bbox: list[int], text: str, *, valid: bool = True) -> OCRDenseItem:
    return OCRDenseItem(bbox_2d=bbox, text_content=text, valid=valid)


def _make_task(words: list[OCRDenseItem] | None = None) -> ImageSampleTask[OCRData]:
    return ImageSampleTask(
        dataset_name="test",
        data=OCRData(image_path=Path("test.jpg"), image_id="img_0", ocr_dense=words),
    )


def _make_stage(**kwargs: object) -> OCRScoringQAStage:
    with patch(
        "nemo_curator.stages.synthetic.omni.ocr_scoring_qa.NVInferenceClient",
        return_value=MagicMock(),
    ):
        return OCRScoringQAStage(**kwargs)


def _verifier_response(
    *,
    ocr_mode: str = "word",
    items: list[dict] | None = None,
    missing_text: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "ocr_mode": ocr_mode,
            "text": items or [],
            "missing_text": missing_text or [],
        }
    )


class TestParseJsonObject:
    """JSON extraction from verifier responses — must survive code fences and inline prose."""

    def test_strips_markdown_code_fences(self) -> None:
        assert _parse_json_object('```json\n{"k": 42}\n```') == {"k": 42}
        assert _parse_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_finds_embedded_json_in_prose(self) -> None:
        result = _parse_json_object('Result: {"ocr_mode": "word", "text": []} done.')
        assert result == {"ocr_mode": "word", "text": []}

    def test_returns_none_for_unparseable_input(self) -> None:
        assert _parse_json_object("plain text") is None
        assert _parse_json_object("{broken:") is None

    def test_returns_none_for_top_level_array(self) -> None:
        # The verifier protocol expects an object; arrays at root are rejected.
        assert _parse_json_object("[1, 2, 3]") is None


class TestOCRScoringQAStage:
    """Prompt building and response interpretation against the verifier protocol."""

    # ----- build_prompt --------------------------------------------------

    def test_build_prompt_skips_when_no_bboxes(self) -> None:
        stage = _make_stage()
        for words in (None, []):
            with pytest.raises(SkipSample):
                stage.build_prompt(_make_task(words=words))

    def test_build_prompt_embeds_bboxes_and_text(self) -> None:
        stage = _make_stage()
        words = [_make_word([10, 20, 100, 50], "FOO"), _make_word([200, 0, 300, 50], "BAR")]
        task = _make_task(words=words)
        prompt = stage.build_prompt(task)
        # Both texts and a y-first coordinate from the first bbox must appear.
        assert "FOO" in prompt
        assert "BAR" in prompt
        assert "20" in prompt  # y1 of first bbox
        # The stage records what it sent for later inspection.
        assert task.data.ocr_scoring_prompt == prompt
        assert task.data.ocr_scoring_model

    # ----- handle_response: failure modes -------------------------------

    def test_empty_response_marks_image_invalid(self) -> None:
        result = _make_stage().handle_response(_make_task([_make_word([0, 0, 1, 1], "X")]), "")
        assert result.data.is_valid is False
        assert "empty response" in (result.data.error or "")

    def test_unparseable_response_marks_image_invalid(self) -> None:
        result = _make_stage().handle_response(_make_task([_make_word([0, 0, 1, 1], "X")]), "not json {{")
        assert result.data.is_valid is False
        assert "could not parse" in (result.data.error or "")

    def test_response_missing_idx_invalidates_that_bbox(self) -> None:
        stage = _make_stage()
        task = _make_task([_make_word([0, 0, 1, 1], "X")])
        # Response has no entry for idx=0
        result = stage.handle_response(task, _verifier_response(items=[]))
        assert result.data.ocr_dense[0].valid is False

    # ----- handle_response: threshold semantics --------------------------

    @pytest.mark.parametrize(
        ("min_match", "max_errors", "score", "errors", "expected_valid"),
        [
            (7, 0, 9, 0, True),  # passes both thresholds
            (7, 0, 5, 0, False),  # bbox_match below min
            (7, 0, 10, 1, False),  # text_errors above max
        ],
    )
    def test_per_bbox_validity_threshold(
        self,
        min_match: int,
        max_errors: int,
        score: int,
        errors: int,
        expected_valid: bool,
    ) -> None:
        stage = _make_stage(min_bbox_match=min_match, max_text_errors=max_errors)
        task = _make_task([_make_word([0, 0, 1, 1], "X")])
        response = _verifier_response(items=[{"idx": 0, "bbox_match": score, "text_errors": errors}])
        result = stage.handle_response(task, response)
        assert result.data.ocr_dense[0].valid is expected_valid
        assert result.data.ocr_dense[0].bbox_match == score
        assert result.data.ocr_dense[0].text_errors == errors

    def test_all_bboxes_below_threshold_invalidates_image(self) -> None:
        stage = _make_stage(min_bbox_match=7)
        task = _make_task([_make_word([0, 0, 1, 1], "X")])
        response = _verifier_response(items=[{"idx": 0, "bbox_match": 3, "text_errors": 0}])
        result = stage.handle_response(task, response)
        assert result.data.is_valid is False
        assert "no bboxes passed quality threshold" in (result.data.error or "")

    # ----- handle_response: missing_text behavior ------------------------

    def test_fail_on_missing_text_true_invalidates_image(self) -> None:
        stage = _make_stage(fail_on_missing_text=True)
        task = _make_task([_make_word([0, 0, 1, 1], "X")])
        response = _verifier_response(
            items=[{"idx": 0, "bbox_match": 10, "text_errors": 0}],
            missing_text=[{"text": "GONE", "bbox_2d": [10, 10, 50, 50]}],
        )
        result = stage.handle_response(task, response)
        assert result.data.is_valid is False
        assert "missing text region" in (result.data.error or "")

    def test_fail_on_missing_text_false_keeps_partial_qa(self) -> None:
        stage = _make_stage(fail_on_missing_text=False, dense_dump_prob=0.0)
        task = _make_task([_make_word([0, 0, 1, 1], "X")])
        response = _verifier_response(
            items=[{"idx": 0, "bbox_match": 10, "text_errors": 0}],
            missing_text=[{"text": "GONE", "bbox_2d": [10, 10, 50, 50]}],
        )
        result = stage.handle_response(task, response)
        assert result.data.is_valid is True
        assert result.data.conversation is not None

    # ----- handle_response: ocr_mode + conversation shape ---------------

    @pytest.mark.parametrize(("mode", "expected_word_level"), [("word", True), ("line", False)])
    def test_ocr_mode_propagates_to_word_level(self, mode: str, expected_word_level: bool) -> None:
        stage = _make_stage()
        task = _make_task([_make_word([0, 0, 1, 1], "X")])
        response = _verifier_response(
            ocr_mode=mode,
            items=[{"idx": 0, "bbox_match": 10, "text_errors": 0}],
        )
        result = stage.handle_response(task, response)
        assert result.data.ocr_is_word_level is expected_word_level

    def test_dense_dump_prob_1_produces_single_turn_dump(self) -> None:
        stage = _make_stage(dense_dump_prob=1.0)
        task = _make_task([_make_word([0, 0, 1, 1], "X")])
        response = _verifier_response(items=[{"idx": 0, "bbox_match": 10, "text_errors": 0}])
        result = stage.handle_response(task, response)
        # Single QA turn = exactly 2 messages.
        assert len(result.data.conversation.conversation) == 2

    def test_dense_dump_prob_0_produces_multi_turn_qa(self) -> None:
        stage = _make_stage(dense_dump_prob=0.0)
        words = [_make_word([i * 100, 0, (i + 1) * 100, 50], f"WORD{i}") for i in range(5)]
        task = _make_task(words)
        response = _verifier_response(
            items=[{"idx": i, "bbox_match": 10, "text_errors": 0} for i in range(5)],
        )
        result = stage.handle_response(task, response)
        # Multi-turn QA: 2 messages per bbox = 10 messages.
        assert len(result.data.conversation.conversation) == 10
