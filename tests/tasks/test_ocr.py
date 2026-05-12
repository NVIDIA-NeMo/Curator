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

"""Unit tests for nemo_curator.tasks.ocr."""

from pathlib import Path

from nemo_curator.tasks.ocr import OCRData, OCRDenseItem

# ---------------------------------------------------------------------------
# OCRDenseItem.__post_init__
# ---------------------------------------------------------------------------


class TestOCRDenseItemPostInit:
    def test_tuple_bbox_converted_to_list(self) -> None:
        word = OCRDenseItem(bbox_2d=(10, 20, 100, 50), text_content="hi")
        assert isinstance(word.bbox_2d, list)
        assert word.bbox_2d == [10, 20, 100, 50]

    def test_list_bbox_unchanged(self) -> None:
        word = OCRDenseItem(bbox_2d=[1, 2, 3, 4], text_content="x")
        assert word.bbox_2d == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# OCRDenseItem.from_dict
# ---------------------------------------------------------------------------


class TestOCRDenseItemFromDict:
    def test_basic_round_trip(self) -> None:
        word = OCRDenseItem.from_dict({"bbox_2d": [10, 20, 100, 50], "text_content": "HELLO"})
        assert word.bbox_2d == [10, 20, 100, 50]
        assert word.text_content == "HELLO"

    def test_missing_bbox_defaults_to_zero(self) -> None:
        word = OCRDenseItem.from_dict({"text_content": "x"})
        assert word.bbox_2d == [0, 0, 0, 0]

    def test_optional_fields_default_to_none(self) -> None:
        word = OCRDenseItem.from_dict({"bbox_2d": [0, 0, 0, 0], "text_content": "x"})
        assert word.quad is None
        assert word.valid is True
        assert word.bbox_match is None
        assert word.text_errors is None

    def test_optional_fields_set(self) -> None:
        word = OCRDenseItem.from_dict(
            {
                "bbox_2d": [0, 0, 0, 0],
                "text_content": "x",
                "valid": False,
                "bbox_match": 7,
                "text_errors": 2,
            }
        )
        assert word.valid is False
        assert word.bbox_match == 7
        assert word.text_errors == 2

    def test_none_text_content_becomes_empty_string(self) -> None:
        word = OCRDenseItem.from_dict({"bbox_2d": [0, 0, 0, 0], "text_content": None})
        assert word.text_content == ""


# ---------------------------------------------------------------------------
# OCRDenseItem.join
# ---------------------------------------------------------------------------


class TestOCRDenseItemJoin:
    def test_empty_iterator_returns_invalid_word(self) -> None:
        result = OCRDenseItem.join([])
        assert result.valid is False
        assert result.bbox_2d == [0, 0, 0, 0]
        assert result.text_content == ""

    def test_single_word_preserved(self) -> None:
        w = OCRDenseItem(bbox_2d=[10, 20, 100, 50], text_content="HELLO")
        result = OCRDenseItem.join([w])
        assert result.text_content == "HELLO"
        assert result.bbox_2d == [10, 20, 100, 50]
        assert result.valid is True

    def test_bbox_is_union_of_all(self) -> None:
        w1 = OCRDenseItem(bbox_2d=[10, 20, 50, 40], text_content="A")
        w2 = OCRDenseItem(bbox_2d=[60, 5, 120, 80], text_content="B")
        result = OCRDenseItem.join([w1, w2])
        assert result.bbox_2d == [10, 5, 120, 80]

    def test_text_joined_with_space_by_default(self) -> None:
        words = [OCRDenseItem(bbox_2d=[0, 0, 1, 1], text_content=t) for t in ["HELLO", "WORLD"]]
        result = OCRDenseItem.join(words)
        assert result.text_content == "HELLO WORLD"

    def test_custom_separator(self) -> None:
        words = [OCRDenseItem(bbox_2d=[0, 0, 1, 1], text_content=t) for t in ["A", "B", "C"]]
        result = OCRDenseItem.join(words, separator="|")
        assert result.text_content == "A|B|C"

    def test_three_words_bbox_covers_all(self) -> None:
        words = [
            OCRDenseItem(bbox_2d=[100, 200, 150, 250], text_content="X"),
            OCRDenseItem(bbox_2d=[50, 300, 80, 400], text_content="Y"),
            OCRDenseItem(bbox_2d=[200, 100, 300, 500], text_content="Z"),
        ]
        result = OCRDenseItem.join(words)
        assert result.bbox_2d == [50, 100, 300, 500]
        assert result.text_content == "X Y Z"


# ---------------------------------------------------------------------------
# OCRData.from_dict
# ---------------------------------------------------------------------------


class TestOCRDataFromDict:
    def _base(self) -> dict[str, object]:
        return {"image_path": "test.jpg", "image_id": "img0"}

    def test_basic_deserialization(self) -> None:
        data = OCRData.from_dict(self._base())
        assert data.image_path == Path("test.jpg")
        assert data.image_id == "img0"
        assert data.is_valid is True

    def test_is_valid_false(self) -> None:
        data = OCRData.from_dict({**self._base(), "is_valid": False})
        assert data.is_valid is False

    def test_ocr_dense_deserialized(self) -> None:
        d = {**self._base(), "ocr_dense": [{"bbox_2d": [1, 2, 3, 4], "text_content": "HI"}]}
        data = OCRData.from_dict(d)
        assert data.ocr_dense is not None
        assert len(data.ocr_dense) == 1
        assert data.ocr_dense[0].text_content == "HI"

    def test_ocr_dense_none_when_absent(self) -> None:
        data = OCRData.from_dict(self._base())
        assert data.ocr_dense is None

    def test_ocr_is_word_level_default_true(self) -> None:
        data = OCRData.from_dict(self._base())
        assert data.ocr_is_word_level is True

    def test_ocr_is_word_level_override(self) -> None:
        data = OCRData.from_dict({**self._base(), "ocr_is_word_level": False})
        assert data.ocr_is_word_level is False

    def test_optional_scoring_fields(self) -> None:
        d = {**self._base(), "ocr_scoring_model": "nemotron-nano-omni", "ocr_scoring_mode": "word"}
        data = OCRData.from_dict(d)
        assert data.ocr_scoring_model == "nemotron-nano-omni"
        assert data.ocr_scoring_mode == "word"

    def test_error_field(self) -> None:
        data = OCRData.from_dict({**self._base(), "error": "something failed"})
        assert data.error == "something failed"
