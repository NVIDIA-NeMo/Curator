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

"""Unit tests for OCRDenseItem and OCRData — round-trip serialization + join semantics."""

from pathlib import Path

from nemo_curator.tasks.ocr import OCRData, OCRDenseItem


class TestOCRDenseItem:
    """Construction, JSON deserialization, and bbox-union join."""

    def test_tuple_bbox_is_normalized_to_list(self) -> None:
        item = OCRDenseItem(bbox_2d=(10, 20, 100, 50), text_content="hi")
        assert isinstance(item.bbox_2d, list)
        assert item.bbox_2d == [10, 20, 100, 50]

    def test_from_dict_round_trips_all_fields(self) -> None:
        item = OCRDenseItem.from_dict(
            {
                "bbox_2d": [10, 20, 100, 50],
                "text_content": "HELLO",
                "valid": False,
                "bbox_match": 7,
                "text_errors": 2,
            }
        )
        assert item.bbox_2d == [10, 20, 100, 50]
        assert item.text_content == "HELLO"
        assert item.valid is False
        assert item.bbox_match == 7
        assert item.text_errors == 2

    def test_from_dict_tolerates_partial_payloads(self) -> None:
        # Missing bbox → zero bbox; null text → empty string.
        empty_bbox = OCRDenseItem.from_dict({"text_content": "x"})
        assert empty_bbox.bbox_2d == [0, 0, 0, 0]

        null_text = OCRDenseItem.from_dict({"bbox_2d": [0, 0, 0, 0], "text_content": None})
        assert null_text.text_content == ""

    def test_join_empty_iterable_returns_invalid_sentinel(self) -> None:
        result = OCRDenseItem.join([])
        assert result.valid is False
        assert result.bbox_2d == [0, 0, 0, 0]
        assert result.text_content == ""

    def test_join_unions_bboxes_and_concatenates_text(self) -> None:
        items = [
            OCRDenseItem(bbox_2d=[100, 200, 150, 250], text_content="X"),
            OCRDenseItem(bbox_2d=[50, 300, 80, 400], text_content="Y"),
            OCRDenseItem(bbox_2d=[200, 100, 300, 500], text_content="Z"),
        ]
        result = OCRDenseItem.join(items)
        # Smallest mins and largest maxes across all three bboxes.
        assert result.bbox_2d == [50, 100, 300, 500]
        assert result.text_content == "X Y Z"
        assert result.valid is True

    def test_join_uses_custom_separator(self) -> None:
        items = [OCRDenseItem(bbox_2d=[0, 0, 1, 1], text_content=t) for t in ("A", "B", "C")]
        assert OCRDenseItem.join(items, separator="|").text_content == "A|B|C"


_BASE: dict[str, object] = {"image_path": "test.jpg", "image_id": "img0"}


class TestOCRData:
    """JSONL deserialization of OCRData — including the nested OCRDenseItem list."""

    def test_minimal_payload_round_trips(self) -> None:
        data = OCRData.from_dict(_BASE)
        assert data.image_path == Path("test.jpg")
        assert data.image_id == "img0"
        assert data.is_valid is True
        assert data.ocr_dense is None

    def test_ocr_dense_list_deserialized_as_items(self) -> None:
        data = OCRData.from_dict({
            **_BASE,
            "ocr_dense": [{"bbox_2d": [1, 2, 3, 4], "text_content": "HI"}],
        })
        assert data.ocr_dense is not None
        assert len(data.ocr_dense) == 1
        assert data.ocr_dense[0].text_content == "HI"
        assert data.ocr_dense[0].bbox_2d == [1, 2, 3, 4]

    def test_optional_fields_propagate(self) -> None:
        data = OCRData.from_dict({
            **_BASE,
            "is_valid": False,
            "error": "boom",
            "ocr_is_word_level": False,
            "ocr_scoring_model": "nemotron-nano-omni",
            "ocr_scoring_mode": "line",
        })
        assert data.is_valid is False
        assert data.error == "boom"
        assert data.ocr_is_word_level is False
        assert data.ocr_scoring_model == "nemotron-nano-omni"
        assert data.ocr_scoring_mode == "line"
