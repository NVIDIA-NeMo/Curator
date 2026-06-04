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

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pandas as pd

from nemo_curator.stages.interleaved.annotation.qrcode_annotator import InterleavedQRCodeAnnotatorStage

from .conftest import interleaved_task, make_jpeg_bytes


class TestInterleavedQRCodeAnnotatorStage:
    def test_empty_task_unchanged(self) -> None:
        task = interleaved_task([])
        stage = InterleavedQRCodeAnnotatorStage()
        assert stage.process(task).num_items == 0

    def test_text_row_has_na_score(self) -> None:
        rows = [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
        ]
        task = interleaved_task(rows)
        stage = InterleavedQRCodeAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
        col = "qr_area_ratio"
        assert col in out_frame.columns
        assert pd.isna(out_frame.iloc[0][col])

    def test_does_not_drop_high_qr_image(self) -> None:
        jpeg = make_jpeg_bytes()
        rows = [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": jpeg,
                "source_ref": None,
                "materialize_error": None,
            },
        ]
        task = interleaved_task(rows)
        with patch("nemo_curator.stages.interleaved.annotation.qrcode_annotator._qr_code_ratio") as mock_ratio:
            mock_ratio.return_value = 0.9
            stage = InterleavedQRCodeAnnotatorStage()
            out_frame = stage.process(task).to_pandas()
        assert len(out_frame) == 1

    def test_column_name_is_qr_area_ratio(self) -> None:
        rows = [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
        ]
        task = interleaved_task(rows)
        stage = InterleavedQRCodeAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
        assert "qr_area_ratio" in out_frame.columns

    def test_image_gets_qr_ratio_score(self) -> None:
        jpeg = make_jpeg_bytes()
        rows = [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": jpeg,
                "source_ref": None,
                "materialize_error": None,
            },
        ]
        task = interleaved_task(rows)
        stage = InterleavedQRCodeAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
        col = "qr_area_ratio"
        assert not pd.isna(out_frame.iloc[0][col])
        assert 0.0 <= float(out_frame.iloc[0][col]) <= 1.0

    def test_none_bytes_gives_na_score(self) -> None:
        rows = [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": b"unused",
                "source_ref": None,
                "materialize_error": None,
            },
        ]
        task = interleaved_task(rows)

        def _none_bytes(
            self: object, task: object, df: pd.DataFrame, row_mask: pd.Series
        ) -> Iterator[tuple[Any, None]]:
            del self, task
            for idx in df[row_mask].index:
                yield idx, None

        with patch.object(InterleavedQRCodeAnnotatorStage, "iter_materialized_bytes", _none_bytes):
            stage = InterleavedQRCodeAnnotatorStage()
            out_frame = stage.process(task).to_pandas()
        assert pd.isna(out_frame.iloc[0]["qr_area_ratio"])
