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

import pandas as pd

from nemo_curator.stages.interleaved.annotation.image_to_text_ratio_annotator import (
    InterleavedImageToTextRatioAnnotatorStage,
    per_row_image_word_counts_broadcast,
)

from .conftest import interleaved_task


class TestPerRowImageWordCountsBroadcast:
    def test_values(self) -> None:
        rows = [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "one two",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s1",
                "position": 1,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
        ]
        task = interleaved_task(rows)
        df = task.to_pandas()
        img, words = per_row_image_word_counts_broadcast(df)
        assert int(img.iloc[0]) == 1
        assert int(img.iloc[1]) == 1
        assert int(words.iloc[0]) == 2
        assert int(words.iloc[1]) == 2


class TestInterleavedImageToTextRatioAnnotatorStage:
    def test_empty_task_unchanged(self) -> None:
        task = interleaved_task([])
        stage = InterleavedImageToTextRatioAnnotatorStage()
        assert stage.process(task).num_items == 0

    def test_does_not_drop_rows(self) -> None:
        rows = [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "one two three four five",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s1",
                "position": 1,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
        ]
        task = interleaved_task(rows)
        stage = InterleavedImageToTextRatioAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
        assert len(out_frame) == 2

    def test_column_names_are_image_and_word_counts(self) -> None:
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
        stage = InterleavedImageToTextRatioAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
        assert "image_num" in out_frame.columns
        assert "text_word_num" in out_frame.columns

    def test_counts_stored_at_position_zero_only(self) -> None:
        rows = [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "a b",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s1",
                "position": 1,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
        ]
        task = interleaved_task(rows)
        stage = InterleavedImageToTextRatioAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
        col = "image_num"
        assert int(out_frame[out_frame["position"] == 0][col].iloc[0]) == 1
        assert pd.isna(out_frame[out_frame["position"] == 1][col].iloc[0])

    def test_image_only_sample(self) -> None:
        rows = [
            {
                "sample_id": "solo",
                "position": 0,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "solo",
                "position": 1,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
        ]
        task = interleaved_task(rows)
        stage = InterleavedImageToTextRatioAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
        col = "image_num"
        assert int(out_frame.loc[out_frame["position"] == 0, col].iloc[0]) == 2
        assert pd.isna(out_frame.loc[out_frame["position"] == 1, col].iloc[0])

    def test_multiple_samples_stored_correctly(self) -> None:
        rows = [
            {
                "sample_id": "a",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "x y",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "a",
                "position": 1,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "b",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "p q r",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
        ]
        task = interleaved_task(rows)
        stage = InterleavedImageToTextRatioAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
        a_row = out_frame[(out_frame["sample_id"] == "a") & (out_frame["position"] == 0)]
        b_row = out_frame[(out_frame["sample_id"] == "b") & (out_frame["position"] == 0)]
        assert int(a_row["image_num"].iloc[0]) == 1
        assert int(a_row["text_word_num"].iloc[0]) == 2
        assert int(b_row["image_num"].iloc[0]) == 0
        assert int(b_row["text_word_num"].iloc[0]) == 3
