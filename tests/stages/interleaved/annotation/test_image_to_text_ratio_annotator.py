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
    _text_word_count,
    per_row_image_word_counts_broadcast,
)
from nemo_curator.stages.interleaved.annotation.pass_mask import interleaved_score_pass_mask

from .conftest import interleaved_task


def test_text_word_count_none_is_zero() -> None:
    assert _text_word_count(None) == 0


def test_text_word_count_nan_float_is_zero() -> None:
    assert _text_word_count(float("nan")) == 0


def test_text_word_count_splits_on_whitespace() -> None:
    assert _text_word_count("  one   two three  ") == 3


def test_image_to_text_ratio_annotator_empty_task_unchanged() -> None:
    task = interleaved_task([])
    stage = InterleavedImageToTextRatioAnnotatorStage()
    assert stage.process(task).num_items == 0


def test_image_to_text_ratio_annotator_does_not_drop_rows() -> None:
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


def test_image_to_text_ratio_annotator_column_names_use_stage_name() -> None:
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
    assert f"{stage.name}_image_num" in out_frame.columns
    assert f"{stage.name}_text_word_num" in out_frame.columns


def test_image_to_text_ratio_annotator_counts_stored_at_position_zero_only() -> None:
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
    col = f"{stage.name}_image_num"
    pos0 = out_frame[out_frame["position"] == 0]
    pos1 = out_frame[out_frame["position"] == 1]
    assert int(pos0[col].iloc[0]) == 1
    assert pd.isna(pos1[col].iloc[0])


def test_image_to_text_ratio_annotator_image_only_sample() -> None:
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
    col = f"{stage.name}_image_num"
    assert int(out_frame.loc[out_frame["position"] == 0, col].iloc[0]) == 2
    assert pd.isna(out_frame.loc[out_frame["position"] == 1, col].iloc[0])


def test_per_row_image_word_counts_broadcast_values() -> None:
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


def test_image_to_text_ratio_annotator_pass_mask_ratio_in_range() -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "one two three four",
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
    mask = interleaved_score_pass_mask(stage, task, task.to_pandas(), min_ratio=0.2, max_ratio=1.0)
    assert mask.all()


def test_image_to_text_ratio_annotator_pass_mask_ratio_below_min_fails() -> None:
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
    mask = interleaved_score_pass_mask(stage, task, task.to_pandas(), min_ratio=1.0, max_ratio=2.0)
    assert not mask.any()


def test_image_to_text_ratio_annotator_pass_mask_no_sample_id_all_pass() -> None:
    rows = [
        {
            "sample_id": "x",
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
    df = task.to_pandas().drop(columns=["sample_id"])
    stage = InterleavedImageToTextRatioAnnotatorStage()
    mask = interleaved_score_pass_mask(stage, task, df, drop_invalid_rows=False, min_ratio=0.0, max_ratio=1.0)
    assert mask.all()


def test_image_to_text_ratio_annotator_multiple_samples_stored_correctly() -> None:
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
    col_img = f"{stage.name}_image_num"
    col_word = f"{stage.name}_text_word_num"
    a_row = out_frame[(out_frame["sample_id"] == "a") & (out_frame["position"] == 0)]
    b_row = out_frame[(out_frame["sample_id"] == "b") & (out_frame["position"] == 0)]
    assert int(a_row[col_img].iloc[0]) == 1
    assert int(a_row[col_word].iloc[0]) == 2
    assert int(b_row[col_img].iloc[0]) == 0
    assert int(b_row[col_word].iloc[0]) == 3
