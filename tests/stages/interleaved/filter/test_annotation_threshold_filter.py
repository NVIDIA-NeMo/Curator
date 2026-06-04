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

from nemo_curator.stages.interleaved.filter.annotation_threshold_filter import (
    InterleavedAnnotationThresholdFilterStage,
    _clip_cell_max_score,
)
from nemo_curator.tasks import InterleavedBatch

from .conftest import interleaved_task


def _annotated_task(rows: list[dict]) -> InterleavedBatch:
    """Build a task with arbitrary annotation columns (not limited to ``INTERLEAVED_SCHEMA``)."""
    return InterleavedBatch(task_id="test", dataset_name="d", data=pd.DataFrame(rows))


def test_clip_cell_max_score_dict() -> None:
    assert _clip_cell_max_score({0: 0.1, 1: 0.9}) == 0.9


def test_clip_cell_max_score_none_and_empty() -> None:
    assert _clip_cell_max_score(None) is None
    assert _clip_cell_max_score({}) is None


def test_annotation_threshold_empty_task() -> None:
    task = interleaved_task([])
    stage = InterleavedAnnotationThresholdFilterStage()
    out = stage.process(task)
    assert out.num_items == 0


def test_annotation_threshold_skips_when_all_columns_none() -> None:
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
    stage = InterleavedAnnotationThresholdFilterStage(
        blur_score_column=None,
        clip_scores_column=None,
        qrcode_ratio_column=None,
        image_text_image_num_column=None,
        image_text_word_num_column=None,
    )
    out = stage.process(task)
    assert len(out.to_pandas()) == 1


def test_annotation_threshold_none_threshold_keeps_image_and_drops_column() -> None:
    col = "sharpness"
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            col: 10.0,
        },
    ]
    task = _annotated_task(rows)
    stage = InterleavedAnnotationThresholdFilterStage(
        blur_min_sharpness=None,
        clip_scores_column=None,
        qrcode_ratio_column=None,
        image_text_image_num_column=None,
        image_text_word_num_column=None,
    )
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 1
    assert col not in out_frame.columns


def test_annotation_threshold_drops_images_when_blur_column_absent() -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 1,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "hello",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = _annotated_task(rows)
    stage = InterleavedAnnotationThresholdFilterStage(
        blur_min_sharpness=100.0,
        clip_scores_column=None,
        qrcode_ratio_column=None,
        image_text_image_num_column=None,
        image_text_word_num_column=None,
    )
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 1
    assert out_frame.iloc[0]["modality"] == "text"


def test_annotation_threshold_drops_all_when_image_text_columns_absent() -> None:
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
    task = _annotated_task(rows)
    stage = InterleavedAnnotationThresholdFilterStage(
        blur_score_column=None,
        clip_scores_column=None,
        qrcode_ratio_column=None,
        image_text_min_ratio=0.25,
        image_text_max_ratio=1.0,
    )
    out = stage.process(task)
    assert len(out.to_pandas()) == 0


def test_annotation_threshold_blur_column_filters_image() -> None:
    col = "sharpness"
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            col: 200.0,
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
            col: 10.0,
        },
    ]
    task = _annotated_task(rows)
    stage = InterleavedAnnotationThresholdFilterStage(
        blur_min_sharpness=100.0,
        clip_scores_column=None,
        qrcode_ratio_column=None,
        image_text_image_num_column=None,
        image_text_word_num_column=None,
    )
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 1
    assert col not in out_frame.columns
    assert out_frame.iloc[0]["position"] == 0


def test_annotation_threshold_qrcode_column() -> None:
    col = "qr_area_ratio"
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            col: 0.01,
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
            col: 0.5,
        },
    ]
    task = _annotated_task(rows)
    stage = InterleavedAnnotationThresholdFilterStage(
        blur_score_column=None,
        clip_scores_column=None,
        qrcode_max_area_ratio=0.05,
        image_text_image_num_column=None,
        image_text_word_num_column=None,
    )
    out = stage.process(task)
    assert len(out.to_pandas()) == 1


def test_annotation_threshold_clip_scores_column() -> None:
    col = "clip_scores"
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            col: {0: 0.2, 1: 0.05},
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
            col: {0: 0.01},
        },
    ]
    task = _annotated_task(rows)
    stage = InterleavedAnnotationThresholdFilterStage(
        blur_score_column=None,
        clip_min_score=0.15,
        qrcode_ratio_column=None,
        image_text_image_num_column=None,
        image_text_word_num_column=None,
    )
    out = stage.process(task)
    assert len(out.to_pandas()) == 1


def test_annotation_threshold_image_text_sample_level() -> None:
    img_c = "image_num"
    w_c = "text_word_num"
    rows = [
        {
            "sample_id": "ok",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "one two three four",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            img_c: pd.NA,
            w_c: pd.NA,
        },
        {
            "sample_id": "ok",
            "position": 1,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            img_c: 1,
            w_c: 4,
        },
        {
            "sample_id": "bad",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "one two three four five",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            img_c: pd.NA,
            w_c: pd.NA,
        },
        {
            "sample_id": "bad",
            "position": 1,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            img_c: 1,
            w_c: 5,
        },
    ]
    task = _annotated_task(rows)
    stage = InterleavedAnnotationThresholdFilterStage(
        blur_score_column=None,
        clip_scores_column=None,
        qrcode_ratio_column=None,
        image_text_min_ratio=0.25,
        image_text_max_ratio=1.0,
    )
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert set(out_frame["sample_id"].tolist()) == {"ok"}


def test_annotation_threshold_combines_blur_and_clip_and() -> None:
    bcol = "sharpness"
    ccol = "clip_scores"
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            bcol: 200.0,
            ccol: {0: 0.01},
        },
    ]
    task = _annotated_task(rows)
    stage = InterleavedAnnotationThresholdFilterStage(
        blur_min_sharpness=100.0,
        clip_min_score=0.15,
        qrcode_ratio_column=None,
        image_text_image_num_column=None,
        image_text_word_num_column=None,
    )
    out = stage.process(task)
    assert len(out.to_pandas()) == 0
