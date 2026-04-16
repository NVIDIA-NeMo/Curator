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

import numpy as np
import pandas as pd

from nemo_curator.stages.interleaved.annotation.blur_annotator import (
    InterleavedBlurAnnotatorStage,
    _sharpness_score,
)
from nemo_curator.stages.interleaved.annotation.pass_mask import interleaved_score_pass_mask

from .conftest import interleaved_task, make_jpeg_bytes


def test_sharpness_score_solid_image_is_zero() -> None:
    arr = np.full((10, 10, 3), 100, dtype=np.uint8)
    assert _sharpness_score(arr) == 0.0


def test_sharpness_score_high_frequency_is_positive() -> None:
    rng = np.random.default_rng()
    arr = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    assert _sharpness_score(arr) > 0.0


def test_blur_annotator_text_only_no_sharpness_column_filled() -> None:
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
    stage = InterleavedBlurAnnotatorStage()
    out_frame = stage.process(task).to_pandas()
    col = f"{stage.name}_sharpness"
    assert len(out_frame) == 1
    assert col in out_frame.columns
    assert pd.isna(out_frame.iloc[0][col])


def test_blur_annotator_does_not_drop_blurry_image() -> None:
    jpeg = make_jpeg_bytes(sharp=False)
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
    stage = InterleavedBlurAnnotatorStage(score_threshold=1e6)
    out_frame = stage.process(task).to_pandas()
    assert len(out_frame) == 1


def test_blur_annotator_sharp_image_has_high_sharpness_score() -> None:
    jpeg = make_jpeg_bytes(sharp=True)
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
    stage = InterleavedBlurAnnotatorStage(score_threshold=0.0)
    out_frame = stage.process(task).to_pandas()
    col = f"{stage.name}_sharpness"
    assert out_frame.iloc[0][col] > 0


def test_blur_annotator_blurry_image_has_low_sharpness_score() -> None:
    jpeg = make_jpeg_bytes(sharp=False)
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
    stage = InterleavedBlurAnnotatorStage()
    out_frame = stage.process(task).to_pandas()
    col = f"{stage.name}_sharpness"
    assert float(out_frame.iloc[0][col]) < stage.score_threshold


def test_blur_annotator_column_name_uses_stage_name() -> None:
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
    stage = InterleavedBlurAnnotatorStage()
    out_frame = stage.process(task).to_pandas()
    assert f"{stage.name}_sharpness" in out_frame.columns


def test_blur_annotator_none_bytes_gives_na_sharpness() -> None:
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

    with patch.object(InterleavedBlurAnnotatorStage, "iter_materialized_bytes", _none_bytes):
        stage = InterleavedBlurAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
    assert len(out_frame) == 1
    assert pd.isna(out_frame.iloc[0][f"{stage.name}_sharpness"])


def test_blur_annotator_empty_task_unchanged() -> None:
    task = interleaved_task([])
    stage = InterleavedBlurAnnotatorStage()
    assert stage.process(task).num_items == 0


def test_blur_annotator_pass_mask_sharp_image_passes() -> None:
    jpeg = make_jpeg_bytes(sharp=True)
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
    stage = InterleavedBlurAnnotatorStage(score_threshold=0.0)
    mask = interleaved_score_pass_mask(stage, task, task.to_pandas())
    assert mask.all()


def test_blur_annotator_pass_mask_blurry_image_fails() -> None:
    jpeg = make_jpeg_bytes(sharp=False)
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
    stage = InterleavedBlurAnnotatorStage(score_threshold=1e6)
    mask = interleaved_score_pass_mask(stage, task, task.to_pandas())
    assert not mask.any()


def test_blur_annotator_pass_mask_text_rows_always_pass() -> None:
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
    stage = InterleavedBlurAnnotatorStage(score_threshold=1e6)
    mask = interleaved_score_pass_mask(stage, task, task.to_pandas())
    assert mask.all()


def test_blur_annotator_pass_mask_invalid_rows_excluded_when_drop_invalid_rows() -> None:
    jpeg = make_jpeg_bytes(sharp=True)
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "audio",
            "content_type": "audio/wav",
            "text_content": None,
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
            "binary_content": jpeg,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedBlurAnnotatorStage(score_threshold=0.0)
    df = task.to_pandas()
    mask = interleaved_score_pass_mask(stage, task, df, drop_invalid_rows=True)
    assert not bool(mask.loc[df["modality"] == "audio"].iloc[0])
    assert bool(mask.loc[df["modality"] == "image"].iloc[0])


def test_blur_annotator_pass_mask_drop_invalid_rows_false_keeps_all() -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "audio",
            "content_type": "audio/wav",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedBlurAnnotatorStage(score_threshold=1e6)
    mask = interleaved_score_pass_mask(stage, task, task.to_pandas(), drop_invalid_rows=False)
    assert mask.all()
