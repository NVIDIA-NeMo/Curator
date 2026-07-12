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

from nemo_curator.stages.interleaved.filter.image_to_text_ratio_filter import (
    InterleavedImageToTextRatioFilterStage,
    _text_word_count,
)

from .conftest import interleaved_task


def test_text_word_count_none_is_zero() -> None:
    assert _text_word_count(None) == 0


def test_text_word_count_nan_float_is_zero() -> None:
    assert _text_word_count(float("nan")) == 0


def test_text_word_count_splits_on_whitespace() -> None:
    assert _text_word_count("  one   two three  ") == 3


def test_image_to_text_ratio_empty_task_unchanged() -> None:
    task = interleaved_task([])
    stage = InterleavedImageToTextRatioFilterStage()
    out = stage.process(task)
    assert out.num_items == 0


def test_image_to_text_ratio_image_only_uses_denominator_one() -> None:
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
    stage = InterleavedImageToTextRatioFilterStage(min_ratio=1.5, max_ratio=2.5)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 2


def test_image_to_text_ratio_boundary_inclusive_min_and_max() -> None:
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
    stage = InterleavedImageToTextRatioFilterStage(min_ratio=0.25, max_ratio=0.25)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 2
    assert out.num_items == 1


def test_image_to_text_ratio_metadata_only_sample_dropped_as_orphan() -> None:
    rows = [
        {
            "sample_id": "meta_only",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedImageToTextRatioFilterStage()
    out = stage.process(task)
    assert out.num_items == 0


def test_image_to_text_ratio_text_rows_contribute_word_count_across_rows() -> None:
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
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "three four",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 2,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedImageToTextRatioFilterStage(min_ratio=0.15, max_ratio=0.25)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 3
    assert out.num_items == 1


def test_image_to_text_ratio_no_sample_id_passthrough() -> None:
    # InterleavedBatch requires sample_id; test content_keep_mask with frame missing sample_id.
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
        {
            "sample_id": "x",
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
    task_frame = task.to_pandas().drop(columns=["sample_id"])
    stage = InterleavedImageToTextRatioFilterStage(min_ratio=0.0, max_ratio=1.0)
    keep = stage.content_keep_mask(task, task_frame)
    assert keep.all()


def test_image_to_text_ratio_ratio_in_range_kept() -> None:
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
    stage = InterleavedImageToTextRatioFilterStage(min_ratio=0.2, max_ratio=1.0)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 2


def test_image_to_text_ratio_ratio_below_min_dropped() -> None:
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
    stage = InterleavedImageToTextRatioFilterStage(min_ratio=1.0, max_ratio=2.0)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 0


def test_image_to_text_ratio_ratio_above_max_dropped() -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "x",
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
        {
            "sample_id": "s1",
            "position": 2,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedImageToTextRatioFilterStage(min_ratio=0.0, max_ratio=1.0)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 0


def test_image_to_text_ratio_multiple_samples_one_dropped() -> None:
    rows = [
        {
            "sample_id": "keep",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "a b",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "keep",
            "position": 1,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "drop",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "one two three",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "drop",
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
    stage = InterleavedImageToTextRatioFilterStage(min_ratio=0.4, max_ratio=0.6)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 2
    assert set(out_frame["sample_id"].tolist()) == {"keep"}
