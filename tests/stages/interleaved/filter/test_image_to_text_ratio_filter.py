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
)

from .conftest import interleaved_task


def test_image_to_text_ratio_no_sample_id_passthrough() -> None:
    # InterleavedBatch requires sample_id; test content_keep_mask with df missing sample_id.
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
    df = task.to_pandas().drop(columns=["sample_id"])
    stage = InterleavedImageToTextRatioFilterStage(min_ratio=0.0, max_ratio=1.0)
    keep = stage.content_keep_mask(task, df)
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
    df = out.to_pandas()
    assert len(df) == 2


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
    df = out.to_pandas()
    assert len(df) == 0


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
    df = out.to_pandas()
    assert len(df) == 0


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
    df = out.to_pandas()
    assert len(df) == 2
    assert set(df["sample_id"].tolist()) == {"keep"}
