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

import pytest

from nemo_curator.stages.interleaved.annotation.pass_mask import (
    basic_interleaved_row_validity_mask,
    interleaved_score_pass_mask,
)
from nemo_curator.stages.interleaved.stages import BaseInterleavedScoreFilterStage

from .conftest import interleaved_task


def test_basic_validity_mask_allows_text_and_image_and_metadata() -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
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
    mask = basic_interleaved_row_validity_mask(task.to_pandas())
    assert mask.all()


def test_basic_validity_mask_rejects_unknown_modality() -> None:
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
    mask = basic_interleaved_row_validity_mask(task.to_pandas())
    assert not mask.any()


def test_basic_validity_mask_rejects_content_with_negative_position() -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "hello",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    mask = basic_interleaved_row_validity_mask(task.to_pandas())
    assert not mask.any()


def test_interleaved_score_pass_mask_unknown_stage_raises_type_error() -> None:
    from dataclasses import dataclass

    import pandas as pd

    from nemo_curator.tasks import InterleavedBatch

    @dataclass
    class _CustomStage(BaseInterleavedScoreFilterStage):
        name: str = "custom"

        def annotation_columns(self, task: InterleavedBatch, df: pd.DataFrame) -> dict:
            return {}

    task = interleaved_task([])
    stage = _CustomStage()
    with pytest.raises(TypeError, match="does not know how to combine"):
        interleaved_score_pass_mask(stage, task, task.to_pandas())
