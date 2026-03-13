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

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_curator.stages.interleaved.filter.clip_score_filter import InterleavedCLIPScoreFilterStage

from .conftest import interleaved_task, make_jpeg_bytes


def test_clip_score_filter_requires_model_dir() -> None:
    with pytest.raises(RuntimeError, match="model_dir"):
        InterleavedCLIPScoreFilterStage(model_dir=None)


@patch("nemo_curator.stages.interleaved.filter.clip_score_filter.CLIPImageEmbeddings.download_weights_on_node")
@patch("nemo_curator.stages.interleaved.filter.clip_score_filter.CLIPImageEmbeddings")
def test_clip_score_filter_process_keeps_image_when_score_above_threshold(
    mock_clip_class: MagicMock, mock_download: MagicMock
) -> None:
    mock_download.return_value = None
    dim = 512
    mock_model = mock_clip_class.return_value
    mock_model.return_value = torch.ones(1, dim) / (dim**0.5)
    mock_model.encode_text.return_value = torch.ones(2, dim) / (dim**0.5)

    jpeg = make_jpeg_bytes()
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "a cat",
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
    stage = InterleavedCLIPScoreFilterStage(model_dir="/fake/clip", min_score=0.15)
    out = stage.process(task)
    df = out.to_pandas()
    assert len(df) == 2
    assert (df["modality"] == "image").sum() == 1
    assert (df["modality"] == "text").sum() == 1


@patch("nemo_curator.stages.interleaved.filter.clip_score_filter.CLIPImageEmbeddings.download_weights_on_node")
@patch("nemo_curator.stages.interleaved.filter.clip_score_filter.CLIPImageEmbeddings")
def test_clip_score_filter_process_drops_image_when_score_below_threshold(
    mock_clip_class: MagicMock, mock_download: MagicMock
) -> None:
    mock_download.return_value = None
    dim = 512
    mock_model = mock_clip_class.return_value
    mock_model.return_value = torch.ones(1, dim) / (dim**0.5)
    mock_model.encode_text.return_value = -torch.ones(1, dim) / (dim**0.5)

    jpeg = make_jpeg_bytes()
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "unrelated",
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
    stage = InterleavedCLIPScoreFilterStage(model_dir="/fake/clip", min_score=0.15)
    out = stage.process(task)
    df = out.to_pandas()
    assert len(df) == 1
    assert (df["modality"] == "text").sum() == 1
    assert (df["modality"] == "image").sum() == 0
