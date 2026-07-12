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

import pandas as pd
import pytest
import torch

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.interleaved.filter.clip_score_filter import (
    InterleavedCLIPScoreFilterStage,
    _sample_texts_list_from_df,
)

from .conftest import interleaved_task, make_jpeg_bytes


def test_clip_score_filter_requires_model_dir() -> None:
    stage = InterleavedCLIPScoreFilterStage(model_dir=None)
    with pytest.raises(RuntimeError, match="model_dir"):
        stage.setup_on_node(NodeInfo(), WorkerMetadata())


def test_sample_texts_list_from_df_missing_modality_column() -> None:
    sample_frame = pd.DataFrame({"sample_id": ["s1"], "text_content": ["hello"]})
    assert _sample_texts_list_from_df(sample_frame, "s1") == []


def test_sample_texts_list_from_df_missing_text_content_column() -> None:
    sample_frame = pd.DataFrame({"sample_id": ["s1"], "modality": ["text"]})
    assert _sample_texts_list_from_df(sample_frame, "s1") == []


def test_sample_texts_list_from_df_multiple_text_rows_order() -> None:
    sample_frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s1"],
            "modality": ["text", "text"],
            "text_content": ["first line", "second line"],
        }
    )
    assert _sample_texts_list_from_df(sample_frame, "s1") == ["first line", "second line"]


def test_sample_texts_list_from_df_strips_and_skips_empty() -> None:
    sample_frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s1"],
            "modality": ["text", "text", "text"],
            "text_content": ["  padded  ", "", "   "],
        }
    )
    assert _sample_texts_list_from_df(sample_frame, "s1") == ["padded"]


def test_clip_score_filter_empty_task_unchanged() -> None:
    task = interleaved_task([])
    with (
        patch("nemo_curator.stages.interleaved.filter.clip_score_filter.CLIPImageEmbeddings.download_weights_on_node"),
        patch("nemo_curator.stages.interleaved.filter.clip_score_filter.CLIPImageEmbeddings") as mock_clip_class,
    ):
        mock_clip_class.return_value.return_value = torch.zeros(1, 1)
        mock_clip_class.return_value.encode_text.return_value = torch.zeros(1, 1)
        stage = InterleavedCLIPScoreFilterStage(model_dir="/fake/clip")
        stage.setup_on_node(NodeInfo(), WorkerMetadata())
        stage.setup()
        out = stage.process(task)
    assert out.num_items == 0


@patch("nemo_curator.stages.interleaved.filter.clip_score_filter.CLIPImageEmbeddings.download_weights_on_node")
@patch("nemo_curator.stages.interleaved.filter.clip_score_filter.CLIPImageEmbeddings")
def test_clip_score_filter_drops_image_when_sample_has_no_text(
    mock_clip_class: MagicMock, mock_download: MagicMock
) -> None:
    mock_download.return_value = None
    dim = 512
    mock_model = mock_clip_class.return_value
    mock_model.return_value = torch.ones(1, dim) / (dim**0.5)
    mock_model.encode_text.return_value = torch.ones(1, dim) / (dim**0.5)

    jpeg = make_jpeg_bytes()
    rows = [
        {
            "sample_id": "solo",
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
    stage = InterleavedCLIPScoreFilterStage(model_dir="/fake/clip", min_score=0.15)
    stage.setup_on_node(NodeInfo(), WorkerMetadata())
    stage.setup()
    out = stage.process(task)
    assert out.num_items == 0
    mock_model.encode_text.assert_not_called()


@patch("nemo_curator.stages.interleaved.filter.clip_score_filter.CLIPImageEmbeddings.download_weights_on_node")
@patch("nemo_curator.stages.interleaved.filter.clip_score_filter.CLIPImageEmbeddings")
def test_clip_score_filter_passes_all_sample_texts_to_encode_text(
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
            "text_content": "alpha",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 1,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "beta",
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
            "binary_content": jpeg,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedCLIPScoreFilterStage(model_dir="/fake/clip", min_score=0.15)
    stage.setup_on_node(NodeInfo(), WorkerMetadata())
    stage.setup()
    stage.process(task)
    mock_model.encode_text.assert_called_once()
    texts_arg = mock_model.encode_text.call_args[0][0]
    assert texts_arg == ["alpha", "beta"]


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
    stage.setup_on_node(NodeInfo(), WorkerMetadata())
    stage.setup()
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 2
    assert (out_frame["modality"] == "image").sum() == 1
    assert (out_frame["modality"] == "text").sum() == 1


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
    stage.setup_on_node(NodeInfo(), WorkerMetadata())
    stage.setup()
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 1
    assert (out_frame["modality"] == "text").sum() == 1
    assert (out_frame["modality"] == "image").sum() == 0
