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
from nemo_curator.stages.interleaved.annotation.clip_score_annotator import (
    InterleavedCLIPScoreAnnotatorStage,
    _sample_text_positions_and_texts,
)
from nemo_curator.stages.interleaved.annotation.pass_mask import interleaved_score_pass_mask

from .conftest import interleaved_task, make_jpeg_bytes


def test_clip_score_annotator_requires_model_dir() -> None:
    stage = InterleavedCLIPScoreAnnotatorStage(model_dir=None)
    with pytest.raises(RuntimeError, match="model_dir"):
        stage.setup_on_node(NodeInfo(), WorkerMetadata())


def test_sample_text_positions_and_texts_missing_columns() -> None:
    df = pd.DataFrame({"sample_id": ["s1"], "text_content": ["hello"]})
    positions, texts = _sample_text_positions_and_texts(df, "s1")
    assert positions == []
    assert texts == []


def test_sample_text_positions_and_texts_multiple_rows() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s1"],
            "modality": ["text", "text"],
            "position": [0, 1],
            "text_content": ["first", "second"],
        }
    )
    positions, texts = _sample_text_positions_and_texts(df, "s1")
    assert positions == [0, 1]
    assert texts == ["first", "second"]


def test_sample_text_positions_and_texts_strips_and_skips_empty() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s1"],
            "modality": ["text", "text", "text"],
            "position": [0, 1, 2],
            "text_content": ["  padded  ", "", "   "],
        }
    )
    positions, texts = _sample_text_positions_and_texts(df, "s1")
    assert positions == [0]
    assert texts == ["padded"]


def test_clip_score_annotator_empty_task_unchanged() -> None:
    task = interleaved_task([])
    with (
        patch(
            "nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings.download_weights_on_node"
        ),
        patch(
            "nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings"
        ) as mock_clip_class,
    ):
        mock_clip_class.return_value.return_value = torch.zeros(1, 1)
        mock_clip_class.return_value.encode_text.return_value = torch.zeros(1, 1)
        stage = InterleavedCLIPScoreAnnotatorStage(model_dir="/fake/clip")
        stage.setup_on_node(NodeInfo(), WorkerMetadata())
        stage.setup()
        out = stage.process(task)
    assert out.num_items == 0


@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings.download_weights_on_node")
@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings")
def test_clip_score_annotator_does_not_drop_rows(mock_clip_class: MagicMock, mock_download: MagicMock) -> None:
    mock_download.return_value = None
    dim = 512
    mock_model = mock_clip_class.return_value
    mock_model.return_value = torch.ones(1, dim) / (dim**0.5)
    mock_model.encode_text.return_value = torch.ones(1, dim) / (dim**0.5)

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
    stage = InterleavedCLIPScoreAnnotatorStage(model_dir="/fake/clip")
    stage.setup_on_node(NodeInfo(), WorkerMetadata())
    stage.setup()
    out_frame = stage.process(task).to_pandas()
    assert len(out_frame) == 2


@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings.download_weights_on_node")
@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings")
def test_clip_score_annotator_no_text_image_gets_na(mock_clip_class: MagicMock, mock_download: MagicMock) -> None:
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
    stage = InterleavedCLIPScoreAnnotatorStage(model_dir="/fake/clip")
    stage.setup_on_node(NodeInfo(), WorkerMetadata())
    stage.setup()
    out_frame = stage.process(task).to_pandas()
    col = f"{stage.name}_clip_scores"
    assert col in out_frame.columns
    assert pd.isna(out_frame.iloc[0][col])
    mock_model.encode_text.assert_not_called()


@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings.download_weights_on_node")
@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings")
def test_clip_score_annotator_stores_dict_per_image_row(mock_clip_class: MagicMock, mock_download: MagicMock) -> None:
    mock_download.return_value = None
    dim = 512
    mock_model = mock_clip_class.return_value
    mock_model.return_value = torch.ones(1, dim) / (dim**0.5)
    mock_model.encode_text.return_value = torch.ones(1, dim) / (dim**0.5)

    jpeg = make_jpeg_bytes()
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
    stage = InterleavedCLIPScoreAnnotatorStage(model_dir="/fake/clip")
    stage.setup_on_node(NodeInfo(), WorkerMetadata())
    stage.setup()
    out_frame = stage.process(task).to_pandas()
    col = f"{stage.name}_clip_scores"
    img_row = out_frame[out_frame["modality"] == "image"].iloc[0]
    assert isinstance(img_row[col], dict)
    assert 0 in img_row[col]


@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings.download_weights_on_node")
@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings")
def test_clip_score_annotator_column_name_uses_stage_name(
    mock_clip_class: MagicMock,  # noqa: ARG001
    mock_download: MagicMock,
) -> None:
    mock_download.return_value = None
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
    stage = InterleavedCLIPScoreAnnotatorStage(model_dir="/fake/clip")
    stage.setup_on_node(NodeInfo(), WorkerMetadata())
    stage.setup()
    out_frame = stage.process(task).to_pandas()
    assert f"{stage.name}_clip_scores" in out_frame.columns


@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings.download_weights_on_node")
@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings")
def test_clip_score_annotator_pass_mask_high_score_passes(
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
    stage = InterleavedCLIPScoreAnnotatorStage(model_dir="/fake/clip")
    stage.setup_on_node(NodeInfo(), WorkerMetadata())
    stage.setup()
    mask = interleaved_score_pass_mask(stage, task, task.to_pandas(), min_score=0.0)
    assert mask.all()


@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings.download_weights_on_node")
@patch("nemo_curator.stages.interleaved.annotation.clip_score_annotator.CLIPImageEmbeddings")
def test_clip_score_annotator_pass_mask_low_score_fails(mock_clip_class: MagicMock, mock_download: MagicMock) -> None:
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
    stage = InterleavedCLIPScoreAnnotatorStage(model_dir="/fake/clip")
    stage.setup_on_node(NodeInfo(), WorkerMetadata())
    stage.setup()
    df = task.to_pandas()
    mask = interleaved_score_pass_mask(stage, task, df, min_score=0.15)
    assert not bool(mask.loc[df["modality"] == "image"].iloc[0])
    assert bool(mask.loc[df["modality"] == "text"].iloc[0])
