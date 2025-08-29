# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import (
    ATTENTION_MASK_COLUMN,
    INPUT_ID_COLUMN,
    SEQ_ORDER_COLUMN,
    TOKEN_LENGTH_COLUMN,
)
from nemo_curator.tasks import DocumentBatch


class MockTokenizerOutput:
    def __init__(self, input_ids: list[list[int]], attention_mask: list[list[int]]) -> None:
        self.input_ids = np.array(input_ids)
        self.attention_mask = np.array(attention_mask)


@pytest.fixture
def mock_tokenizer() -> Mock:
    tokenizer = Mock()
    tokenizer.model_max_length = 512
    tokenizer.pad_token = "[PAD]"  # noqa: S105
    tokenizer.unk_token = "[UNK]"  # noqa: S105
    tokenizer.padding_side = "right"

    def mock_batch_encode_plus(texts: list[str], **kwargs: Any) -> MockTokenizerOutput:  # noqa: ANN401
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        max_length = kwargs.get("max_length", 512)

        for text in texts:
            # Simulate tokenization: longer texts get more tokens
            token_count = min(len(text.split()) + 2, max_length)  # +2 for special tokens

            # Create input IDs (101 = [CLS], 102 = [SEP], others are mock token IDs)
            ids = [101, *range(1000, 1000 + token_count - 2), 102]
            # Pad to max_length
            ids = ids + [0] * (max_length - len(ids))
            ids = ids[:max_length]

            mask = [1] * token_count + [0] * (max_length - token_count)
            mask = mask[:max_length]

            input_ids.append(ids)
            attention_masks.append(mask)

        return MockTokenizerOutput(input_ids, attention_masks)

    tokenizer.batch_encode_plus = mock_batch_encode_plus
    return tokenizer


@pytest.fixture
def sample_document_batch() -> DocumentBatch:
    data = pd.DataFrame(
        {
            "text": [
                "This is a short text.",
                "This is a much longer text with many more words to test the sorting functionality.",
                "Medium length text for testing purposes.",
            ]
        }
    )

    return DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=data)


@pytest.fixture(autouse=True)
def setup_mocks(mock_tokenizer: Mock):
    with (
        patch("nemo_curator.stages.text.models.tokenizer.AutoTokenizer") as mock_auto_tokenizer,
        patch("nemo_curator.stages.text.models.tokenizer.AutoConfig") as mock_auto_config,
        patch("nemo_curator.stages.text.models.tokenizer.snapshot_download") as mock_snapshot_download,
    ):
        # Setup AutoTokenizer mock
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup AutoConfig mock
        config = Mock()
        config.max_position_embeddings = 512
        mock_auto_config.from_pretrained.return_value = config

        # snapshot_download doesn't need special setup, just needs to not fail

        yield {
            "auto_tokenizer": mock_auto_tokenizer,
            "auto_config": mock_auto_config,
            "snapshot_download": mock_snapshot_download,
        }


def test_mocks_are_working_automatically():
    # This test can create a TokenizerStage and call setup() without any issues
    # because the setup_mocks fixture is automatically applied due to autouse=True
    stage = TokenizerStage(model_identifier="test/model")

    # This would fail without the mocks being active
    stage.setup()

    # Verify the tokenizer was mocked correctly
    assert stage.tokenizer is not None
    assert hasattr(stage.tokenizer, "batch_encode_plus")


def test_tokenizer_stage_sort_by_length_enabled(sample_document_batch: DocumentBatch):
    stage = TokenizerStage(model_identifier="test/model", sort_by_length=True, text_field="text")

    stage.setup()
    result = stage.process(sample_document_batch).to_pandas()

    assert INPUT_ID_COLUMN in result.columns
    assert ATTENTION_MASK_COLUMN in result.columns
    assert SEQ_ORDER_COLUMN in result.columns
    assert "text" in result.columns

    # Verify that TOKEN_LENGTH_COLUMN was removed after sorting
    assert TOKEN_LENGTH_COLUMN not in result.columns

    # Verify that data is sorted by token length (ascending order)
    token_lengths = []
    for mask in result[ATTENTION_MASK_COLUMN]:
        token_lengths.append(sum(mask))
    assert token_lengths == sorted(token_lengths)

    # Verify that SEQ_ORDER_COLUMN preserves original order information
    assert len(set(result[SEQ_ORDER_COLUMN])) == len(result)
    assert all(0 <= order < len(sample_document_batch.to_pandas()) for order in result[SEQ_ORDER_COLUMN])


def test_tokenizer_stage_sort_by_length_disabled(sample_document_batch: DocumentBatch):
    stage = TokenizerStage(model_identifier="test/model", sort_by_length=False, text_field="text")

    stage.setup()
    result = stage.process(sample_document_batch).to_pandas()

    assert INPUT_ID_COLUMN in result.columns
    assert ATTENTION_MASK_COLUMN in result.columns
    assert "text" in result.columns

    # Verify that SEQ_ORDER_COLUMN does NOT exist when sorting is disabled
    assert SEQ_ORDER_COLUMN not in result.columns

    # Verify that TOKEN_LENGTH_COLUMN does NOT exist when sorting is disabled
    assert TOKEN_LENGTH_COLUMN not in result.columns

    # Verify that the order is preserved (same as input)
    original_texts = sample_document_batch.to_pandas()["text"].tolist()
    result_texts = result["text"].tolist()
    assert original_texts == result_texts


def test_tokenizer_stage_max_chars_truncation():
    data = pd.DataFrame(
        {"text": ["This is a very long text that should be truncated when max_chars is set to a small value"]}
    )
    batch = DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=data)

    stage = TokenizerStage(model_identifier="test/model", max_chars=20, sort_by_length=False, text_field="text")

    stage.setup()
    result = stage.process(batch).to_pandas()

    truncated_text = result["text"].iloc[0]
    assert len(truncated_text) <= 20
    assert truncated_text == "This is a very long "


def test_tokenizer_stage_setup_unk_token():
    stage = TokenizerStage(model_identifier="test/model", unk_token=True)

    stage.setup()
    assert stage.tokenizer.pad_token == stage.tokenizer.unk_token


def test_tokenizer_stage_max_seq_length_fallback(setup_mocks: dict[str, Mock]):
    mock_tokenizer = setup_mocks["auto_tokenizer"].from_pretrained.return_value
    mock_tokenizer.model_max_length = 999999999

    stage = TokenizerStage(model_identifier="test/model", max_seq_length=None)

    stage.setup()
    assert stage.max_seq_length == 512
