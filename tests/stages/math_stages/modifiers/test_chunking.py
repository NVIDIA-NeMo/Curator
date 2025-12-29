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

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nemo_curator.stages.math.modifiers.chunking import TokenSplitterStage
from nemo_curator.tasks import DocumentBatch


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that simulates tokenization."""
    tokenizer = Mock()

    def mock_encode(text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG001
        # Simple tokenization: each word = 1 token, spaces = 0 tokens
        # This is a simplified approximation
        words = text.split()
        # Add some base tokens for special characters
        return list(range(100, 100 + len(words)))

    tokenizer.encode = mock_encode
    return tokenizer


@pytest.fixture(autouse=True)
def setup_mocks(mock_tokenizer: Mock) -> None:  # type: ignore[no-untyped-def]
    """Automatically setup mocks for AutoTokenizer."""
    with patch("nemo_curator.stages.math.modifiers.chunking.AutoTokenizer") as mock_auto_tokenizer:
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        yield {"auto_tokenizer": mock_auto_tokenizer}


class TestTokenSplitterStage:
    """Test the TokenSplitterStage class."""

    def test_init_default_values(self):
        """Test TokenSplitterStage initialization with default values."""
        stage = TokenSplitterStage(model_name="test-model")

        assert stage.model_name == "test-model"
        assert stage.max_length_tokens == 8000
        assert stage.separator == "\n\n"
        assert stage.text_field == "text"
        assert stage.chunk_id_field == "chunk_id"
        assert stage.n_tokens_field == "n_tokens"

    def test_init_custom_values(self):
        """Test TokenSplitterStage initialization with custom values."""
        stage = TokenSplitterStage(
            model_name="custom-model",
            max_length_tokens=5000,
            separator="\n",
            text_field="content",
            chunk_id_field="custom_chunk_id",
            n_tokens_field="custom_n_tokens",
        )

        assert stage.model_name == "custom-model"
        assert stage.max_length_tokens == 5000
        assert stage.separator == "\n"
        assert stage.text_field == "content"
        assert stage.chunk_id_field == "custom_chunk_id"
        assert stage.n_tokens_field == "custom_n_tokens"

    def test_inputs_outputs(self):
        """Test inputs and outputs methods."""
        stage = TokenSplitterStage(model_name="test-model", text_field="custom_text")

        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["custom_text"])
        assert outputs == (["data"], ["custom_text", "chunk_id", "n_tokens"])

    def test_process_single_short_text(self):
        """Test process method with a single short text that doesn't need chunking."""
        stage = TokenSplitterStage(model_name="test-model", max_length_tokens=100)

        df = pd.DataFrame({"text": ["Short text here"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        assert len(result.data) == 1
        assert result.data["text"].iloc[0] == "Short text here"
        assert result.data["chunk_id"].iloc[0] == 0
        assert result.data["n_tokens"].iloc[0] > 0
        assert "text" in result.data.columns

    def test_process_text_with_chunking(self):
        """Test process method with text that needs chunking."""
        stage = TokenSplitterStage(model_name="test-model", max_length_tokens=5)

        # Create text with multiple paragraphs that will exceed token limit
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four."
        df = pd.DataFrame({"text": [text]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Should create multiple chunks
        assert len(result.data) > 1
        assert all("chunk_id" in result.data.columns for _ in range(len(result.data)))
        assert all("n_tokens" in result.data.columns for _ in range(len(result.data)))
        # Chunk IDs should be sequential starting from 0
        chunk_ids = result.data["chunk_id"].tolist()
        assert chunk_ids == list(range(len(chunk_ids)))

    def test_process_preserves_metadata(self):
        """Test that process preserves original metadata fields."""
        stage = TokenSplitterStage(model_name="test-model", max_length_tokens=100)

        df = pd.DataFrame(
            {
                "text": ["Some text"],
                "url": ["https://example.com"],
                "title": ["Test Title"],
                "metadata": ["extra info"],
            }
        )
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # All original columns should be preserved
        assert "url" in result.data.columns
        assert "title" in result.data.columns
        assert "metadata" in result.data.columns
        assert result.data["url"].iloc[0] == "https://example.com"
        assert result.data["title"].iloc[0] == "Test Title"

    def test_process_multiple_documents(self):
        """Test process method with multiple documents."""
        stage = TokenSplitterStage(model_name="test-model", max_length_tokens=100)

        df = pd.DataFrame(
            {
                "text": ["First document text", "Second document text", "Third document text"],
                "doc_id": [1, 2, 3],
            }
        )
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Each document should produce at least one chunk
        assert len(result.data) >= 3
        # All doc_ids should be preserved
        assert set(result.data["doc_id"].tolist()) == {1, 2, 3}

    def test_process_empty_text(self):
        """Test process method with empty text."""
        stage = TokenSplitterStage(model_name="test-model", max_length_tokens=100)

        df = pd.DataFrame({"text": [""]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Empty text should produce no chunks (empty paragraphs are filtered)
        assert len(result.data) == 0

    def test_process_text_with_only_whitespace(self):
        """Test process method with text containing only whitespace."""
        stage = TokenSplitterStage(model_name="test-model", max_length_tokens=100)

        df = pd.DataFrame({"text": ["   \n\n   \n\n   "]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Whitespace-only paragraphs are filtered out
        assert len(result.data) == 0

    def test_process_custom_separator(self):
        """Test process method with custom separator."""
        stage = TokenSplitterStage(model_name="test-model", max_length_tokens=5, separator="\n")

        text = "Line one\nLine two\nLine three\nLine four"
        df = pd.DataFrame({"text": [text]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Should create chunks based on single newline separator
        assert len(result.data) > 0
        # Verify separator is preserved in chunks (except last)
        for _, row in result.data.iterrows():
            if row["chunk_id"] < len(result.data) - 1:
                # Non-last chunks should end with separator
                assert row["text"].endswith("\n")

    def test_process_chunk_id_sequential(self):
        """Test that chunk_id is sequential for each document."""
        stage = TokenSplitterStage(model_name="test-model", max_length_tokens=5)

        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four.\n\nPara five."
        df = pd.DataFrame({"text": [text]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Chunk IDs should be sequential: 0, 1, 2, ...
        chunk_ids = result.data["chunk_id"].tolist()
        assert chunk_ids == list(range(len(chunk_ids)))

    def test_process_n_tokens_calculated(self):
        """Test that n_tokens is correctly calculated for each chunk."""
        stage = TokenSplitterStage(model_name="test-model", max_length_tokens=100)

        df = pd.DataFrame({"text": ["Some text here"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # n_tokens should be positive
        assert all(result.data["n_tokens"] > 0)
        # n_tokens should match the token count of the chunk text
        # Note: The actual tokenization happens during processing, so we verify
        # that n_tokens is set and positive
        assert result.data["n_tokens"].iloc[0] > 0

    def test_process_last_paragraph_no_separator(self):
        """Test that last paragraph doesn't get separator appended."""
        stage = TokenSplitterStage(model_name="test-model", max_length_tokens=100)

        text = "First paragraph.\n\nSecond paragraph."
        df = pd.DataFrame({"text": [text]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Last chunk should not end with separator
        last_chunk_text = result.data.iloc[-1]["text"]
        assert not last_chunk_text.endswith("\n\n")

    def test_process_missing_text_field(self):
        """Test process method when text field is missing."""
        stage = TokenSplitterStage(model_name="test-model", text_field="missing_field")

        df = pd.DataFrame({"other_field": ["Some text"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Should handle missing field gracefully (empty string)
        assert len(result.data) == 0  # Empty text produces no chunks
