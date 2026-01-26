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

import gc
from contextlib import suppress
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

with suppress(ImportError):
    from sentence_transformers import SentenceTransformer

    from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage

import numpy as np
import pandas as pd
import torch

from nemo_curator.tasks import DocumentBatch

# Test model that works with both VLLM and SentenceTransformer
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def sample_data() -> DocumentBatch:
    """Create sample text data for testing."""
    texts = ["Hello world", "This is a test", "Machine learning is great"]
    data = pd.DataFrame({"text": texts})
    return DocumentBatch(task_id="test_batch", dataset_name="test_dataset", data=data)


@pytest.fixture(scope="module")
def reference_model() -> "SentenceTransformer":
    """Load SentenceTransformer model once for the module."""
    return SentenceTransformer(TEST_MODEL).to("cuda")


@pytest.mark.gpu
class TestVLLMEmbeddingModelStage:
    """Test VLLMEmbeddingModelStage initialization and processing."""

    @pytest.fixture(autouse=True)
    def cleanup_vllm(self) -> None:
        """Clean up vLLM resources after each test to prevent Ray cluster corruption.

        vLLM uses Ray internally, and its LLM destructor communicates with Ray workers.
        If not properly cleaned up, it can leave the shared Ray cluster in a bad state,
        causing subsequent tests to fail with GCS connection errors.
        """
        yield
        # Force garbage collection to trigger LLM destructor before Ray state becomes stale
        gc.collect()
        # Clear CUDA cache to release GPU memory
        torch.cuda.empty_cache()

    def test_default_initialization(self) -> None:
        """Test initialization with default parameters."""
        stage = VLLMEmbeddingModelStage(model_identifier=TEST_MODEL)

        assert stage.model_identifier == TEST_MODEL
        assert stage.text_field == "text"
        assert stage.embedding_field == "embeddings"
        assert stage.pretokenize is False
        assert stage.verbose is False
        assert stage.model is None
        assert stage.tokenizer is None

        assert stage.inputs() == (["data"], ["text"])
        assert stage.outputs() == (["data"], ["text", "embeddings"])

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            text_field="content",
            embedding_field="emb",
            pretokenize=True,
            cache_dir="/tmp/cache",  # noqa: S108
            hf_token="test-token",  # noqa: S106
            verbose=True,
        )

        assert stage.model_identifier == TEST_MODEL
        assert stage.text_field == "content"
        assert stage.embedding_field == "emb"
        assert stage.pretokenize is True
        assert stage.cache_dir == "/tmp/cache"  # noqa: S108
        assert stage.hf_token == "test-token"  # noqa: S105
        assert stage.verbose is True

        assert stage.inputs() == (["data"], ["content"])
        assert stage.outputs() == (["data"], ["content", "emb"])

        assert stage.resources.gpus == 1
        assert stage.resources.cpus == 1

    def test_llm_uses_cache_dir_for_download(self, tmp_path: Path) -> None:
        """Ensure vLLM receives download_dir so weights reuse snapshot cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        hf_token = "test-token"  # noqa: S105

        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            cache_dir=str(cache_dir),
            hf_token=hf_token,
            verbose=True,
        )

        with (
            patch("nemo_curator.stages.text.embedders.vllm.snapshot_download") as mock_snapshot_download,
            patch("nemo_curator.stages.text.embedders.vllm.LLM") as mock_llm,
        ):
            stage.setup_on_node()

            mock_snapshot_download.assert_called_once()
            assert mock_snapshot_download.call_args.kwargs["cache_dir"] == str(cache_dir)
            assert mock_snapshot_download.call_args.kwargs["token"] == hf_token
            assert mock_snapshot_download.call_args.kwargs["local_files_only"] is False

            mock_llm.assert_called_once()
            assert mock_llm.call_args.kwargs["model"] == TEST_MODEL
            assert mock_llm.call_args.kwargs["download_dir"] == str(cache_dir)

    def test_vllm_produces_valid_embeddings_pretokenize_false(
        self, sample_data: DocumentBatch, reference_model: "SentenceTransformer"
    ) -> None:
        """Test that VLLM produces embeddings matching SentenceTransformer reference."""
        vllm_stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=False,
            verbose=False,
        )
        try:
            vllm_stage.setup_on_node()
        except Exception:  # noqa: BLE001
            pytest.skip("Skipping test due to model download failure")
        vllm_stage.setup()
        result = vllm_stage.process(sample_data)

        assert isinstance(result, DocumentBatch)
        result_df = result.to_pandas()
        assert "embeddings" in result_df.columns
        assert len(result_df) == 3

        reference_embeddings = reference_model.encode(sample_data.to_pandas()["text"].tolist())
        vllm_embeddings = np.array(result_df["embeddings"].tolist())

        vllm_embeddings_torch = torch.tensor(vllm_embeddings)
        reference_embeddings_torch = torch.tensor(reference_embeddings)

        cosine_sim = torch.nn.functional.cosine_similarity(vllm_embeddings_torch, reference_embeddings_torch, dim=1)
        assert torch.allclose(cosine_sim, torch.ones_like(cosine_sim), atol=1e-5)

        # Explicit cleanup: delete the vLLM model before test ends to prevent Ray state corruption
        del vllm_stage.model
        vllm_stage.model = None

    def test_pretokenize_uses_tokenizer_and_tokens_prompt(self, sample_data: DocumentBatch) -> None:
        """Test pretokenization path without running full vLLM."""
        vllm_stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=True,
            verbose=False,
        )

        captured_embed_input: list[Any] = []

        def _fake_embed(input_data: list[Any], **_: object) -> list[Any]:
            captured_embed_input.extend(input_data)
            return [Mock(outputs=Mock(embedding=[0.1, 0.2, 0.3])) for _ in range(len(input_data))]

        with (
            patch.object(vllm_stage, "tokenizer") as mock_tokenizer,
            patch.object(vllm_stage, "model") as mock_model,
        ):
            mock_tokenizer.batch_encode_plus.return_value = Mock(
                input_ids=[[1, 2, 3] for _ in sample_data.data["text"]]
            )
            mock_model.model_config.max_model_len = 16
            mock_model.embed.side_effect = _fake_embed

            result = vllm_stage.process(sample_data)

            mock_tokenizer.batch_encode_plus.assert_called_once()
            mock_model.embed.assert_called_once()

        assert isinstance(result, DocumentBatch)
        result_df = result.to_pandas()
        assert result_df["embeddings"].apply(len).tolist() == [3, 3, 3]

        assert all(isinstance(item, dict) for item in captured_embed_input)
        assert all("prompt_token_ids" in item for item in captured_embed_input)
        assert all(isinstance(item["prompt_token_ids"], list) for item in captured_embed_input)
