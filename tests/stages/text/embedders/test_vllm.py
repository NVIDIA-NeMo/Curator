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

# ruff: noqa: E402
from contextlib import suppress
from pathlib import Path
from typing import Any

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

    def test_llm_uses_cache_dir_for_download(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

        captured: dict[str, Any] = {}

        def _fake_snapshot_download(
            model_identifier: str,
            cache_dir: str | None = None,
            token: str | None = None,
            local_files_only: bool | None = None,
        ) -> None:
            captured["snapshot_download"] = {
                "model_identifier": model_identifier,
                "cache_dir": cache_dir,
                "token": token,
                "local_files_only": local_files_only,
            }

        class _FakeLLM:
            def __init__(self, model: str, **kwargs: Any) -> None:  # noqa: ANN401
                captured["llm"] = {"model": model, "kwargs": kwargs}

        monkeypatch.setattr("nemo_curator.stages.text.embedders.vllm.snapshot_download", _fake_snapshot_download)
        monkeypatch.setattr("nemo_curator.stages.text.embedders.vllm.LLM", _FakeLLM)

        stage.setup_on_node()

        assert captured["snapshot_download"]["cache_dir"] == str(cache_dir)
        assert captured["snapshot_download"]["token"] == hf_token
        assert captured["snapshot_download"]["local_files_only"] is False

        assert captured["llm"]["model"] == TEST_MODEL
        assert captured["llm"]["kwargs"]["download_dir"] == str(cache_dir)

    @pytest.mark.parametrize("pretokenize", [True, False])
    def test_vllm_produces_valid_embeddings(
        self, sample_data: DocumentBatch, pretokenize: bool, reference_model: "SentenceTransformer"
    ) -> None:
        """Test that VLLM produces embeddings matching SentenceTransformer reference."""
        vllm_stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=pretokenize,
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
