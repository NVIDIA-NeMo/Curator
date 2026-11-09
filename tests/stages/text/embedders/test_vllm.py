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

from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

with suppress(ImportError):
    from sentence_transformers import SentenceTransformer

    from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage

import numpy as np
import pyarrow as pa
import torch

from nemo_curator.tasks import DocumentBatch

# Test model that works with both VLLM and SentenceTransformer
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def sample_data() -> DocumentBatch:
    """Create sample text data for testing."""
    data = pa.table(
        {
            "id": pa.array([10, 11, 12, 13, 14], type=pa.int64()),
            "text": ["Hello world", "This is a test", "Machine learning is great", "A fourth row", "Final row"],
            "nested": pa.array([[1, 2], None, [3], [], [4, 5]], type=pa.list_(pa.int32())),
        }
    )
    return DocumentBatch(dataset_name="test_dataset", data=data)


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
        assert stage.embedding_output_dtype == "float32"
        assert stage.metadata_fields == []
        assert stage.model_inference_batch_size == 8192
        assert stage.pretokenize is True
        assert stage.verbose is False
        assert stage.model is None
        assert stage.tokenizer is None

        assert stage.inputs() == (["data"], ["text"])
        assert stage.outputs() == (["data"], ["embeddings"])

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            text_field="content",
            embedding_field="emb",
            embedding_output_dtype="float16",
            metadata_fields=["id", "content", "id"],
            model_inference_batch_size=17,
            pretokenize=True,
            cache_dir="/tmp/cache",  # noqa: S108
            hf_token="test-token",  # noqa: S106
            verbose=True,
        )

        assert stage.model_identifier == TEST_MODEL
        assert stage.text_field == "content"
        assert stage.embedding_field == "emb"
        assert stage.embedding_output_dtype == "float16"
        assert stage.metadata_fields == ["id", "content"]
        assert stage.model_inference_batch_size == 17
        assert stage.pretokenize is True
        assert stage.cache_dir == "/tmp/cache"  # noqa: S108
        assert stage.hf_token == "test-token"  # noqa: S105
        assert stage.verbose is True

        assert stage.inputs() == (["data"], ["content"])
        assert stage.outputs() == (["data"], ["id", "content", "emb"])

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
        ) -> str:
            captured.setdefault("snapshot_download_calls", []).append(
                {
                    "model_identifier": model_identifier,
                    "cache_dir": cache_dir,
                    "token": token,
                    "local_files_only": local_files_only,
                }
            )
            return f"/resolved/snapshots/{model_identifier}"

        def _fake_create_vllm_llm_with_retry(*, model: str, **kwargs: Any) -> object:  # noqa: ANN401
            captured["llm"] = {"model": model, "kwargs": kwargs}
            return object()

        import nemo_curator.stages.text.embedders.vllm as _vllm_mod

        monkeypatch.setattr(_vllm_mod, "snapshot_download", _fake_snapshot_download)
        monkeypatch.setattr(_vllm_mod, "create_vllm_llm_with_retry", _fake_create_vllm_llm_with_retry)

        stage.setup_on_node()

        # setup_on_node calls snapshot_download(local_files_only=False) to download the model
        download_call = captured["snapshot_download_calls"][0]
        assert download_call["cache_dir"] == str(cache_dir)
        assert download_call["token"] == hf_token
        assert download_call["local_files_only"] is False

        # vLLM receives the resolved snapshot path (not the repo ID)
        assert captured["llm"]["model"] == f"/resolved/snapshots/{TEST_MODEL}"
        assert captured["llm"]["kwargs"]["download_dir"] == str(cache_dir)

    def test_process_rejects_invalid_configurations(self) -> None:
        invalid_configurations = [
            (pa.table({"id": [1]}), None, "missing required text field"),
            (pa.table({"text": ["first"], "id": [1]}), ["id", "source"], "missing metadata fields"),
        ]
        for table, metadata_fields, message in invalid_configurations:
            stage = VLLMEmbeddingModelStage(
                model_identifier=TEST_MODEL,
                pretokenize=False,
                metadata_fields=metadata_fields,
            )
            stage.model = object()  # type: ignore[assignment]

            with pytest.raises(ValueError, match=message):
                stage.process(DocumentBatch(dataset_name="test_dataset", data=table))

    def test_arrow_embeddings_split_before_list_offset_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_curator.stages.text.embedders.vllm as _vllm_mod

        monkeypatch.setattr(_vllm_mod, "_MAX_LIST_ARRAY_VALUES", 7)
        embedding_matrix = np.arange(12, dtype=np.float32).reshape(4, 3)

        embeddings = VLLMEmbeddingModelStage._to_arrow_embeddings(embedding_matrix)

        assert isinstance(embeddings, pa.ChunkedArray)
        assert [len(chunk) for chunk in embeddings.chunks] == [2, 2]
        assert embeddings.type == pa.list_(pa.float32())
        assert embeddings.to_pylist() == embedding_matrix.tolist()

    @pytest.mark.parametrize(
        "embedding_case",
        [(True, "float16", pa.float16()), (False, "float32", pa.float32())],
        ids=["pretokenized-float16", "raw-text-float32"],
    )
    def test_process_batches_real_model_equivalent_to_unbatched(
        self,
        sample_data: DocumentBatch,
        embedding_case: tuple[bool, str, pa.DataType],
        reference_model: "SentenceTransformer",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bounded batches preserve embeddings, Arrow metadata, and aggregate metrics."""
        pretokenize, embedding_output_dtype, arrow_dtype = embedding_case

        class _RecordingStage(VLLMEmbeddingModelStage):
            embedded_chunk_sizes: list[int]

            def _embed_chunk(self, input_data: list[Any]) -> tuple[np.ndarray, dict[str, float]]:
                self.embedded_chunk_sizes.append(len(input_data))
                return super()._embed_chunk(input_data)

        input_table = sample_data.to_pyarrow()
        texts = input_table["text"].to_pylist()

        def _fail_to_pandas() -> None:
            pytest.fail("Arrow input must not be converted to pandas")

        monkeypatch.setattr(sample_data, "to_pandas", _fail_to_pandas)
        vllm_stage = _RecordingStage(
            model_identifier=TEST_MODEL,
            pretokenize=pretokenize,
            embedding_output_dtype=embedding_output_dtype,
            metadata_fields=["id", "nested"],
            model_inference_batch_size=None,
            verbose=False,
        )
        try:
            vllm_stage.setup_on_node()
        except Exception:  # noqa: BLE001
            pytest.skip("Skipping test due to model download failure")
        vllm_stage.setup()
        try:
            vllm_stage.embedded_chunk_sizes = []
            unbatched_result = vllm_stage.process(sample_data).to_pyarrow()
            unbatched_metrics = vllm_stage._consume_custom_metrics()
            assert vllm_stage.embedded_chunk_sizes == [5]
            assert unbatched_result["embeddings"].num_chunks == 1

            unbatched_embeddings = np.asarray(unbatched_result["embeddings"].to_pylist())
            reference_embeddings = reference_model.encode(texts)
            cosine_sim = torch.nn.functional.cosine_similarity(
                torch.tensor(unbatched_embeddings), torch.tensor(reference_embeddings), dim=1
            )
            reference_atol = 5e-5 if embedding_output_dtype == "float16" else 1e-5
            assert torch.allclose(cosine_sim, torch.ones_like(cosine_sim), atol=reference_atol)

            for batch_size, expected_chunk_sizes in [(2, [2, 2, 1]), (3, [3, 2])]:
                vllm_stage.model_inference_batch_size = batch_size
                vllm_stage.embedded_chunk_sizes = []

                result = vllm_stage.process(sample_data).to_pyarrow()
                metrics = vllm_stage._consume_custom_metrics()

                assert vllm_stage.embedded_chunk_sizes == expected_chunk_sizes
                assert result["embeddings"].num_chunks == len(expected_chunk_sizes)
                assert result.column_names == ["id", "nested", "embeddings"]
                for field_name in ["id", "nested"]:
                    assert result.schema.field(field_name).equals(input_table.schema.field(field_name))
                    assert result[field_name].equals(input_table[field_name])
                assert result.schema.field("embeddings").type == pa.list_(arrow_dtype)
                batched_embeddings = np.asarray(result["embeddings"].to_pylist())
                batch_cosine_sim = torch.nn.functional.cosine_similarity(
                    torch.tensor(batched_embeddings), torch.tensor(unbatched_embeddings), dim=1
                )
                assert torch.allclose(batch_cosine_sim, torch.ones_like(batch_cosine_sim), atol=1e-5)
                assert metrics["input_tokens"] == unbatched_metrics["input_tokens"]
                assert metrics["vllm_embedding_time"] > 0
                assert metrics["tokenization_time"] >= 0
        finally:
            vllm_stage.teardown()
