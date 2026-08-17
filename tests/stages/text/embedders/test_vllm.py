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
from types import SimpleNamespace
from typing import Any

import pytest

with suppress(ImportError):
    from sentence_transformers import SentenceTransformer

    from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from nemo_curator.tasks import DocumentBatch

# Test model that works with both VLLM and SentenceTransformer
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def batch_encode_plus(
        self,
        input_data: list[str],
        *,
        truncation: bool,
        max_length: int,
    ) -> SimpleNamespace:
        assert truncation is True
        assert max_length == 512
        self.calls.append(input_data)
        return SimpleNamespace(input_ids=[[int(text), int(text) + 100] for text in input_data])


class _FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.model_config = SimpleNamespace(max_model_len=512)

    def embed(
        self,
        input_data: list[Any],
        *,
        tokenization_kwargs: dict[str, int],
        use_tqdm: bool,
    ) -> list[SimpleNamespace]:
        assert tokenization_kwargs == {"truncate_prompt_tokens": -1}
        assert use_tqdm is False
        row_ids = [
            int(prompt["prompt_token_ids"][0]) if isinstance(prompt, dict) else int(prompt) for prompt in input_data
        ]
        self.calls.append(row_ids)
        return [
            SimpleNamespace(
                prompt_token_ids=[row_id, row_id + 100],
                outputs=SimpleNamespace(embedding=[float(row_id), 1.0]),
            )
            for row_id in row_ids
        ]


@pytest.fixture
def sample_data() -> DocumentBatch:
    """Create sample text data for testing."""
    texts = ["Hello world", "This is a test", "Machine learning is great"]
    data = pd.DataFrame({"text": texts})
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
        assert stage.metadata_fields is None
        assert stage.model_inference_batch_size == 10_000
        assert stage.pretokenize is True
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

    def test_new_options_preserve_existing_positional_arguments(self) -> None:
        stage = VLLMEmbeddingModelStage(
            TEST_MODEL,
            {"gpu_memory_utilization": 0.5},
            "content",
            False,
            "vector",
            123,
            "/tmp/cache",  # noqa: S108
            "test-token",
            True,
        )

        assert stage.vllm_init_kwargs == {"gpu_memory_utilization": 0.5}
        assert stage.text_field == "content"
        assert stage.pretokenize is False
        assert stage.embedding_field == "vector"
        assert stage.max_chars == 123
        assert stage.cache_dir == "/tmp/cache"  # noqa: S108
        assert stage.hf_token == "test-token"  # noqa: S105
        assert stage.verbose is True

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

    @pytest.mark.parametrize("batch_size", [None, True, 1.5, 0, -1])
    def test_rejects_invalid_model_inference_batch_size(self, batch_size: Any) -> None:  # noqa: ANN401
        with pytest.raises(ValueError, match="model_inference_batch_size must be a positive integer"):
            VLLMEmbeddingModelStage(
                model_identifier=TEST_MODEL,
                model_inference_batch_size=batch_size,
            )

    @pytest.mark.parametrize("pretokenize", [True, False])
    def test_process_rejects_uninitialized_model(self, pretokenize: bool) -> None:
        stage = VLLMEmbeddingModelStage(model_identifier=TEST_MODEL, pretokenize=pretokenize)
        batch = DocumentBatch(dataset_name="test_dataset", data=pa.table({"text": ["first"]}))

        with pytest.raises(ValueError, match="vLLM model is not initialized"):
            stage.process(batch)

    def test_process_rejects_empty_batch(self) -> None:
        stage = VLLMEmbeddingModelStage(model_identifier=TEST_MODEL, pretokenize=False)
        stage.model = SimpleNamespace()
        batch = DocumentBatch(
            dataset_name="test_dataset",
            data=pa.table({"text": pa.array([], type=pa.string())}),
        )

        with pytest.raises(ValueError, match="empty document batch"):
            stage.process(batch)

    def test_process_rejects_uninitialized_tokenizer(self) -> None:
        stage = VLLMEmbeddingModelStage(model_identifier=TEST_MODEL, pretokenize=True)
        stage.model = _FakeEmbeddingModel()  # type: ignore[assignment]
        batch = DocumentBatch(dataset_name="test_dataset", data=pa.table({"text": ["0"]}))

        with pytest.raises(ValueError, match="Tokenizer is not initialized"):
            stage.process(batch)

    @pytest.mark.parametrize("pretokenize", [True, False])
    def test_process_chunks_rows_in_both_pretokenization_modes(self, pretokenize: bool) -> None:
        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=pretokenize,
            metadata_fields=["id"],
            model_inference_batch_size=2,
        )
        model = _FakeEmbeddingModel()
        tokenizer = _FakeTokenizer()
        stage.model = model  # type: ignore[assignment]
        stage.tokenizer = tokenizer  # type: ignore[assignment]
        batch = DocumentBatch(
            dataset_name="test_dataset",
            data=pa.table({"text": ["0", "1", "2", "3", "4"], "id": [10, 11, 12, 13, 14]}),
        )

        result = stage.process(batch)

        assert model.calls == [[0, 1], [2, 3], [4]]
        assert tokenizer.calls == ([["0", "1"], ["2", "3"], ["4"]] if pretokenize else [])
        assert isinstance(result.data, pa.Table)
        assert result.data.column_names == ["id", "embeddings"]
        assert result.data["id"].to_pylist() == [10, 11, 12, 13, 14]
        assert result.data["embeddings"].to_pylist() == [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
        ]

    def test_process_preserves_configured_arrow_columns_without_pandas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        input_table = pa.table(
            {
                "id": pa.array([7, 8], type=pa.int64()),
                "text": pa.array(["00", "11"], type=pa.string()),
                "nullable": pa.array(["value", None], type=pa.string()),
                "nested": pa.array([[1, 2], None], type=pa.list_(pa.int32())),
            }
        )
        batch = DocumentBatch(dataset_name="test_dataset", data=input_table)

        def _fail_to_pandas() -> None:
            pytest.fail("Arrow input must not be converted to pandas")

        monkeypatch.setattr(batch, "to_pandas", _fail_to_pandas)
        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=False,
            metadata_fields=["nested", "nullable", "text", "id"],
            model_inference_batch_size=1,
            max_chars=1,
        )
        model = _FakeEmbeddingModel()
        stage.model = model  # type: ignore[assignment]

        result = stage.process(batch)

        assert isinstance(result.data, pa.Table)
        assert result.data.column_names == ["nested", "nullable", "text", "id", "embeddings"]
        assert model.calls == [[0], [1]]
        for field_name in ["nested", "nullable", "text", "id"]:
            assert result.data.schema.field(field_name).equals(input_table.schema.field(field_name))
            assert result.data[field_name].equals(input_table[field_name])
        assert result.data.schema.field("embeddings").type == pa.list_(pa.float32())

    def test_process_prepares_exactly_one_chunk_ahead(self) -> None:
        from threading import Event

        second_chunk_prepared = Event()
        third_chunk_prepared = Event()

        class _TrackingStage(VLLMEmbeddingModelStage):
            def _prepare_input_chunk(
                self,
                text_column: pa.ChunkedArray,
                offset: int,
                chunk_size: int,
            ) -> tuple[list[Any], float]:
                prepared = super()._prepare_input_chunk(text_column, offset, chunk_size)
                if offset == 2:
                    second_chunk_prepared.set()
                elif offset == 4:
                    third_chunk_prepared.set()
                return prepared

        class _TrackingModel(_FakeEmbeddingModel):
            def embed(
                self,
                input_data: list[Any],
                *,
                tokenization_kwargs: dict[str, int],
                use_tqdm: bool,
            ) -> list[SimpleNamespace]:
                if not self.calls:
                    assert second_chunk_prepared.wait(timeout=2)
                    assert not third_chunk_prepared.is_set()
                return super().embed(
                    input_data,
                    tokenization_kwargs=tokenization_kwargs,
                    use_tqdm=use_tqdm,
                )

        stage = _TrackingStage(
            model_identifier=TEST_MODEL,
            pretokenize=False,
            model_inference_batch_size=2,
        )
        model = _TrackingModel()
        stage.model = model  # type: ignore[assignment]
        batch = DocumentBatch(
            dataset_name="test_dataset",
            data=pa.table({"text": ["0", "1", "2", "3", "4"]}),
        )

        stage.process(batch)

        assert model.calls == [[0, 1], [2, 3], [4]]
        assert third_chunk_prepared.is_set()

    @pytest.mark.parametrize(
        ("table", "metadata_fields", "message"),
        [
            (pa.table({"id": [1]}), None, "missing required text field"),
            (pa.table({"text": ["first"], "id": [1]}), ["id", "source"], "missing metadata fields"),
        ],
    )
    def test_process_rejects_missing_fields(
        self,
        table: pa.Table,
        metadata_fields: list[str] | None,
        message: str,
    ) -> None:
        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=False,
            metadata_fields=metadata_fields,
        )
        stage.model = SimpleNamespace()

        with pytest.raises(ValueError, match=message):
            stage.process(DocumentBatch(dataset_name="test_dataset", data=table))

    def test_process_reports_aggregate_chunk_metrics(self) -> None:
        class _TimedPreparationStage(VLLMEmbeddingModelStage):
            def _prepare_input_chunk(
                self,
                text_column: pa.ChunkedArray,
                offset: int,
                chunk_size: int,
            ) -> tuple[list[Any], float]:
                input_data, _ = super()._prepare_input_chunk(text_column, offset, chunk_size)
                return input_data, 0.25

            def _embed_chunk(
                self,
                input_data: list[Any],
                expected_size: int,
            ) -> tuple[np.ndarray, dict[str, float]]:
                embedding_matrix, metrics = super()._embed_chunk(input_data, expected_size)
                metrics["vllm_embedding_time"] = 0.5
                return embedding_matrix, metrics

        stage = _TimedPreparationStage(
            model_identifier=TEST_MODEL,
            pretokenize=False,
            metadata_fields=["text"],
            model_inference_batch_size=2,
        )
        stage.model = _FakeEmbeddingModel()  # type: ignore[assignment]
        batch = DocumentBatch(
            dataset_name="test_dataset",
            data=pa.table({"text": ["0", "1", "2", "3", "4"]}),
        )

        result = stage.process(batch)
        metrics = stage._consume_custom_metrics()

        assert result.num_items == 5
        assert metrics["tokenization_time"] == 0.75
        assert metrics["input_tokens"] == 10
        assert metrics["vllm_embedding_time"] == 1.5

    def test_process_rejects_wrong_vllm_output_count(self) -> None:
        class _ShortOutputModel(_FakeEmbeddingModel):
            def embed(
                self,
                input_data: list[Any],
                *,
                tokenization_kwargs: dict[str, int],
                use_tqdm: bool,
            ) -> list[SimpleNamespace]:
                return super().embed(
                    input_data,
                    tokenization_kwargs=tokenization_kwargs,
                    use_tqdm=use_tqdm,
                )[:-1]

        stage = VLLMEmbeddingModelStage(model_identifier=TEST_MODEL, pretokenize=False)
        stage.model = _ShortOutputModel()  # type: ignore[assignment]

        with pytest.raises(ValueError, match="returned 1 embeddings for a 2-row input chunk"):
            stage.process(DocumentBatch(dataset_name="test_dataset", data=pa.table({"text": ["0", "1"]})))

    def test_process_rejects_non_matrix_embeddings(self) -> None:
        class _ScalarEmbeddingModel(_FakeEmbeddingModel):
            def embed(
                self,
                input_data: list[Any],
                *,
                tokenization_kwargs: dict[str, int],
                use_tqdm: bool,
            ) -> list[SimpleNamespace]:
                outputs = super().embed(
                    input_data,
                    tokenization_kwargs=tokenization_kwargs,
                    use_tqdm=use_tqdm,
                )
                for output in outputs:
                    output.outputs.embedding = 1.0
                return outputs

        stage = VLLMEmbeddingModelStage(model_identifier=TEST_MODEL, pretokenize=False)
        stage.model = _ScalarEmbeddingModel()  # type: ignore[assignment]

        with pytest.raises(ValueError, match="two-dimensional embedding matrix"):
            stage.process(DocumentBatch(dataset_name="test_dataset", data=pa.table({"text": ["0"]})))

    def test_process_rejects_embedding_dimension_changes(self) -> None:
        class _ChangingDimensionModel(_FakeEmbeddingModel):
            def embed(
                self,
                input_data: list[Any],
                *,
                tokenization_kwargs: dict[str, int],
                use_tqdm: bool,
            ) -> list[SimpleNamespace]:
                outputs = super().embed(
                    input_data,
                    tokenization_kwargs=tokenization_kwargs,
                    use_tqdm=use_tqdm,
                )
                if len(self.calls) == 2:
                    outputs[0].outputs.embedding.append(2.0)
                return outputs

        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=False,
            model_inference_batch_size=1,
        )
        stage.model = _ChangingDimensionModel()  # type: ignore[assignment]

        with pytest.raises(ValueError, match="embedding dimension changed from 2 to 3"):
            stage.process(DocumentBatch(dataset_name="test_dataset", data=pa.table({"text": ["0", "1"]})))

    def test_process_replaces_selected_embedding_field_in_place(self) -> None:
        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=False,
            metadata_fields=["id", "embeddings", "text"],
        )
        stage.model = _FakeEmbeddingModel()  # type: ignore[assignment]
        batch = DocumentBatch(
            dataset_name="test_dataset",
            data=pa.table({"text": ["0"], "id": [7], "embeddings": [[99]]}),
        )

        result = stage.process(batch)

        assert stage.outputs() == (["data"], ["id", "embeddings", "text"])
        assert result.data.column_names == ["id", "embeddings", "text"]
        assert result.data["embeddings"].to_pylist() == [[0.0, 1.0]]
        assert result.data.schema.field("embeddings").type == pa.list_(pa.float32())

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
