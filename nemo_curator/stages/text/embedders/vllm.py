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

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
import torch
from huggingface_hub import snapshot_download

try:
    from vllm import LLM

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:  # dummy for type hints
        pass


from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.models.utils import format_name_with_suffix
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.vllm_utils import create_vllm_llm_with_retry

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import Future

    from transformers import AutoTokenizer

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_VLLM_INSTALL_HINT = "vLLM is required for VLLMEmbeddingModelStage. Install with: pip install nemo_curator[vllm]"
_MAX_LIST_ARRAY_VALUES = np.iinfo(np.int32).max


class VLLMEmbeddingModelStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        vllm_init_kwargs: dict[str, Any] | None = None,
        text_field: str = "text",
        pretokenize: bool = True,
        embedding_field: str = "embeddings",
        max_chars: int | None = None,
        cache_dir: str | None = None,
        hf_token: str | None = None,
        verbose: bool = False,
        *,
        metadata_fields: list[str] | None = None,
        model_inference_batch_size: int | None = 8192,
        # Keep float32 when feeding semantic dedup; cuDF cannot read nested Float16 Parquet as numeric list values.
        embedding_output_dtype: Literal["float16", "float32", "float64"] = "float32",
    ):
        self.model_identifier = model_identifier
        self.vllm_init_kwargs = vllm_init_kwargs or {}

        self.text_field = text_field
        self.pretokenize = pretokenize
        self.embedding_field = embedding_field
        self.embedding_output_dtype = embedding_output_dtype
        # Retained columns are opt-in so large source-text columns are not carried
        # alongside embeddings unless a caller explicitly requests them.
        self.metadata_fields = list(dict.fromkeys(metadata_fields or []))
        if model_inference_batch_size is not None and model_inference_batch_size < 0:
            msg = (
                f"model_inference_batch_size must be a non-negative integer or None, got {model_inference_batch_size}"
            )
            raise ValueError(msg)
        self.model_inference_batch_size = model_inference_batch_size
        self.max_chars = max_chars

        self.cache_dir = cache_dir
        self.hf_token = hf_token

        self.verbose = verbose
        # after setup
        self.model: None | LLM = None
        self.tokenizer: None | AutoTokenizer = None
        # stage setup
        self.resources = Resources(
            cpus=1,
            gpus=1,
        )
        self.name = format_name_with_suffix(model_identifier, suffix="_vllm")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = list(self.metadata_fields)
        if self.embedding_field not in output_fields:
            output_fields.append(self.embedding_field)
        return ["data"], output_fields

    def _initialize_vllm(self, local_files_only: bool) -> None:
        """Download (or locate) the model and initialize vLLM.

        We pass the resolved snapshot path to ``LLM(model=...)`` instead of the
        HuggingFace repo ID because vLLM does not pass the ``download_dir`` through
        to its config resolution code — passing a repo ID with a custom cache dir
        fails offline.
        """
        if not VLLM_AVAILABLE:
            raise ImportError(_VLLM_INSTALL_HINT)
        model_path = snapshot_download(
            self.model_identifier,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            local_files_only=local_files_only,
        )

        vllm_init_kwargs = self.vllm_init_kwargs.copy()
        if "enforce_eager" not in vllm_init_kwargs:
            vllm_init_kwargs["enforce_eager"] = False
        if "runner" not in vllm_init_kwargs:
            vllm_init_kwargs["runner"] = "pooling"
        if "model_impl" not in vllm_init_kwargs:
            # TODO: Once transformers is bumped to 5.0 then we should also support transformers
            vllm_init_kwargs["model_impl"] = "vllm"
        if self.cache_dir is not None and "download_dir" not in vllm_init_kwargs:
            vllm_init_kwargs["download_dir"] = self.cache_dir

        # Reduce verbosity when not in verbose mode
        if not self.verbose and "disable_log_stats" not in vllm_init_kwargs:
            vllm_init_kwargs["disable_log_stats"] = True

        self.model = create_vllm_llm_with_retry(model=model_path, **vllm_init_kwargs)

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if not self.verbose:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()

        # Download model to cache_dir (or default HF cache) and initialize vLLM.
        # local_files_only=False allows downloading when online; if the model is
        # already cached (e.g. in air-gapped environments), snapshot_download falls
        # back to the local cache automatically.
        self._initialize_vllm(local_files_only=False)

    def teardown(self) -> None:
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self.model is None:
            # Load from local cache only — model must already be downloaded (by setup_on_node or pre-cached)
            self._initialize_vllm(local_files_only=True)
        if self.pretokenize:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                local_files_only=True,
            )

    def _prepare_input_chunk(
        self, text_column: pa.ChunkedArray, offset: int, chunk_size: int
    ) -> tuple[list[Any], float]:
        input_data = text_column.slice(offset, chunk_size).to_pylist()
        if self.max_chars is not None:
            input_data = [text[: self.max_chars] if text is not None else None for text in input_data]
        if not self.pretokenize:
            return input_data, 0.0

        from vllm.inputs import TokensPrompt

        t0 = time.perf_counter()
        tokenized_data = self.tokenizer.batch_encode_plus(
            input_data,
            truncation=True,
            max_length=self.model.model_config.max_model_len,
        )
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in tokenized_data.input_ids]
        return prompts, time.perf_counter() - t0

    def _iter_prepared_chunks(self, text_column: pa.ChunkedArray, num_rows: int) -> Iterator[tuple[list[Any], float]]:
        """Prepare one chunk ahead while the caller embeds the current chunk.

        One worker is intentional: this is a single-producer prefetch pipeline,
        not parallel tokenization. While the caller embeds the current chunk on
        the GPU, that worker prepares only the immediately following chunk on the
        CPU. This overlaps CPU and GPU work without allowing an unbounded queue of
        prepared chunks. ``Future.result`` blocks, so the generator never polls.
        """
        inference_batch_size = self.model_inference_batch_size or num_rows
        chunk_specs = iter(
            (offset, min(inference_batch_size, num_rows - offset))
            for offset in range(0, num_rows, inference_batch_size)
        )
        pending_offset, pending_chunk_size = next(chunk_specs)

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm-tokenizer") as executor:
            pending_input: Future[tuple[list[Any], float]] = executor.submit(
                self._prepare_input_chunk,
                text_column,
                pending_offset,
                pending_chunk_size,
            )
            for next_offset, next_chunk_size in chunk_specs:
                input_data, tokenization_time = pending_input.result()
                next_input = executor.submit(
                    self._prepare_input_chunk,
                    text_column,
                    next_offset,
                    next_chunk_size,
                )

                yield input_data, tokenization_time
                pending_offset = next_offset
                pending_chunk_size = next_chunk_size
                pending_input = next_input

            input_data, tokenization_time = pending_input.result()
            yield input_data, tokenization_time

    def _embed_chunk(self, input_data: list[Any]) -> tuple[np.ndarray, dict[str, float]]:
        t0 = time.perf_counter()
        vllm_output = self.model.embed(
            input_data,
            tokenization_kwargs={"truncate_prompt_tokens": -1},
            use_tqdm=self.verbose,
        )
        elapsed = time.perf_counter() - t0
        chunk_embedding_matrix = np.asarray(
            [output.outputs.embedding for output in vllm_output],
            dtype=self.embedding_output_dtype,
        )
        return chunk_embedding_matrix, {
            "vllm_embedding_time": elapsed,
            "input_tokens": sum(len(output.prompt_token_ids) for output in vllm_output),
        }

    def _select_output_table(self, input_table: pa.Table) -> pa.Table:
        """Validate the input and select columns retained beside embeddings."""
        if self.text_field not in input_table.column_names:
            msg = f"Input batch is missing required text field {self.text_field!r}"
            raise ValueError(msg)

        missing_fields = [field for field in self.metadata_fields if field not in input_table.column_names]
        if missing_fields:
            msg = f"Input batch is missing metadata fields: {missing_fields}"
            raise ValueError(msg)
        return input_table.select(self.metadata_fields)

    def _collect_embeddings(
        self, text_column: pa.ChunkedArray, num_rows: int
    ) -> tuple[pa.ChunkedArray, dict[str, float]]:
        """Embed bounded chunks and assemble one ordered Arrow array."""
        embedding_chunks: list[pa.Array] = []
        tokenization_time = 0.0
        vllm_embedding_time = 0.0
        input_tokens = 0

        for input_data, chunk_tokenization_time in self._iter_prepared_chunks(text_column, num_rows):
            tokenization_time += chunk_tokenization_time
            chunk_embedding_matrix, chunk_metrics = self._embed_chunk(input_data)
            vllm_embedding_time += chunk_metrics["vllm_embedding_time"]
            input_tokens += chunk_metrics["input_tokens"]
            embedding_chunks.extend(self._to_arrow_embeddings(chunk_embedding_matrix).chunks)
            del chunk_embedding_matrix

        return pa.chunked_array(embedding_chunks), {
            "tokenization_time": tokenization_time,
            "vllm_embedding_time": vllm_embedding_time,
            "input_tokens": input_tokens,
        }

    @staticmethod
    def _to_arrow_embeddings(embedding_matrix: np.ndarray) -> pa.ChunkedArray:
        """Convert a dense matrix to bounded Arrow list-array chunks."""
        embedding_dim = embedding_matrix.shape[1]
        rows_per_chunk = _MAX_LIST_ARRAY_VALUES // embedding_dim
        value_type = pa.from_numpy_dtype(embedding_matrix.dtype)
        chunks = []
        for offset in range(0, embedding_matrix.shape[0], rows_per_chunk):
            matrix_chunk = embedding_matrix[offset : offset + rows_per_chunk]
            values = pa.array(matrix_chunk.reshape(-1), type=value_type, from_pandas=False)
            offsets = pa.array(
                np.arange(0, matrix_chunk.size + 1, embedding_dim, dtype=np.int32),
            )
            chunks.append(pa.ListArray.from_arrays(offsets, values))
        return pa.chunked_array(chunks, type=pa.list_(value_type))

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        input_table = batch.to_pyarrow()
        output_table = self._select_output_table(input_table)
        embedding_array, metrics = self._collect_embeddings(input_table[self.text_field], input_table.num_rows)
        if self.embedding_field in output_table.column_names:
            embedding_index = output_table.column_names.index(self.embedding_field)
            output_table = output_table.set_column(embedding_index, self.embedding_field, embedding_array)
        else:
            output_table = output_table.append_column(self.embedding_field, embedding_array)

        self._log_metrics(metrics)

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=output_table,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
