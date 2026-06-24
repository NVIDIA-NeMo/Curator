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

# ruff: noqa: PLR0913

"""Embedding generation benchmarking script.

Supports multiple embedding model backends (through the model_variation argument):
- sentence_transformer: EmbeddingCreatorStage with SentenceTransformer
- pytorch_model: EmbeddingCreatorStage with raw PyTorch model + custom pooling
- vllm_text: VLLMEmbeddingModelStage with text input
- vllm_text_pretokenized: VLLMEmbeddingModelStage with pretokenization
"""

import argparse
import asyncio
import base64
import inspect
import json
import os
import secrets
import time
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from utils import load_dataset_files, setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter
from nemo_curator.tasks import DocumentBatch

_MATRIX_NDIM = 2


class _EmbeddingData(Protocol):
    index: int
    embedding: object


class _EmbeddingResponse(Protocol):
    data: list[_EmbeddingData]


class EmbeddingModelVariation(Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    PYTORCH_MODEL = "pytorch_model"
    VLLM_TEXT = "vllm_text"
    VLLM_TEXT_PRETOKENIZED = "vllm_text_pretokenized"
    RAY_SERVE_ENDPOINT = "ray_serve_endpoint"
    DYNAMO_ENDPOINT = "dynamo_endpoint"


class EmbeddingParquetWriter(ParquetWriter):
    """Write Arrow embedding columns without round-tripping through pandas."""

    def write_data(self, task: DocumentBatch, file_path: str) -> None:
        if isinstance(task.data, pa.Table):
            table = task.data
            if self.fields is not None:
                table = table.select(self.fields)
            pq.write_table(table, file_path, **self.write_kwargs)
            return

        super().write_data(task, file_path)


class OpenAIEmbeddingClientStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """CPU stage that calls an OpenAI-compatible /v1/embeddings endpoint."""

    name = "openai_embedding_client"
    resources = Resources(cpus=1.0)

    def __init__(
        self,
        model_identifier: str,
        endpoint: str,
        max_concurrent_requests: int,
        timeout: float,
        max_retries: int = 3,
        retry_base_delay_s: float = 1.0,
        endpoint_truncate_prompt_tokens: int | None = -1,
        endpoint_input_format: str = "text",
        endpoint_request_batch_size: int = 8,
        endpoint_encoding_format: str = "float",
        endpoint_max_chars: int | None = None,
        cache_dir: str | None = None,
        text_field: str = "text",
        embedding_field: str = "embeddings",
        api_key: str = "unused",  # pragma: allowlist secret
    ):
        self.model_identifier = model_identifier
        self.endpoint = endpoint
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.endpoint_truncate_prompt_tokens = endpoint_truncate_prompt_tokens
        self.endpoint_input_format = endpoint_input_format
        self.endpoint_request_batch_size = endpoint_request_batch_size
        self.endpoint_encoding_format = endpoint_encoding_format
        self.endpoint_max_chars = endpoint_max_chars
        self.cache_dir = cache_dir
        self.text_field = text_field
        self.embedding_field = embedding_field
        self.api_key = api_key
        self.client = None
        self.http_client = None
        self.tokenizer = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.embedding_field]

    def setup(self, worker_metadata=None) -> None:  # noqa: ANN001, ARG002
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=self.endpoint, api_key=self.api_key, timeout=self.timeout)
        if self.endpoint_input_format == "token_ids":
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )

    def teardown(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            close_result = self.client.close()
            if inspect.isawaitable(close_result):
                asyncio.run(close_result)
        self.client = None
        self.http_client = None
        self.tokenizer = None

    def _request_inputs(self, texts: list[str]) -> list[str] | list[list[int]]:
        if self.endpoint_input_format == "text":
            return texts
        if self.endpoint_input_format != "token_ids":
            msg = f"Unsupported endpoint_input_format={self.endpoint_input_format!r}"
            raise ValueError(msg)
        if self.tokenizer is None:
            msg = "Tokenizer is not initialized for token_ids endpoint input"
            raise RuntimeError(msg)

        max_length = self.endpoint_truncate_prompt_tokens
        truncation = max_length is not None and max_length > 0
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=truncation,
            max_length=max_length if truncation else None,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [list(map(int, input_ids)) for input_ids in encoded["input_ids"]]

    def _request_batches(self, request_inputs: list[str] | list[list[int]]) -> list[list[Any]]:
        batch_size = max(1, self.endpoint_request_batch_size)
        return [request_inputs[start : start + batch_size] for start in range(0, len(request_inputs), batch_size)]

    def _response_embeddings(self, response: _EmbeddingResponse, request_input: list[Any]) -> list[Any]:
        response_data = sorted(response.data, key=lambda item: item.index)
        if len(response_data) != len(request_input):
            msg = f"Endpoint returned {len(response_data)} embeddings for {len(request_input)} inputs"
            raise RuntimeError(msg)
        response_indexes = [int(item.index) for item in response_data]
        expected_indexes = list(range(len(request_input)))
        if response_indexes != expected_indexes:
            msg = (
                f"Endpoint returned non-contiguous embedding indexes: {response_indexes}, expected {expected_indexes}"
            )
            raise RuntimeError(msg)
        return [item.embedding for item in response_data]

    async def _embed_batch(self, semaphore: asyncio.Semaphore, request_input: list[Any]) -> list[Any]:
        async with semaphore:
            last_exception = None
            for attempt in range(self.max_retries + 1):
                if attempt > 0 and last_exception is not None:
                    if not _is_retryable_endpoint_error(last_exception):
                        raise last_exception
                    delay = self.retry_base_delay_s * (2 ** (attempt - 1)) + secrets.randbelow(100) / 100.0
                    await asyncio.sleep(delay)

                try:
                    request_kwargs: dict[str, Any] = {
                        "model": self.model_identifier,
                        "input": request_input,
                        "encoding_format": self.endpoint_encoding_format,
                        "timeout": self.timeout,
                    }
                    if self.endpoint_truncate_prompt_tokens is not None and self.endpoint_truncate_prompt_tokens > 0:
                        request_kwargs["extra_body"] = {
                            "truncate_prompt_tokens": self.endpoint_truncate_prompt_tokens,
                        }
                    if self.endpoint_input_format == "token_ids":
                        request_kwargs.setdefault("extra_body", {})
                        request_kwargs["extra_body"]["add_special_tokens"] = False
                    response = await self.client.embeddings.create(**request_kwargs)  # type: ignore[union-attr]
                    return self._response_embeddings(response, request_input)
                except Exception as exc:
                    last_exception = exc
                    if attempt == self.max_retries:
                        raise
            msg = "embedding request retry loop exited without a result"
            raise RuntimeError(msg)

    async def _embed_batches(self, request_batches: list[list[Any]]) -> list[Any]:
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        batch_embeddings = await asyncio.gather(
            *(self._embed_batch(semaphore, request_input) for request_input in request_batches)
        )
        return [embedding for embeddings in batch_embeddings for embedding in embeddings]

    def _base64_embedding_to_array(self, embedding: str, row_index: int) -> np.ndarray:
        raw = base64.b64decode(embedding, validate=True)
        if len(raw) % np.dtype(np.float32).itemsize != 0:
            msg = f"row={row_index}, base64 payload has {len(raw)} bytes, not a multiple of float32 size"
            raise RuntimeError(msg)
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)

    def _embeddings_to_float32_array(self, embeddings: list[Any]) -> np.ndarray:
        rows: list[np.ndarray] = []
        expected_dim: int | None = None
        bad_shapes: list[str] = []

        for row_index, embedding in enumerate(embeddings):
            if isinstance(embedding, str):
                row = self._base64_embedding_to_array(embedding, row_index)
            else:
                row = np.asarray(embedding, dtype=np.float32)
            if row.ndim == _MATRIX_NDIM and row.shape[0] == 1:
                row = row[0]
            if row.ndim != 1:
                bad_shapes.append(f"row={row_index}, shape={tuple(row.shape)}")
                continue
            if expected_dim is None:
                expected_dim = int(row.shape[0])
            elif row.shape[0] != expected_dim:
                bad_shapes.append(f"row={row_index}, shape={tuple(row.shape)}, expected_dim={expected_dim}")
                continue
            rows.append(row)

        if bad_shapes:
            msg = "Endpoint returned non-flat or variable-size embeddings: " + "; ".join(bad_shapes[:8])
            raise RuntimeError(msg)

        if not rows:
            return np.empty((0, 0), dtype=np.float32)

        return np.stack(rows).astype(np.float32, copy=False)

    def _embeddings_to_arrow_table(self, embeddings: list[Any]) -> pa.Table:
        embedding_array = self._embeddings_to_float32_array(embeddings)

        flat_values = pa.array(embedding_array.reshape(-1), type=pa.float32())
        embedding_column = pa.FixedSizeListArray.from_arrays(flat_values, embedding_array.shape[1])
        return pa.Table.from_arrays([embedding_column], names=[self.embedding_field])

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if self.client is None:
            self.setup()

        df = batch.to_pandas()
        texts = [str(text) for text in df[self.text_field].tolist()]
        if self.endpoint_max_chars is not None and self.endpoint_max_chars > 0:
            texts = [text[: self.endpoint_max_chars] for text in texts]

        t0 = time.perf_counter()
        request_inputs = self._request_inputs(texts)
        request_batches = self._request_batches(request_inputs)
        tokenization_elapsed = time.perf_counter() - t0
        embeddings = asyncio.run(self._embed_batches(request_batches))
        elapsed = time.perf_counter() - t0

        self._log_metrics(
            {
                "endpoint_embedding_time": elapsed,
                "endpoint_tokenization_time": tokenization_elapsed,
                "embedding_request_count": len(request_batches),
                "endpoint_request_count": len(request_batches),
                "endpoint_document_count": len(texts),
                "endpoint_request_batch_size": self.endpoint_request_batch_size,
                "endpoint_encoding_base64": float(self.endpoint_encoding_format == "base64"),
                "endpoint_max_concurrent_requests": self.max_concurrent_requests,
                "endpoint_truncate_prompt_tokens": self.endpoint_truncate_prompt_tokens or 0,
                "endpoint_max_chars": self.endpoint_max_chars or 0,
                "endpoint_pretokenized": float(self.endpoint_input_format == "token_ids"),
            }
        )

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=self._embeddings_to_arrow_table(embeddings),
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def _is_retryable_endpoint_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate",
            "connection",
            "readerror",
            "brokenresourceerror",
            "apiconnectionerror",
            "timeout",
        )
    )


def _resolve_max_seq_length(model_identifier: str, cache_dir: str | None = None) -> int:
    """Resolve max_seq_length from the sentence-transformers config.

    vLLM reads max_seq_length from the Sentence Transformers config (e.g. 256
    for MiniLM), which is often lower than max_position_embeddings (512).
    SentenceTransformers also silently truncates to this value, but HF
    AutoModel does not — it uses the full max_position_embeddings.

    We use the sentence-transformers config as the single source of truth
    so all backends process the same number of tokens.
    """
    from huggingface_hub import snapshot_download
    from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config

    # Resolve to a local snapshot path so the vLLM helper can find configs
    # even when the model lives in a non-default cache directory.
    model_path = snapshot_download(model_identifier, cache_dir=cache_dir, local_files_only=True)

    st_config = get_sentence_transformer_tokenizer_config(model_path)
    if st_config is None:
        msg = f"No sentence-transformers config found for {model_identifier}"
        raise ValueError(msg)

    model_limit = st_config["max_seq_length"]
    logger.info(f"Resolved max_seq_length={model_limit} from sentence-transformers config")
    return model_limit


def _create_embedding_stages(
    model_identifier: str,
    model_variation: EmbeddingModelVariation,
    model_inference_batch_size: int,
    model_num_workers: int | None,
    max_seq_length: int,
    embedding_pooling: str,
    max_chars: int | None = None,
    cache_dir: str | None = None,
) -> list:
    """Create the embedding stage(s) for the given model variation."""
    if model_variation in {EmbeddingModelVariation.SENTENCE_TRANSFORMER, EmbeddingModelVariation.PYTORCH_MODEL}:
        from nemo_curator.stages.text.embedders import EmbeddingCreatorStage

        use_sentence_transformer = model_variation == EmbeddingModelVariation.SENTENCE_TRANSFORMER
        return [
            EmbeddingCreatorStage(
                model_identifier=model_identifier,
                use_sentence_transformer=use_sentence_transformer,
                text_field="text",
                embedding_field="embeddings",
                model_inference_batch_size=model_inference_batch_size,
                sort_by_length=True,
                max_seq_length=max_seq_length,
                embedding_pooling=embedding_pooling,
                cache_dir=cache_dir,
            ),
        ]

    if model_variation in {EmbeddingModelVariation.VLLM_TEXT, EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED}:
        from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage

        # vLLM strictly enforces max_model_len from the model config, unlike
        # sentence-transformers which silently truncates.  Pass max_seq_length
        # through so vLLM knows the intended limit and won't error on inputs
        # that exceed the model's default max_position_embeddings.
        vllm_init_kwargs: dict[str, Any] = {"max_model_len": max_seq_length}

        stage = VLLMEmbeddingModelStage(
            model_identifier=model_identifier,
            text_field="text",
            embedding_field="embeddings",
            pretokenize=model_variation == EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED,
            vllm_init_kwargs=vllm_init_kwargs,
            max_chars=max_chars,
            cache_dir=cache_dir,
        )
        stage_overrides: dict[str, Any] = {}
        if cache_dir is not None:
            stage_overrides["runtime_env"] = _endpoint_runtime_env(cache_dir)
        if model_num_workers is not None:
            stage_overrides["num_workers"] = model_num_workers
        if stage_overrides:
            stage = stage.with_(**stage_overrides)
        return [stage]

    msg = f"Unsupported model variation: {model_variation}"
    raise ValueError(msg)


def _endpoint_engine_kwargs(
    max_seq_length: int,
    cache_dir: str | None,
    dtype: str,
    pooler_config: dict[str, Any],
) -> dict[str, Any]:
    engine_kwargs: dict[str, Any] = {
        "runner": "pooling",
        "convert": "embed",
        "tensor_parallel_size": 1,
        "pooler_config": pooler_config,
        "max_model_len": max_seq_length,
        "dtype": dtype,
        "enable_prefix_caching": False,
        "trust_remote_code": False,
    }
    if cache_dir is not None:
        engine_kwargs["download_dir"] = cache_dir
    return engine_kwargs


def _resolve_endpoint_model_path(
    model_identifier: str, cache_dir: str | None, explicit_model_path: str | None
) -> str | None:
    if explicit_model_path:
        return explicit_model_path
    if cache_dir is None:
        return None

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model_identifier, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        return snapshot_download(model_identifier, cache_dir=cache_dir, local_files_only=False)


def _endpoint_runtime_env(cache_dir: str | None) -> dict[str, Any]:
    if cache_dir is None:
        return {}

    cache_root = Path(cache_dir)
    home = cache_root / "curator_endpoint_home"
    xdg_cache = home / ".cache"
    hub_cache = cache_root / "hub"
    return {
        "env_vars": {
            "HOME": str(home),
            "XDG_CACHE_HOME": str(xdg_cache),
            "TORCH_HOME": str(xdg_cache / "torch"),
            "HF_HOME": str(cache_root),
            "HF_HUB_CACHE": str(hub_cache),
            "TRANSFORMERS_CACHE": str(hub_cache),
            "UV_CACHE_DIR": str(cache_root / "uv_cache"),
        }
    }


def _merge_runtime_envs(*runtime_envs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    env_vars: dict[str, str] = {}
    for runtime_env in runtime_envs:
        for key, value in runtime_env.items():
            if key == "env_vars":
                env_vars.update({env_key: str(env_value) for env_key, env_value in value.items()})
            else:
                merged[key] = value
    if env_vars:
        merged["env_vars"] = env_vars
    return merged


def _dynamo_embedding_runtime_env() -> dict[str, Any]:
    patch_dir = Path(__file__).resolve().parent / "dynamo_embedding_patch"
    return {
        "env_vars": {
            "NEMO_CURATOR_PATCH_DYNAMO_EMBED_POOLING_TASK": "1",
            "PYTHONPATH": os.pathsep.join([str(patch_dir), os.environ.get("PYTHONPATH", "")]),
        }
    }


def _endpoint_env_vars(cache_dir: str | None) -> dict[str, str]:
    return {key: str(value) for key, value in _endpoint_runtime_env(cache_dir).get("env_vars", {}).items()}


def _dynamo_embedding_subprocess_env(cache_dir: str | None) -> dict[str, str]:
    patch_dir = Path(__file__).resolve().parent / "dynamo_embedding_patch"
    env_vars = _endpoint_env_vars(cache_dir)
    env_vars.update(
        {
            "NEMO_CURATOR_PATCH_DYNAMO_EMBED_POOLING_TASK": "1",
            "PYTHONPATH": os.pathsep.join([str(patch_dir), os.environ.get("PYTHONPATH", "")]),
        }
    )
    return env_vars


def _start_ray_serve_embedding_server(
    model_identifier: str,
    endpoint_model_path: str | None,
    max_seq_length: int,
    server_replicas: int,
    cache_dir: str | None,
    endpoint_dtype: str,
    endpoint_pooler_config: dict[str, Any],
) -> object:
    from nemo_curator.core.serve import InferenceServer, RayServeModelConfig

    engine_kwargs = _endpoint_engine_kwargs(max_seq_length, cache_dir, endpoint_dtype, endpoint_pooler_config)
    model_config = RayServeModelConfig(
        model_identifier=endpoint_model_path or model_identifier,
        model_name=model_identifier if endpoint_model_path else None,
        deployment_config={
            "autoscaling_config": {
                "min_replicas": server_replicas,
                "max_replicas": server_replicas,
            },
            "max_ongoing_requests": 512,
            "max_queued_requests": 32768,
        },
        engine_kwargs=engine_kwargs,
        runtime_env=_endpoint_runtime_env(cache_dir),
    )
    server = InferenceServer(models=[model_config])
    server.start()
    return server


def _start_dynamo_embedding_server(
    model_identifier: str,
    endpoint_model_path: str | None,
    max_seq_length: int,
    server_replicas: int,
    cache_dir: str | None,
    endpoint_dtype: str,
    endpoint_pooler_config: dict[str, Any],
) -> object:
    from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer
    from nemo_curator.core.serve.dynamo.config import DynamoRouterConfig

    engine_kwargs = _endpoint_engine_kwargs(max_seq_length, cache_dir, endpoint_dtype, endpoint_pooler_config)
    model_config = DynamoVLLMModelConfig(
        model_identifier=endpoint_model_path or model_identifier,
        model_name=model_identifier if endpoint_model_path else None,
        engine_kwargs=engine_kwargs,
        dynamo_kwargs={"embedding_worker": True},
        num_replicas=server_replicas,
        runtime_env=_merge_runtime_envs(_endpoint_runtime_env(cache_dir), _dynamo_embedding_runtime_env()),
    )
    server = InferenceServer(
        models=[model_config],
        backend=DynamoServerConfig(
            router=DynamoRouterConfig(mode="random"),
            subprocess_env=_dynamo_embedding_subprocess_env(cache_dir),
        ),
    )
    server.start()
    return server


def _create_endpoint_embedding_stage(
    model_identifier: str,
    endpoint: str,
    client_num_workers: int,
    endpoint_max_concurrent_requests: int,
    endpoint_request_timeout_s: float,
    endpoint_max_retries: int,
    endpoint_retry_base_delay_s: float,
    endpoint_truncate_prompt_tokens: int | None,
    endpoint_input_format: str,
    endpoint_request_batch_size: int,
    endpoint_encoding_format: str,
    endpoint_max_chars: int | None,
    endpoint_client_mode: str,
    cache_dir: str | None,
) -> OpenAIEmbeddingClientStage:
    stage = OpenAIEmbeddingClientStage(
        model_identifier=model_identifier,
        endpoint=endpoint,
        max_concurrent_requests=endpoint_max_concurrent_requests,
        timeout=endpoint_request_timeout_s,
        max_retries=endpoint_max_retries,
        retry_base_delay_s=endpoint_retry_base_delay_s,
        endpoint_truncate_prompt_tokens=endpoint_truncate_prompt_tokens,
        endpoint_input_format=endpoint_input_format,
        endpoint_request_batch_size=endpoint_request_batch_size,
        endpoint_encoding_format=endpoint_encoding_format,
        endpoint_max_chars=endpoint_max_chars,
        cache_dir=cache_dir,
    )

    if endpoint_client_mode == "tasks":
        task_pool_size = client_num_workers if client_num_workers > 0 else None
        return stage.with_(
            num_workers=task_pool_size,
            ray_stage_spec={"is_actor_stage": False},
        )

    if endpoint_client_mode != "actor_pool":
        msg = f"Unsupported endpoint_client_mode={endpoint_client_mode!r}"
        raise ValueError(msg)
    if client_num_workers < 1:
        msg = "client_num_workers must be >= 1 when endpoint_client_mode='actor_pool'"
        raise ValueError(msg)

    return stage.with_(
        num_workers=client_num_workers,
        ray_stage_spec={"is_actor_stage": True},
    )


def _effective_endpoint_truncate_prompt_tokens(
    requested_truncate_prompt_tokens: int | None,
    max_seq_length: int,
) -> int | None:
    if requested_truncate_prompt_tokens is None:
        return None
    if requested_truncate_prompt_tokens <= 0:
        return max_seq_length
    return requested_truncate_prompt_tokens


def _read_ray_num_cpus(benchmark_results_path: str | Path | None) -> int | None:
    if benchmark_results_path is None:
        return None
    params_path = Path(benchmark_results_path) / "params.json"
    try:
        data = json.loads(params_path.read_text())
        return int(data["ray_num_cpus"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _estimate_endpoint_client_workers(
    *,
    client_num_workers: int,
    endpoint_client_mode: str,
    input_file_count: int,
    benchmark_results_path: str | Path | None,
) -> int:
    if input_file_count <= 0:
        return 0
    if endpoint_client_mode == "actor_pool":
        return min(input_file_count, client_num_workers)
    if client_num_workers > 0:
        return min(input_file_count, client_num_workers)

    ray_num_cpus = _read_ray_num_cpus(benchmark_results_path)
    if ray_num_cpus is None or ray_num_cpus <= 0:
        return input_file_count
    return min(input_file_count, ray_num_cpus)


def run_embedding_generation_benchmark(  # noqa: PLR0915
    input_path: str,
    output_path: str,
    executor: str,
    dataset_size_gb: float | None,
    load_dataset_ratio: float | None,
    model_identifier: str,
    model_inference_batch_size: int,
    model_num_workers: int | None,
    model_variation: str,
    embedding_pooling: str,
    input_format: str = "parquet",
    max_chars: int | None = None,
    cache_dir: str | None = None,
    server_replicas: int = 4,
    client_num_workers: int = 32,
    endpoint_max_concurrent_requests: int = 16,
    endpoint_request_timeout_s: float = 900.0,
    endpoint_max_retries: int = 3,
    endpoint_retry_base_delay_s: float = 1.0,
    endpoint_truncate_prompt_tokens: int | None = -1,
    endpoint_input_format: str = "text",
    endpoint_request_batch_size: int = 8,
    endpoint_encoding_format: str = "float",
    endpoint_max_chars: int | None = None,
    endpoint_client_mode: str = "actor_pool",
    endpoint_dtype: str = "auto",
    endpoint_pooler_config: str = '{"task":"embed"}',
    endpoint_model_path: str | None = None,
    allow_raw_inprocess_vllm: bool = False,
    benchmark_results_path: str | None = None,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> dict[str, Any]:
    """Run the embedding generation benchmark and collect comprehensive metrics."""
    variation = EmbeddingModelVariation(model_variation)
    pretokenized = variation == EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED or (
        variation in {EmbeddingModelVariation.RAY_SERVE_ENDPOINT, EmbeddingModelVariation.DYNAMO_ENDPOINT}
        and endpoint_input_format == "token_ids"
    )
    if variation == EmbeddingModelVariation.VLLM_TEXT and not allow_raw_inprocess_vllm:
        msg = (
            "Raw in-process vLLM text benchmarking is disabled by default because tokenizer work "
            "can dominate throughput. Use --model-variation=vllm_text_pretokenized for SOTA "
            "comparisons, or pass --allow-raw-inprocess-vllm for an intentional raw-text experiment."
        )
        raise ValueError(msg)
    max_seq_length = _resolve_max_seq_length(model_identifier, cache_dir=cache_dir)
    input_path = Path(input_path)
    output_path = Path(output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting embedding generation benchmark")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Dataset size: {dataset_size_gb} GB")
    logger.info(f"Load dataset ratio: {load_dataset_ratio}")
    logger.info(f"Model: {model_identifier}")
    logger.info(f"Model variation: {variation.name}")
    logger.info(f"Pretokenized: {pretokenized}")
    logger.info(f"Batch size: {model_inference_batch_size}")
    logger.info(f"Model workers: {model_num_workers}")
    logger.info(f"Embedding pooling: {embedding_pooling}")
    logger.info(f"Input format: {input_format}")
    logger.info(f"Max chars: {max_chars}")
    logger.info(f"Executor: {executor}")

    run_start_time = time.perf_counter()

    keep_ext = "jsonl" if input_format == "jsonl" else "parquet"
    input_files = load_dataset_files(
        input_path,
        dataset_size_gb=dataset_size_gb,
        dataset_ratio=load_dataset_ratio,
        keep_extensions=keep_ext,
    )
    logger.info(f"Selected {len(input_files)} input files")
    executor_obj = setup_executor(executor)

    inference_server = None
    serve_startup_s = 0.0
    resolved_endpoint_model_path = None
    endpoint_effective_client_workers = client_num_workers
    endpoint_total_max_concurrent_requests = client_num_workers * endpoint_max_concurrent_requests
    effective_endpoint_max_chars = endpoint_max_chars if endpoint_max_chars is not None else max_chars
    effective_endpoint_truncate_prompt_tokens = _effective_endpoint_truncate_prompt_tokens(
        endpoint_truncate_prompt_tokens,
        max_seq_length,
    )
    if variation in {EmbeddingModelVariation.RAY_SERVE_ENDPOINT, EmbeddingModelVariation.DYNAMO_ENDPOINT}:
        pooler_config = json_loads_arg(endpoint_pooler_config, "--endpoint-pooler-config")
        resolved_endpoint_model_path = _resolve_endpoint_model_path(model_identifier, cache_dir, endpoint_model_path)
        endpoint_effective_client_workers = _estimate_endpoint_client_workers(
            client_num_workers=client_num_workers,
            endpoint_client_mode=endpoint_client_mode,
            input_file_count=len(input_files),
            benchmark_results_path=benchmark_results_path,
        )
        endpoint_total_max_concurrent_requests = endpoint_effective_client_workers * endpoint_max_concurrent_requests
        logger.info(f"Endpoint server replicas: {server_replicas}")
        logger.info(f"Endpoint client workers: {client_num_workers}")
        logger.info(f"Endpoint effective client workers estimate: {endpoint_effective_client_workers}")
        logger.info(f"Endpoint client max concurrent requests per worker: {endpoint_max_concurrent_requests}")
        logger.info(f"Endpoint aggregate max concurrent requests: {endpoint_total_max_concurrent_requests}")
        logger.info(f"Endpoint requested truncate_prompt_tokens: {endpoint_truncate_prompt_tokens}")
        logger.info(f"Endpoint effective truncate_prompt_tokens: {effective_endpoint_truncate_prompt_tokens}")
        logger.info(f"Endpoint input format: {endpoint_input_format}")
        logger.info(f"Endpoint request batch size: {endpoint_request_batch_size}")
        logger.info(f"Endpoint encoding format: {endpoint_encoding_format}")
        logger.info(f"Endpoint max chars: {effective_endpoint_max_chars}")
        logger.info(f"Endpoint client mode: {endpoint_client_mode}")
        logger.info(f"Endpoint model path: {resolved_endpoint_model_path}")

        serve_start = time.perf_counter()
        starter = (
            _start_ray_serve_embedding_server
            if variation == EmbeddingModelVariation.RAY_SERVE_ENDPOINT
            else _start_dynamo_embedding_server
        )
        inference_server = starter(
            model_identifier,
            resolved_endpoint_model_path,
            max_seq_length,
            server_replicas,
            cache_dir,
            endpoint_dtype,
            pooler_config,
        )
        serve_startup_s = time.perf_counter() - serve_start
        logger.info(f"Embedding endpoint ready at {inference_server.endpoint} (startup: {serve_startup_s:.1f}s)")
        embedding_stages = [
            _create_endpoint_embedding_stage(
                model_identifier=model_identifier,
                endpoint=inference_server.endpoint,
                client_num_workers=client_num_workers,
                endpoint_max_concurrent_requests=endpoint_max_concurrent_requests,
                endpoint_request_timeout_s=endpoint_request_timeout_s,
                endpoint_max_retries=endpoint_max_retries,
                endpoint_retry_base_delay_s=endpoint_retry_base_delay_s,
                endpoint_truncate_prompt_tokens=effective_endpoint_truncate_prompt_tokens,
                endpoint_input_format=endpoint_input_format,
                endpoint_request_batch_size=endpoint_request_batch_size,
                endpoint_encoding_format=endpoint_encoding_format,
                endpoint_max_chars=effective_endpoint_max_chars,
                endpoint_client_mode=endpoint_client_mode,
                cache_dir=cache_dir,
            )
        ]
    else:
        embedding_stages = _create_embedding_stages(
            model_identifier=model_identifier,
            model_variation=variation,
            model_inference_batch_size=model_inference_batch_size,
            model_num_workers=model_num_workers,
            max_seq_length=max_seq_length,
            embedding_pooling=embedding_pooling,
            max_chars=max_chars,
            cache_dir=cache_dir,
        )

    if input_format == "jsonl":
        reader = JsonlReader(file_paths=input_files, files_per_partition=1, fields=["text"], _generate_ids=False)
        writer = JsonlWriter(path=str(output_path), fields=["embeddings"])
    else:
        reader = ParquetReader(file_paths=input_files, files_per_partition=1, fields=["text"], _generate_ids=False)
        writer = EmbeddingParquetWriter(path=str(output_path), fields=["embeddings"])

    pipeline = Pipeline(
        name="embedding_generation_pipeline",
        stages=[reader, *embedding_stages, writer],
    )
    try:
        output_tasks = pipeline.run(executor_obj)
    finally:
        if inference_server is not None:
            inference_server.stop()

    run_time_taken = time.perf_counter() - run_start_time

    num_documents_processed = sum(task._stage_perf[-1].num_items_processed for task in output_tasks)
    throughput_docs_per_sec = num_documents_processed / run_time_taken if run_time_taken > 0 else 0

    logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
    logger.success(f"Processed {num_documents_processed} documents")

    return {
        "params": {
            "max_seq_length": max_seq_length,
            "serve_startup_s": serve_startup_s,
            "endpoint_client_num_workers": client_num_workers,
            "endpoint_effective_client_workers": endpoint_effective_client_workers,
            "endpoint_max_concurrent_requests_per_worker": endpoint_max_concurrent_requests,
            "endpoint_total_max_concurrent_requests": endpoint_total_max_concurrent_requests,
            "endpoint_requested_truncate_prompt_tokens": endpoint_truncate_prompt_tokens,
            "endpoint_truncate_prompt_tokens": effective_endpoint_truncate_prompt_tokens,
            "endpoint_input_format": endpoint_input_format,
            "endpoint_request_batch_size": endpoint_request_batch_size,
            "endpoint_encoding_format": endpoint_encoding_format,
            "max_chars": max_chars,
            "endpoint_max_chars": effective_endpoint_max_chars,
            "endpoint_client_mode": endpoint_client_mode,
            "endpoint_model_path": resolved_endpoint_model_path,
            "endpoint_pretokenized": endpoint_input_format == "token_ids",
            "num_input_files": len(input_files),
            "load_dataset_ratio": load_dataset_ratio,
            "pretokenized": pretokenized,
            "model_num_workers": model_num_workers,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "num_documents_processed": num_documents_processed,
            "throughput_docs_per_sec": throughput_docs_per_sec,
            "serve_startup_s": serve_startup_s,
            "endpoint_client_num_workers": client_num_workers,
            "endpoint_effective_client_workers": endpoint_effective_client_workers,
            "endpoint_max_concurrent_requests_per_worker": endpoint_max_concurrent_requests,
            "endpoint_total_max_concurrent_requests": endpoint_total_max_concurrent_requests,
            "endpoint_requested_truncate_prompt_tokens": endpoint_truncate_prompt_tokens,
            "endpoint_truncate_prompt_tokens": effective_endpoint_truncate_prompt_tokens,
            "endpoint_input_format": endpoint_input_format,
            "endpoint_request_batch_size": endpoint_request_batch_size,
            "endpoint_encoding_format": endpoint_encoding_format,
            "max_chars": max_chars,
            "endpoint_max_chars": effective_endpoint_max_chars,
            "endpoint_client_mode": endpoint_client_mode,
            "endpoint_model_path": resolved_endpoint_model_path,
            "endpoint_pretokenized": endpoint_input_format == "token_ids",
            "num_input_files": len(input_files),
            "load_dataset_ratio": load_dataset_ratio,
            "pretokenized": pretokenized,
            "model_num_workers": model_num_workers,
        },
        "tasks": output_tasks,
    }


def json_loads_arg(raw: str, arg_name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"{arg_name} must be valid JSON: {raw}"
        raise ValueError(msg) from exc
    if not isinstance(value, dict):
        msg = f"{arg_name} must decode to a JSON object: {raw}"
        raise TypeError(msg)
    return value


def optional_int_arg(raw: str) -> int | None:
    if raw.lower() in {"none", "null"}:
        return None
    return int(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding generation benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--output-path", default="./embedding_generation_output", help="Output directory for results")
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"], help="Executor to use")
    parser.add_argument("--dataset-size-gb", type=float, default=None, help="Size of dataset to process in GB")
    parser.add_argument(
        "--load-dataset-ratio",
        type=float,
        default=None,
        help="Fraction of dataset file bytes to process; mutually exclusive with --dataset-size-gb",
    )
    parser.add_argument(
        "--model-identifier",
        required=True,
        help="Model identifier (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument("--model-inference-batch-size", type=int, default=1024, help="Batch size for model inference")
    parser.add_argument(
        "--model-num-workers",
        type=optional_int_arg,
        default=None,
        help="Fixed worker count for in-process model stages; use 'none' for executor default scaling",
    )
    parser.add_argument(
        "--model-variation",
        default="vllm_text_pretokenized",
        choices=[v.value for v in EmbeddingModelVariation],
        help="Embedding model backend (default: vllm_text_pretokenized)",
    )
    parser.add_argument(
        "--allow-raw-inprocess-vllm",
        action="store_true",
        help=(
            "Allow --model-variation=vllm_text. Keep this off for SOTA comparisons; raw text "
            "in-process vLLM can benchmark tokenizer overhead instead of embedding throughput."
        ),
    )
    parser.add_argument(
        "--embedding-pooling",
        default="mean_pooling",
        choices=["mean_pooling", "last_token"],
        help="Pooling strategy for pytorch_model variation (ignored by sentence_transformer)",
    )
    parser.add_argument(
        "--input-format",
        default="parquet",
        choices=["parquet", "jsonl"],
        help="Input file format (default: parquet)",
    )
    parser.add_argument(
        "--max-chars",
        type=optional_int_arg,
        default=None,
        help="Maximum characters per input text before embedding; use 'none' to disable",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory for model weights (uses default HF cache if not set)",
    )
    parser.add_argument("--server-replicas", type=int, default=4, help="Number of endpoint server replicas")
    parser.add_argument(
        "--client-num-workers",
        type=int,
        default=32,
        help="Fixed Ray Data actor pool size for endpoint client workers",
    )
    parser.add_argument(
        "--endpoint-max-concurrent-requests",
        type=int,
        default=16,
        help=(
            "Maximum in-flight embedding requests per endpoint client worker. "
            "Total in-flight requests are approximately client-num-workers times this value."
        ),
    )
    parser.add_argument(
        "--endpoint-request-timeout-s",
        type=float,
        default=900.0,
        help="Per-request OpenAI embeddings timeout for endpoint client workers",
    )
    parser.add_argument(
        "--endpoint-max-retries",
        type=int,
        default=3,
        help="Retry attempts for endpoint embedding requests",
    )
    parser.add_argument(
        "--endpoint-retry-base-delay-s",
        type=float,
        default=1.0,
        help="Base delay for endpoint request exponential backoff",
    )
    parser.add_argument(
        "--endpoint-truncate-prompt-tokens",
        type=optional_int_arg,
        default=-1,
        help=(
            "vLLM truncate_prompt_tokens value passed through OpenAI extra_body for endpoint embeddings; "
            "non-positive values use the model context length, and 'none' omits the extra_body field"
        ),
    )
    parser.add_argument(
        "--endpoint-input-format",
        default="text",
        choices=["text", "token_ids"],
        help="OpenAI embeddings input payload for endpoint runs",
    )
    parser.add_argument(
        "--endpoint-request-batch-size",
        type=int,
        default=8,
        help="Number of documents sent in each OpenAI embeddings request for endpoint runs",
    )
    parser.add_argument(
        "--endpoint-encoding-format",
        default="float",
        choices=["float", "base64"],
        help="OpenAI embeddings response encoding format for endpoint runs",
    )
    parser.add_argument(
        "--endpoint-max-chars",
        type=optional_int_arg,
        default=None,
        help="Maximum characters per input text before sending endpoint requests; use 'none' to disable",
    )
    parser.add_argument(
        "--endpoint-client-mode",
        default="actor_pool",
        choices=["actor_pool", "tasks"],
        help=(
            "How endpoint client work is scheduled. actor_pool uses client-num-workers actors; "
            "tasks uses Ray task scheduling and treats client-num-workers=0 as executor default."
        ),
    )
    parser.add_argument(
        "--endpoint-dtype",
        default="auto",
        help="dtype passed to vLLM-backed endpoint servers",
    )
    parser.add_argument(
        "--endpoint-pooler-config",
        default='{"task":"embed"}',
        help="JSON pooler_config passed to vLLM-backed endpoint servers",
    )
    parser.add_argument(
        "--endpoint-model-path",
        default=None,
        help=(
            "Optional local model path for endpoint servers. Clients still use --model-identifier as the "
            "served model name."
        ),
    )

    args = parser.parse_args()

    logger.info("=== Embedding Generation Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    success_code = 1
    result_dict: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    result_dict["params"]["pretokenized"] = (
        args.model_variation == EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED.value
        or (
            args.model_variation
            in {EmbeddingModelVariation.RAY_SERVE_ENDPOINT.value, EmbeddingModelVariation.DYNAMO_ENDPOINT.value}
            and args.endpoint_input_format == "token_ids"
        )
    )
    result_dict["params"]["endpoint_pretokenized"] = args.endpoint_input_format == "token_ids"
    try:
        run_result = run_embedding_generation_benchmark(**vars(args))
        result_dict["params"].update(run_result.pop("params", {}))
        result_dict.update(run_result)
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
