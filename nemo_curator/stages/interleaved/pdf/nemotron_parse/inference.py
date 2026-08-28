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

"""In-process and HTTP inference stages for Nemotron-Parse."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import time
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import torch
from loguru import logger
from PIL import Image

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch

DEFAULT_MODEL_PATH = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
PROMPT_BASE = "</s><s><predict_bbox><predict_classes><output_markdown>"
DEFAULT_MAX_TOKENS = 8192

_NEMOTRON_PARSE_SAMPLING_PARAMS: dict[str, Any] = {
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 1,
    "repetition_penalty": 1.1,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "skip_special_tokens": False,
    "seed": None,
}
_OPENAI_GENERATION_CONFIG_FIELDS = {"max_tokens", "seed", "temperature", "top_p"}


def _nemotron_parse_sampling_params(max_tokens: int) -> dict[str, Any]:
    return {**_NEMOTRON_PARSE_SAMPLING_PARAMS, "max_tokens": max_tokens}


def _nemotron_parse_server_generation_config(max_tokens: int) -> GenerationConfig:
    sampling_params = _nemotron_parse_sampling_params(max_tokens)
    config_params = {key: value for key, value in sampling_params.items() if key in _OPENAI_GENERATION_CONFIG_FIELDS}
    extra_body = {key: value for key, value in sampling_params.items() if key not in config_params}
    return GenerationConfig(**config_params, extra_kwargs={"extra_body": extra_body})


def build_task_prompt(*, text_in_pic: bool = False) -> str:
    """Build the Nemotron-Parse task prompt with the appropriate text-in-pic token."""
    suffix = "<predict_text_in_pic>" if text_in_pic else "<predict_no_text_in_pic>"
    return f"{PROMPT_BASE}{suffix}"


@dataclass
class NemotronParseInferenceStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """GPU stage: run Nemotron-Parse inference on pre-rendered page images.

    Reads PNG page images from ``binary_content``, runs model inference, and
    writes raw Nemotron-Parse output into ``text_content``.

    Supports two inference backends:

    - ``"vllm"`` (recommended): vLLM offline mode with continuous batching.
      Batching is handled internally by vLLM via ``max_num_seqs``.
    - ``"hf"``: HuggingFace Transformers with manual micro-batching via
      ``inference_batch_size``.

    Parameters
    ----------
    model_path
        HuggingFace model ID or local path (e.g. ``nvidia/NVIDIA-Nemotron-Parse-v1.2``).
    text_in_pic
        Whether to predict text inside pictures. When ``True``, uses the
        ``<predict_text_in_pic>`` prompt token; when ``False`` (default), uses
        ``<predict_no_text_in_pic>``. Only applies to Nemotron-Parse v1.2+.
    task_prompt
        Override the full prompt string. When set, ``text_in_pic`` is ignored.
    backend
        Inference backend: ``"vllm"`` or ``"hf"``.
    inference_batch_size
        Pages per GPU forward pass (HF backend only).
    max_num_seqs
        Maximum concurrent sequences (vLLM backend only).
    max_tokens
        Maximum number of generated tokens (vLLM backend only).
    engine_kwargs
        Extra keyword arguments forwarded to the vLLM engine (e.g.
        ``gpu_memory_utilization``, ``max_num_batched_tokens``). vLLM backend only.
    """

    model_path: str = DEFAULT_MODEL_PATH
    text_in_pic: bool = False
    task_prompt: str | None = None
    backend: str = "vllm"
    inference_batch_size: int = 4
    max_num_seqs: int = 64
    max_tokens: int = DEFAULT_MAX_TOKENS
    enforce_eager: bool = False
    engine_kwargs: dict[str, Any] | None = None
    name: str = "nemotron_parse_inference"
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0, gpus=1.0))

    def __post_init__(self) -> None:
        if self.task_prompt is None:
            self.task_prompt = build_task_prompt(text_in_pic=self.text_in_pic)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    # -- setup / teardown --

    def setup_on_node(self, node_info: dict | None = None, worker_metadata: dict | None = None) -> None:  # noqa: ARG002
        """Initialize model once per node (serially) to avoid torch.compile race conditions."""
        self._initialize_model()

    def setup(self, worker_metadata: dict | None = None) -> None:  # noqa: ARG002
        if not (hasattr(self, "_llm") or hasattr(self, "_model")):
            self._initialize_model()

    def _initialize_model(self) -> None:
        if self.backend == "vllm":
            self._setup_vllm()
        else:
            self._setup_hf()

    def _setup_hf(self) -> None:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig

        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        logger.info(f"[HF] Loading {self.model_path} on {device}")
        self._device = device
        self._model = (
            AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._gen_config = GenerationConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self._proc_size: tuple[int, int] = tuple(self._processor.image_processor.final_size)
        logger.info(f"[HF] Model loaded, proc_size={self._proc_size}")

    def _setup_vllm(self) -> None:
        from vllm import SamplingParams

        from nemo_curator.utils.vllm_utils import create_vllm_llm, resolve_local_model_path

        resolved_path = resolve_local_model_path(self.model_path)
        engine_kwargs = {
            "max_num_seqs": self.max_num_seqs,
            "enforce_eager": self.enforce_eager,
            **(self.engine_kwargs or {}),
        }
        self._llm = create_vllm_llm(resolved_path, **engine_kwargs)
        self._sampling_params = SamplingParams(**_nemotron_parse_sampling_params(self.max_tokens))
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        self._proc_size = tuple(processor.image_processor.final_size)
        del processor

    def teardown(self) -> None:
        if self.backend == "vllm":
            for attr in ("_llm", "_sampling_params"):
                with contextlib.suppress(AttributeError):
                    delattr(self, attr)
        else:
            for attr in ("_model", "_tokenizer", "_processor", "_gen_config"):
                with contextlib.suppress(AttributeError):
                    delattr(self, attr)
        torch.cuda.empty_cache()

    # -- inference --

    @torch.inference_mode()
    def _infer_batch_hf(self, images: list[Image.Image]) -> list[str]:
        if not images:
            return []
        inputs = self._processor(
            images=images,
            text=[self.task_prompt] * len(images),
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self._device)
        outputs = self._model.generate(**inputs, generation_config=self._gen_config)
        return self._processor.batch_decode(outputs, skip_special_tokens=True)

    def _reset_vllm(self) -> None:
        """Teardown and reinit vLLM engine (mirrors Cosmos Curate's _reset pattern)."""
        logger.warning("[vLLM] Resetting engine after inference failure")
        with contextlib.suppress(Exception):
            del self._llm
            del self._sampling_params
            torch.cuda.empty_cache()
        self._setup_vllm()

    @staticmethod
    def _vllm_metrics_from_outputs(  # noqa: PLR0913
        outputs: list[Any],
        *,
        inference_time_s: float,
        num_input_pages: int,
        num_valid_pages: int,
        num_skipped_pages: int,
        vllm_retries: int = 0,
    ) -> dict[str, float]:
        """Build additive per-task vLLM metrics for TaskPerfUtils aggregation."""
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_output_chars = 0
        num_length_truncated = 0
        num_empty_outputs = 0

        for req_out in outputs:
            prompt_ids = getattr(req_out, "prompt_token_ids", None)
            if prompt_ids is not None:
                total_prompt_tokens += len(prompt_ids)

            if not req_out.outputs:
                num_empty_outputs += 1
                continue

            completion = req_out.outputs[0]
            token_ids = getattr(completion, "token_ids", None)
            if token_ids is not None:
                total_output_tokens += len(token_ids)

            text = getattr(completion, "text", "") or ""
            total_output_chars += len(text)
            if not text.strip():
                num_empty_outputs += 1

            if getattr(completion, "finish_reason", None) == "length":
                num_length_truncated += 1

        return {
            "vllm_inference_time": inference_time_s,
            "num_input_pages": float(num_input_pages),
            "num_valid_pages": float(num_valid_pages),
            "num_skipped_pages": float(num_skipped_pages),
            "total_prompt_tokens": float(total_prompt_tokens),
            "total_output_tokens": float(total_output_tokens),
            "total_output_chars": float(total_output_chars),
            "num_output_length_truncated": float(num_length_truncated),
            "num_empty_outputs": float(num_empty_outputs),
            "vllm_retries": float(vllm_retries),
        }

    def _infer_vllm(self, images: list[Image.Image]) -> tuple[list[str], list[Any], int]:
        if not images:
            return [], [], 0
        prompts = [{"prompt": self.task_prompt, "multi_modal_data": {"image": img}} for img in images]

        max_retries = 3
        vllm_retries = 0
        for attempt in range(1, max_retries + 1):
            try:
                outputs = self._llm.generate(prompts, self._sampling_params)
            except Exception as e:
                logger.warning(f"[vLLM] Inference failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    vllm_retries += 1
                    self._reset_vllm()
                else:
                    raise
            else:
                texts = [output.outputs[0].text if output.outputs else "" for output in outputs]
                return texts, outputs, vllm_retries
        msg = "unreachable"
        raise RuntimeError(msg)

    def _infer_hf(self, images: list[Image.Image]) -> list[str]:
        all_outputs: list[str] = []
        for start in range(0, len(images), self.inference_batch_size):
            batch = images[start : start + self.inference_batch_size]
            try:
                all_outputs.extend(self._infer_batch_hf(batch))
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning(f"Batch inference failed for pages {start}-{start + len(batch) - 1}: {e}")
                all_outputs.extend(self._infer_hf_single_fallback(batch))
        return all_outputs

    def _infer_hf_single_fallback(self, images: list[Image.Image]) -> list[str]:
        """Process each image individually when batch inference fails."""
        results: list[str] = []
        for img in images:
            try:
                results.extend(self._infer_batch_hf([img]))
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning(f"Single page fallback failed: {e}")
                results.append("")
        return results

    # -- process --

    def process(self, task: InterleavedBatch) -> InterleavedBatch | None:
        task_df = task.to_pandas()
        images = []
        image_t0 = time.perf_counter()
        for idx, b in enumerate(task_df["binary_content"]):
            try:
                images.append(Image.open(io.BytesIO(b)))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Skipping page {idx} in {task.task_id}: {e}")
                images.append(None)
        self._log_metrics({"image_load_time": time.perf_counter() - image_t0})
        valid_mask = [img is not None for img in images]
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            return None

        if self.backend == "vllm":
            t0 = time.perf_counter()
            valid_outputs, raw_outputs, vllm_retries = self._infer_vllm(valid_images)
            inference_time_s = time.perf_counter() - t0
            self._log_metrics(
                self._vllm_metrics_from_outputs(
                    raw_outputs,
                    inference_time_s=inference_time_s,
                    num_input_pages=len(images),
                    num_valid_pages=len(valid_images),
                    num_skipped_pages=len(images) - len(valid_images),
                    vllm_retries=vllm_retries,
                )
            )
        else:
            valid_outputs = self._infer_hf(valid_images)
            self._log_metrics(
                {
                    "num_input_pages": float(len(images)),
                    "num_valid_pages": float(len(valid_images)),
                    "num_skipped_pages": float(len(images) - len(valid_images)),
                }
            )

        all_outputs = []
        valid_iter = iter(valid_outputs)
        for is_valid in valid_mask:
            all_outputs.append(next(valid_iter) if is_valid else "")

        task_df["text_content"] = all_outputs

        metadata = dict(task._metadata)
        metadata["proc_size"] = list(self._proc_size)
        metadata["model_path"] = self.model_path

        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=pa.Table.from_pandas(task_df, preserve_index=False),
            _metadata=metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class _HTTPPageResult:
    text: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    error: str | None = None


@dataclass
class NemotronParseHTTPClientStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Call Nemotron-Parse through an OpenAI-compatible HTTP endpoint.

    ``model_name`` is the served name used in requests. ``model_path`` is the
    underlying model identifier recorded for postprocessing and defaults to the
    served name. For Dynamo, set ``inference_batch_size`` to 32 or 64 concurrent
    requests per worker. ``proc_size`` must match the served model's image processor.
    """

    endpoint: str
    model_name: str
    model_path: str | None = None
    text_in_pic: bool = False
    task_prompt: str | None = None
    request_timeout_s: float = 300.0
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    inference_batch_size: int = 4
    max_tokens: int = DEFAULT_MAX_TOKENS
    proc_size: tuple[int, int] = (2048, 1664)
    name: str = "nemotron_parse_inference"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if self.task_prompt is None:
            self.task_prompt = build_task_prompt(text_in_pic=self.text_in_pic)
        self.max_retries = max(0, int(self.max_retries))
        self.inference_batch_size = int(self.inference_batch_size)
        if self.inference_batch_size < 1:
            msg = "inference_batch_size must be at least 1"
            raise ValueError(msg)
        self._generation_config = _nemotron_parse_server_generation_config(self.max_tokens)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_actor_stage": False}

    async def _query_page(
        self,
        client: AsyncOpenAIClient,
        image_bytes: bytes,
        content_type: str,
    ) -> _HTTPPageResult:
        image_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        try:
            response = await client.query_model_response(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.task_prompt or ""},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                generation_config=self._generation_config,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Inference request failed: {error}")
            return _HTTPPageResult(error=str(error))

        choice = response.choices[0] if response.choices else None
        usage = response.usage
        return _HTTPPageResult(
            text=str(getattr(getattr(choice, "message", None), "content", "") or ""),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=getattr(choice, "finish_reason", None),
        )

    async def _query_pages(self, images: list[tuple[bytes, str]]) -> list[_HTTPPageResult]:
        client = AsyncOpenAIClient(
            max_concurrent_requests=self.inference_batch_size,
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay_s,
            api_key="unused",  # pragma: allowlist secret
            base_url=self.endpoint.rstrip("/"),
            timeout=self.request_timeout_s,
        )
        try:
            return list(
                await asyncio.gather(
                    *(self._query_page(client, image_bytes, content_type) for image_bytes, content_type in images)
                )
            )
        finally:
            if hasattr(client, "client"):
                with contextlib.suppress(Exception):
                    await client.client.close()

    def _build_metrics(
        self,
        results: list[_HTTPPageResult],
        *,
        request_time_s: float,
        num_input_pages: int,
        num_valid_pages: int,
    ) -> dict[str, float]:
        total_output_tokens = float(sum(result.output_tokens for result in results))
        total_output_chars = float(sum(len(result.text) for result in results))
        return {
            "inference_server_request_time": request_time_s,
            "num_input_pages": float(num_input_pages),
            "num_valid_pages": float(num_valid_pages),
            "num_skipped_pages": float(num_input_pages - num_valid_pages),
            "total_prompt_tokens": float(sum(result.prompt_tokens for result in results)),
            "total_output_tokens": total_output_tokens,
            "total_output_chars": total_output_chars,
            "num_output_length_truncated": float(sum(result.finish_reason == "length" for result in results)),
            "num_empty_outputs": float(sum(not result.text.strip() for result in results)),
            "num_request_errors": float(sum(result.error is not None for result in results)),
        }

    def process(self, task: InterleavedBatch) -> InterleavedBatch | None:
        task_df = task.to_pandas()
        valid_mask: list[bool] = []
        images: list[tuple[bytes, str]] = []

        for _, row in task_df.iterrows():
            raw_bytes = row.get("binary_content")
            try:
                image_bytes = bytes(raw_bytes) if raw_bytes is not None else b""
            except TypeError:
                image_bytes = b""
            is_valid = bool(image_bytes)
            valid_mask.append(is_valid)
            if is_valid:
                images.append((image_bytes, str(row.get("content_type") or "image/png")))

        if not images:
            return None

        request_start = time.perf_counter()
        results = asyncio.run(self._query_pages(images))
        request_time_s = time.perf_counter() - request_start
        self._log_metrics(
            self._build_metrics(
                results,
                request_time_s=request_time_s,
                num_input_pages=len(valid_mask),
                num_valid_pages=len(images),
            )
        )

        result_iter = iter(result.text for result in results)
        task_df["text_content"] = [next(result_iter) if is_valid else "" for is_valid in valid_mask]

        metadata = dict(task._metadata)
        metadata["proc_size"] = list(self.proc_size)
        metadata["model_path"] = self.model_path or self.model_name
        metadata["inference_server_endpoint"] = self.endpoint

        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=pa.Table.from_pandas(task_df, preserve_index=False),
            _metadata=metadata,
            _stage_perf=task._stage_perf,
        )
