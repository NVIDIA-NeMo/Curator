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

"""MinerU-HTML extraction stages: DripperHTMLExtractionStage, PreprocessStage, InferenceStage, PostprocessStage."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import pandas as pd
from loguru import logger

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.translation.utils.async_utils import run_async_safe
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.models.client.llm_client import AsyncLLMClient

from nemo_curator.stages.text.experimental.dripper.stage import (
    _DRIPPER_EMPTY_INPUT_COL,
    _DRIPPER_LAYOUT_FINALIZED_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _DRIPPER_PROMPT_COL,
    _STRUCTURED_OUTPUT_MODES,
    _append_warning,
    _apply_fallback_extraction,
    _case_has_item_ids,
    _coerce_html,
    _coerce_optional_str,
    _coerce_usage_int,
    _count_item_ids,
    _DripperInferenceResult,
    _DripperPostResult,
    _DripperPrepResult,
    _DripperRowResult,
    _generation_config_for_item_count,
    _get_processed_attr,
    _is_empty_document_error,
    _load_mineru_html_bindings,
    _MinerUHTMLBindings,
    _numeric_series_or_zero,
    _query_dripper_model,
    _rebuild_batch,
    _run_dripper_health_check,
    _sanitize_case_output_html,
    _with_structured_output_config,
)


def _col_str_list(df: pd.DataFrame, col: str, n: int) -> list[str]:
    return df[col].astype(str).tolist() if col in df else [""] * n


def _col_float_list(df: pd.DataFrame, col: str, n: int) -> list[float]:
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).tolist() if col in df else [0.0] * n


def _col_int_list(df: pd.DataFrame, col: str, n: int) -> list[int]:
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).tolist() if col in df else [0] * n


@runtime_checkable
class _HasDynamicTokenParams(Protocol):
    dynamic_max_token_padding: int
    dynamic_max_tokens_per_item: int
    dynamic_min_max_tokens: int


@runtime_checkable
class _HasLLMClientParams(Protocol):
    client: Any
    model_name: str
    max_concurrent_requests: int
    structured_output_mode: str


def _validate_dynamic_token_params(obj: _HasDynamicTokenParams) -> None:
    if obj.dynamic_max_token_padding < 0:
        msg = "dynamic_max_token_padding must be non-negative"
        raise ValueError(msg)
    if obj.dynamic_max_tokens_per_item <= 0:
        msg = "dynamic_max_tokens_per_item must be positive"
        raise ValueError(msg)
    if obj.dynamic_min_max_tokens <= 0:
        msg = "dynamic_min_max_tokens must be positive"
        raise ValueError(msg)


def _validate_llm_client_params(obj: _HasLLMClientParams, class_name: str) -> None:
    if obj.client is None:
        msg = f"{class_name} requires a non-None 'client' (AsyncLLMClient)"
        raise ValueError(msg)
    obj.model_name = obj.model_name.strip()
    if not obj.model_name:
        msg = f"{class_name} requires a non-empty 'model_name'"
        raise ValueError(msg)
    if obj.max_concurrent_requests <= 0:
        msg = "max_concurrent_requests must be positive"
        raise ValueError(msg)
    if obj.structured_output_mode not in _STRUCTURED_OUTPUT_MODES:
        msg = f"structured_output_mode must be one of {sorted(_STRUCTURED_OUTPUT_MODES)}"
        raise ValueError(msg)


def _apply_conversion_to_row_result(case: object, base: _DripperRowResult, conversion_error: str) -> _DripperRowResult:
    output_data = getattr(case, "output_data", None)
    main_html = getattr(output_data, "main_html", "") if output_data is not None else ""
    main_content = getattr(output_data, "main_content", "") or ""
    warning = base.warning
    error = ""
    if conversion_error:
        if _is_empty_document_error(conversion_error) and not str(main_html).strip():
            warning = _append_warning(warning, conversion_error)
        else:
            error = conversion_error
    return replace(base, main_html=main_html, main_content=main_content, error=error, warning=warning)


@dataclass(kw_only=True)
class _DripperColumnsMixin:
    html_col: str = "html"
    url_col: str | None = "url"
    raw_response_col: str = "dripper_response"
    preprocess_time_col: str = "dripper_preprocess_time_s"
    inference_time_col: str = "dripper_inference_time_s"
    postprocess_time_col: str = "dripper_postprocess_time_s"
    total_time_col: str = "dripper_time_s"
    error_col: str = "dripper_error"
    warning_col: str = "dripper_warning"
    item_count_col: str = "dripper_item_count"
    prompt_chars_col: str = "dripper_prompt_chars"
    request_max_tokens_col: str = "dripper_request_max_tokens"
    prompt_tokens_col: str = "dripper_prompt_tokens"
    completion_tokens_col: str = "dripper_completion_tokens"
    total_tokens_col: str = "dripper_total_tokens"
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"
    prompt_version: str = "short_compact"
    generation_config: GenerationConfig | None = None
    dynamic_max_tokens: bool = False
    dynamic_max_token_padding: int = 16
    dynamic_max_tokens_per_item: int = 6
    dynamic_min_max_tokens: int = 32


@dataclass(kw_only=True)
class DripperHTMLExtractionStage(_DripperColumnsMixin, ProcessingStage[DocumentBatch, DocumentBatch]):
    name: str = "DripperHTMLExtractionStage"
    client: AsyncLLMClient | None
    model_name: str
    output_html_col: str = "dripper_html"
    output_content_col: str = "dripper_content"
    output_format: str = "mm_md"
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    structured_output_mode: Literal["none", "structured_outputs", "guided_regex"] = "none"
    max_concurrent_requests: int = 64
    health_check: bool = True
    keep_intermediate: bool = False

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _fallback_handler: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        _validate_llm_client_params(self, "DripperHTMLExtractionStage")
        _validate_dynamic_token_params(self)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.html_col]

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = [
            self.output_html_col,
            self.output_content_col,
            self.raw_response_col,
            self.preprocess_time_col,
            self.inference_time_col,
            self.postprocess_time_col,
            self.total_time_col,
            self.error_col,
            self.warning_col,
            self.item_count_col,
            self.prompt_chars_col,
            self.request_max_tokens_col,
            self.prompt_tokens_col,
            self.completion_tokens_col,
            self.total_tokens_col,
        ]
        if self.keep_intermediate:
            columns.extend([self.simplified_html_col, self.mapped_html_col])
        return ["data"], columns

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._fallback_handler = self._bindings.get_fallback_handler(self.fallback)
        self.client.setup()
        if self.health_check:
            run_async_safe(lambda: _run_dripper_health_check(self.client, self.model_name, self.generation_config))
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        if self.html_col not in df.columns:
            msg = f"Input batch is missing required HTML column: {self.html_col!r}"
            raise ValueError(msg)

        html_values = df[self.html_col].tolist()
        if self.url_col is not None and self.url_col in df.columns:
            url_values = df[self.url_col].tolist()
        else:
            url_values = [None] * len(df)

        results = run_async_safe(lambda: self._extract_all_async(html_values, url_values))
        df[self.output_html_col] = [r.main_html for r in results]
        df[self.output_content_col] = [r.main_content for r in results]
        df[self.raw_response_col] = [r.raw_response for r in results]
        df[self.preprocess_time_col] = [r.preprocess_time_s for r in results]
        df[self.inference_time_col] = [r.inference_time_s for r in results]
        df[self.postprocess_time_col] = [r.postprocess_time_s for r in results]
        df[self.total_time_col] = [r.total_time_s for r in results]
        df[self.error_col] = [r.error for r in results]
        df[self.warning_col] = [r.warning for r in results]
        df[self.item_count_col] = [r.item_count for r in results]
        df[self.prompt_chars_col] = [r.prompt_chars for r in results]
        df[self.request_max_tokens_col] = [r.request_max_tokens for r in results]
        df[self.prompt_tokens_col] = [r.prompt_tokens for r in results]
        df[self.completion_tokens_col] = [r.completion_tokens for r in results]
        df[self.total_tokens_col] = [r.total_tokens for r in results]
        if self.keep_intermediate:
            df[self.simplified_html_col] = [r.simplified_html for r in results]
            df[self.mapped_html_col] = [r.mapped_html for r in results]

        return _rebuild_batch(batch, df)

    async def _extract_all_async(self, html_values: list[object], url_values: list[object]) -> list[_DripperRowResult]:
        sem = asyncio.Semaphore(self.max_concurrent_requests)

        async def _extract_one_throttled(html_value: object, url_value: object) -> _DripperRowResult:
            async with sem:
                return await self._extract_one_async(html_value, url_value)

        tasks = [
            _extract_one_throttled(html_value, url_value)
            for html_value, url_value in zip(html_values, url_values, strict=False)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[_DripperRowResult] = []
        for idx, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.error("Dripper extraction failed for row {}: {}", idx, result)
                results.append(_DripperRowResult(error=str(result)))
            else:
                results.append(result)
        return results

    def _preprocess_case(self, case: object) -> tuple[object, int, str, str, bool]:
        case = self._bindings.simplify_single_input(case)
        item_count = _count_item_ids(case)
        if not _case_has_item_ids(case):
            case = self._bindings.extract_main_html_fallback(case, fallback_handler=self._fallback_handler)
            return (
                case,
                item_count,
                "",
                "no _item_id attributes after simplification; used fallback without LLM",
                False,
            )
        case = self._bindings.build_prompt(case, prompt_version=self.prompt_version)
        prompt = case.generate_input.full_prompt
        return case, item_count, prompt, "", True

    async def _run_inference_async(
        self, case: object, prompt: str, item_count: int
    ) -> tuple[object, str, int, int, int, int]:
        generation_config = _with_structured_output_config(
            _generation_config_for_item_count(self, item_count), prompt, self.structured_output_mode
        )
        request_max_tokens = generation_config.max_tokens or 0
        raw_response, prompt_tokens, completion_tokens, total_tokens = await _query_dripper_model(
            self.client, self.model_name, [{"role": "user", "content": prompt}], generation_config
        )
        case.generate_output = self._bindings.generate_output_cls(response=raw_response)
        case = self._bindings.parse_result(case)
        case = self._bindings.extract_main_html_single(case)
        return case, raw_response, request_max_tokens, prompt_tokens, completion_tokens, total_tokens

    async def _extract_one_async(self, html_value: object, url_value: object) -> _DripperRowResult:
        start_total = time.perf_counter()
        html = _coerce_html(html_value)
        if not html.strip():
            return _DripperRowResult(total_time_s=time.perf_counter() - start_total, warning="empty HTML input")

        url = _coerce_optional_str(url_value)
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html, url=url))
        raw_response = ""
        preprocess_time_s = 0.0
        inference_time_s = 0.0
        postprocess_time_s = 0.0
        warning = ""
        item_count = 0
        prompt_chars = 0
        request_max_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        try:
            t0 = time.perf_counter()
            case, item_count, prompt, warning, needs_llm = self._preprocess_case(case)
            preprocess_time_s = time.perf_counter() - t0
            if needs_llm:
                prompt_chars = len(prompt)
                t1 = time.perf_counter()
                (
                    case,
                    raw_response,
                    request_max_tokens,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                ) = await self._run_inference_async(case, prompt, item_count)
                inference_time_s = time.perf_counter() - t1
        except Exception as exc:  # noqa: BLE001
            if preprocess_time_s == 0.0:
                preprocess_time_s = time.perf_counter() - start_total
            primary_error = str(exc)
            logger.debug("Dripper primary extraction failed, applying {} fallback: {}", self.fallback, primary_error)
            try:
                t2 = time.perf_counter()
                case = self._bindings.extract_main_html_fallback(case, fallback_handler=self._fallback_handler)
                postprocess_time_s += time.perf_counter() - t2
                warning = primary_error
            except Exception as fallback_exc:  # noqa: BLE001
                return _DripperRowResult(
                    raw_response=raw_response,
                    preprocess_time_s=preprocess_time_s,
                    inference_time_s=inference_time_s,
                    postprocess_time_s=postprocess_time_s,
                    total_time_s=time.perf_counter() - start_total,
                    error=f"{primary_error}; fallback failed: {fallback_exc}",
                    warning=primary_error,
                    simplified_html=_get_processed_attr(case, "simpled_html"),
                    mapped_html=_get_processed_attr(case, "map_html"),
                    item_count=item_count,
                    prompt_chars=prompt_chars,
                    request_max_tokens=request_max_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

        partial = _DripperRowResult(
            raw_response=raw_response,
            warning=warning,
            preprocess_time_s=preprocess_time_s,
            inference_time_s=inference_time_s,
            postprocess_time_s=postprocess_time_s,
            item_count=item_count,
            prompt_chars=prompt_chars,
            request_max_tokens=request_max_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            simplified_html=_get_processed_attr(case, "simpled_html"),
            mapped_html=_get_processed_attr(case, "map_html"),
        )
        t3 = time.perf_counter()
        conversion_error, case = self._convert_case(case)
        postprocess_time_s += time.perf_counter() - t3
        partial = replace(
            partial, postprocess_time_s=postprocess_time_s, total_time_s=time.perf_counter() - start_total
        )
        return self._apply_conversion_result(case, partial, conversion_error)

    def _convert_case(self, case: object) -> tuple[str, object]:
        try:
            _sanitize_case_output_html(case)
            return "", self._bindings.convert2content(case, output_format=self.output_format)
        except Exception as exc:  # noqa: BLE001
            conversion_error = str(exc)
            logger.debug("Dripper content conversion failed: {}", conversion_error)
            return conversion_error, case

    def _apply_conversion_result(
        self, case: object, base: _DripperRowResult, conversion_error: str
    ) -> _DripperRowResult:
        return _apply_conversion_to_row_result(case, base, conversion_error)


@dataclass(kw_only=True)
class DripperHTMLPreprocessStage(_DripperColumnsMixin, ProcessingStage[DocumentBatch, DocumentBatch]):
    name: str = "DripperHTMLPreprocessStage"
    worker_count: int | None = None

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        _validate_dynamic_token_params(self)
        if self.worker_count is not None and self.worker_count <= 0:
            msg = "worker_count must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.html_col]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.raw_response_col,
            self.preprocess_time_col,
            self.inference_time_col,
            self.postprocess_time_col,
            self.total_time_col,
            self.error_col,
            self.warning_col,
            self.item_count_col,
            self.prompt_chars_col,
            self.request_max_tokens_col,
            self.prompt_tokens_col,
            self.completion_tokens_col,
            self.total_tokens_col,
            self.simplified_html_col,
            self.mapped_html_col,
            _DRIPPER_PROMPT_COL,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
        ]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        if self.html_col not in df.columns:
            msg = f"Input batch is missing required HTML column: {self.html_col!r}"
            raise ValueError(msg)

        html_values = df[self.html_col].tolist()
        if self.url_col is not None and self.url_col in df.columns:
            url_values = df[self.url_col].tolist()
        else:
            url_values = [None] * len(df)

        results = [
            self._prepare_one(html_value, url_value)
            for html_value, url_value in zip(html_values, url_values, strict=False)
        ]

        pt = [r.preprocess_time_s for r in results]
        df = df.assign(
            **{
                self.raw_response_col: "",
                self.preprocess_time_col: pt,
                self.inference_time_col: 0.0,
                self.postprocess_time_col: 0.0,
                self.total_time_col: pt,
                self.error_col: "",
                self.warning_col: [r.warning for r in results],
                self.item_count_col: [r.item_count for r in results],
                self.prompt_chars_col: [r.prompt_chars for r in results],
                self.request_max_tokens_col: [r.request_max_tokens for r in results],
                self.prompt_tokens_col: 0,
                self.completion_tokens_col: 0,
                self.total_tokens_col: 0,
                self.simplified_html_col: [r.simplified_html for r in results],
                self.mapped_html_col: [r.mapped_html for r in results],
                _DRIPPER_PROMPT_COL: [r.prompt for r in results],
                _DRIPPER_NEEDS_LLM_COL: [r.needs_llm for r in results],
                _DRIPPER_PRIMARY_ERROR_COL: [r.primary_error for r in results],
                _DRIPPER_EMPTY_INPUT_COL: [r.empty_input for r in results],
            }
        )

        self._log_metrics(
            {
                "preprocess_rows": float(len(df)),
                "preprocess_llm_rows": float(sum(r.needs_llm for r in results)),
                "preprocess_fallback_rows": float(sum((not r.needs_llm) and (not r.empty_input) for r in results)),
            }
        )
        return _rebuild_batch(batch, df)

    def _prepare_one(self, html_value: object, url_value: object) -> _DripperPrepResult:
        started = time.perf_counter()
        html = _coerce_html(html_value)
        if not html.strip():
            return _DripperPrepResult(
                empty_input=True,
                preprocess_time_s=time.perf_counter() - started,
                warning="empty HTML input",
            )

        url = _coerce_optional_str(url_value)
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html, url=url))
        simplified_html = ""
        mapped_html = ""
        item_count = 0
        try:
            case = self._bindings.simplify_single_input(case)
            simplified_html = _get_processed_attr(case, "simpled_html")
            mapped_html = _get_processed_attr(case, "map_html")
            item_count = _count_item_ids(case)
            if not _case_has_item_ids(case):
                return _DripperPrepResult(
                    needs_llm=False,
                    preprocess_time_s=time.perf_counter() - started,
                    warning="no _item_id attributes after simplification; used fallback without LLM",
                    simplified_html=simplified_html,
                    mapped_html=mapped_html,
                    item_count=item_count,
                )

            case = self._bindings.build_prompt(case, prompt_version=self.prompt_version)
            prompt = case.generate_input.full_prompt
            return _DripperPrepResult(
                prompt=prompt,
                needs_llm=True,
                preprocess_time_s=time.perf_counter() - started,
                simplified_html=simplified_html,
                mapped_html=mapped_html,
                item_count=item_count,
                prompt_chars=len(prompt),
                request_max_tokens=_generation_config_for_item_count(self, item_count).max_tokens or 0,
            )
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper preprocessing failed; postprocess stage will apply fallback: {}", primary_error)
            return _DripperPrepResult(
                needs_llm=False,
                preprocess_time_s=time.perf_counter() - started,
                primary_error=primary_error,
                warning=primary_error,
                simplified_html=simplified_html,
                mapped_html=mapped_html,
                item_count=item_count,
            )


@dataclass(kw_only=True)
class DripperHTMLInferenceStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name: str = "DripperHTMLInferenceStage"
    client: AsyncLLMClient | None
    model_name: str
    raw_response_col: str = "dripper_response"
    inference_time_col: str = "dripper_inference_time_s"
    warning_col: str = "dripper_warning"
    item_count_col: str = "dripper_item_count"
    request_max_tokens_col: str = "dripper_request_max_tokens"
    prompt_tokens_col: str = "dripper_prompt_tokens"
    completion_tokens_col: str = "dripper_completion_tokens"
    total_tokens_col: str = "dripper_total_tokens"
    generation_config: GenerationConfig | None = None
    structured_output_mode: Literal["none", "structured_outputs", "guided_regex"] = "none"
    max_concurrent_requests: int = 64
    health_check: bool = False
    worker_count: int | None = None

    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        _validate_llm_client_params(self, "DripperHTMLInferenceStage")
        if self.worker_count is not None and self.worker_count <= 0:
            msg = "worker_count must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [_DRIPPER_PROMPT_COL, _DRIPPER_NEEDS_LLM_COL, self.request_max_tokens_col]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.raw_response_col,
            self.inference_time_col,
            self.warning_col,
            self.prompt_tokens_col,
            self.completion_tokens_col,
            self.total_tokens_col,
            _DRIPPER_PRIMARY_ERROR_COL,
        ]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self.client.setup()
        if self.health_check:
            run_async_safe(lambda: _run_dripper_health_check(self.client, self.model_name, self.generation_config))
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        results = run_async_safe(lambda: self._infer_all_async(df))

        n = len(df)
        needs_llm = df[_DRIPPER_NEEDS_LLM_COL].astype(bool).tolist()
        existing_raw_responses = _col_str_list(df, self.raw_response_col, n)
        existing_inference_times = _col_float_list(df, self.inference_time_col, n)
        existing_prompt_tokens = _col_int_list(df, self.prompt_tokens_col, n)
        existing_completion_tokens = _col_int_list(df, self.completion_tokens_col, n)
        existing_total_tokens = _col_int_list(df, self.total_tokens_col, n)
        existing_warnings = df[self.warning_col].astype(str) if self.warning_col in df else pd.Series([""] * n)
        existing_primary_errors = (
            df[_DRIPPER_PRIMARY_ERROR_COL].astype(str) if _DRIPPER_PRIMARY_ERROR_COL in df else pd.Series([""] * n)
        )
        df[self.raw_response_col] = [
            r.raw_response if q else e for r, q, e in zip(results, needs_llm, existing_raw_responses, strict=True)
        ]
        df[self.inference_time_col] = [
            r.inference_time_s if q else e
            for r, q, e in zip(results, needs_llm, existing_inference_times, strict=True)
        ]
        df[self.warning_col] = [
            _append_warning(ew, r.warning) for ew, r in zip(existing_warnings.tolist(), results, strict=True)
        ]
        df[_DRIPPER_PRIMARY_ERROR_COL] = [
            _append_warning(ee, r.primary_error)
            for ee, r in zip(existing_primary_errors.tolist(), results, strict=True)
        ]
        for col, attr, existing in (
            (self.prompt_tokens_col, "prompt_tokens", existing_prompt_tokens),
            (self.completion_tokens_col, "completion_tokens", existing_completion_tokens),
            (self.total_tokens_col, "total_tokens", existing_total_tokens),
        ):
            df[col] = [getattr(r, attr) if q else e for r, q, e in zip(results, needs_llm, existing, strict=True)]

        llm_prompts = [
            str(row.get(_DRIPPER_PROMPT_COL, "") or "")
            for _, row in df.iterrows()
            if bool(row.get(_DRIPPER_NEEDS_LLM_COL, False))
        ]
        non_empty_llm_prompts = [prompt for prompt in llm_prompts if prompt.strip()]
        unique_llm_prompts = len(set(non_empty_llm_prompts))
        self._log_metrics(
            {
                "inference_rows": float(len(df)),
                "inference_llm_rows": float(sum(bool(v) for v in df[_DRIPPER_NEEDS_LLM_COL].tolist())),
                "inference_unique_llm_prompts": float(unique_llm_prompts),
                "inference_dedup_saved_rows": float(len(non_empty_llm_prompts) - unique_llm_prompts),
                "inference_errors": float(sum(1 for r in results if r.primary_error)),
            }
        )
        return _rebuild_batch(batch, df)

    async def _infer_all_async(self, df: pd.DataFrame) -> list[_DripperInferenceResult]:
        sem = asyncio.Semaphore(self.max_concurrent_requests)
        prompts = df[_DRIPPER_PROMPT_COL].astype(str).tolist()
        needs_llm = df[_DRIPPER_NEEDS_LLM_COL].astype(bool).tolist()
        request_max_tokens = _col_int_list(df, self.request_max_tokens_col, len(df))

        async def _infer_one_throttled(prompt: str, row_max_tokens: int) -> _DripperInferenceResult:
            async with sem:
                return await self._infer_one_async(prompt, row_max_tokens)

        grouped_indexes: dict[tuple[str, int], list[int]] = defaultdict(list)
        results: list[_DripperInferenceResult | None] = [None] * len(df)
        for idx, (prompt, should_query, row_max_tokens) in enumerate(
            zip(prompts, needs_llm, request_max_tokens, strict=True)
        ):
            if not should_query:
                results[idx] = _DripperInferenceResult()
            elif not prompt.strip():
                results[idx] = _DripperInferenceResult(
                    primary_error="empty Dripper prompt", warning="empty Dripper prompt"
                )
            else:
                grouped_indexes[(prompt, row_max_tokens)].append(idx)

        tasks = {key: _infer_one_throttled(prompt=key[0], row_max_tokens=key[1]) for key in grouped_indexes}
        raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for (_key, indexes), result in zip(grouped_indexes.items(), raw_results, strict=True):
            if isinstance(result, BaseException):
                logger.error("Dripper inference failed for prompt group {} rows: {}", len(indexes), result)
                error = str(result)
                first_result = _DripperInferenceResult(primary_error=error, warning=error)
            else:
                first_result = result
            first_idx = indexes[0]
            results[first_idx] = first_result
            for duplicate_idx in indexes[1:]:
                results[duplicate_idx] = replace(
                    first_result,
                    inference_time_s=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )

        return [result if result is not None else _DripperInferenceResult() for result in results]

    async def _infer_one_async(self, prompt: str, row_max_tokens: int) -> _DripperInferenceResult:
        started = time.perf_counter()
        try:
            generation_config = self.generation_config or GenerationConfig()
            if row_max_tokens > 0 and generation_config.max_tokens != row_max_tokens:
                generation_config = replace(generation_config, max_tokens=row_max_tokens)
            generation_config = _with_structured_output_config(generation_config, prompt, self.structured_output_mode)
            raw_response, prompt_tokens, completion_tokens, total_tokens = await self._query_model_with_usage(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                generation_config=generation_config,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            logger.debug("Dripper inference failed; postprocess stage will apply fallback: {}", error)
            return _DripperInferenceResult(
                inference_time_s=time.perf_counter() - started,
                primary_error=error,
                warning=error,
            )
        return _DripperInferenceResult(
            raw_response=raw_response,
            inference_time_s=time.perf_counter() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def _query_model_with_usage(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        generation_config: GenerationConfig,
    ) -> tuple[str, int, int, int]:
        query_model_with_usage = getattr(self.client, "query_model_with_usage", None)
        if callable(query_model_with_usage):
            response = await query_model_with_usage(
                model=model,
                messages=messages,
                generation_config=generation_config,
            )
            contents = getattr(response, "contents", [])
            return (
                contents[0] if contents else "",
                _coerce_usage_int(getattr(response, "prompt_tokens", None)),
                _coerce_usage_int(getattr(response, "completion_tokens", None)),
                _coerce_usage_int(getattr(response, "total_tokens", None)),
            )

        response = await self.client.query_model(
            model=model,
            messages=messages,
            generation_config=generation_config,
        )
        return response[0] if response else "", 0, 0, 0


@dataclass(kw_only=True)
class DripperHTMLPostprocessStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name: str = "DripperHTMLPostprocessStage"
    html_col: str = "html"
    url_col: str | None = "url"
    output_html_col: str = "dripper_html"
    output_content_col: str = "dripper_content"
    raw_response_col: str = "dripper_response"
    preprocess_time_col: str = "dripper_preprocess_time_s"
    inference_time_col: str = "dripper_inference_time_s"
    postprocess_time_col: str = "dripper_postprocess_time_s"
    total_time_col: str = "dripper_time_s"
    error_col: str = "dripper_error"
    warning_col: str = "dripper_warning"
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    output_format: str = "mm_md"
    keep_intermediate: bool = False
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"
    worker_count: int | None = None

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _fallback_handler: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if self.worker_count is not None and self.worker_count <= 0:
            msg = "worker_count must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.html_col,
            self.raw_response_col,
            self.simplified_html_col,
            self.mapped_html_col,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = [
            self.output_html_col,
            self.output_content_col,
            self.postprocess_time_col,
            self.total_time_col,
            self.error_col,
            self.warning_col,
        ]
        if self.keep_intermediate:
            columns.extend([self.simplified_html_col, self.mapped_html_col])
        return ["data"], columns

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._fallback_handler = self._bindings.get_fallback_handler(self.fallback)
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        html_values = df[self.html_col].tolist()
        if self.url_col is not None and self.url_col in df.columns:
            url_values = df[self.url_col].tolist()
        else:
            url_values = [None] * len(df)

        results = [
            self._postprocess_one(row, html_value, url_value)
            for (_, row), html_value, url_value in zip(df.iterrows(), html_values, url_values, strict=True)
        ]

        preprocess_times = _numeric_series_or_zero(df, self.preprocess_time_col)
        inference_times = _numeric_series_or_zero(df, self.inference_time_col)
        postprocess_times = pd.Series([r.postprocess_time_s for r in results], index=df.index)

        df[self.output_html_col] = [r.main_html for r in results]
        df[self.output_content_col] = [r.main_content for r in results]
        df[self.postprocess_time_col] = postprocess_times
        df[self.total_time_col] = preprocess_times + inference_times + postprocess_times
        df[self.error_col] = [r.error for r in results]
        df[self.warning_col] = [r.warning for r in results]

        drop_cols = [
            _DRIPPER_PROMPT_COL,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
            _DRIPPER_LAYOUT_FINALIZED_COL,
        ]
        if not self.keep_intermediate:
            drop_cols.extend([self.simplified_html_col, self.mapped_html_col])
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        self._log_metrics(
            {
                "postprocess_rows": float(len(df)),
                "postprocess_errors": float(sum(1 for r in results if r.error)),
                "postprocess_warnings": float(sum(1 for r in results if r.warning)),
            }
        )
        return _rebuild_batch(batch, df)

    def _postprocess_one(self, row: pd.Series, html_value: object, url_value: object) -> _DripperPostResult:
        started = time.perf_counter()
        warning = str(row.get(self.warning_col, "") or "")
        primary_error = str(row.get(_DRIPPER_PRIMARY_ERROR_COL, "") or "")
        if bool(row.get(_DRIPPER_LAYOUT_FINALIZED_COL, False)):
            return _DripperPostResult(
                main_html=str(row.get(self.output_html_col, "") or ""),
                main_content=row.get(self.output_content_col, "") or "",
                postprocess_time_s=float(row.get(self.postprocess_time_col, 0.0) or 0.0),
                error=str(row.get(self.error_col, "") or ""),
                warning=warning,
            )
        html = _coerce_html(html_value)
        if bool(row.get(_DRIPPER_EMPTY_INPUT_COL, False)) or not html.strip():
            return _DripperPostResult(
                postprocess_time_s=time.perf_counter() - started,
                warning=warning or "empty HTML input",
            )

        url = _coerce_optional_str(url_value)
        case = self._build_case(
            html=html,
            url=url,
            simplified_html=str(row.get(self.simplified_html_col, "") or ""),
            mapped_html=str(row.get(self.mapped_html_col, "") or ""),
        )
        raw_response = str(row.get(self.raw_response_col, "") or "")
        needs_llm = bool(row.get(_DRIPPER_NEEDS_LLM_COL, False))

        case, warning, fallback_error = self._postprocess_prepare_case(
            case,
            raw_response=raw_response,
            needs_llm=needs_llm,
            primary_error=primary_error,
            warning=warning,
        )
        if fallback_error:
            return _DripperPostResult(
                postprocess_time_s=time.perf_counter() - started,
                error=fallback_error,
                warning=warning,
            )

        conversion_error = ""
        try:
            _sanitize_case_output_html(case)
            case = self._bindings.convert2content(case, output_format=self.output_format)
        except Exception as exc:  # noqa: BLE001
            conversion_error = str(exc)
            logger.debug("Dripper content conversion failed: {}", conversion_error)

        output_data = getattr(case, "output_data", None)
        main_html = getattr(output_data, "main_html", "") if output_data is not None else ""
        main_content = getattr(output_data, "main_content", "") if output_data is not None else ""
        if main_content is None:
            main_content = ""
        error = ""
        if conversion_error:
            if _is_empty_document_error(conversion_error) and not str(main_html).strip():
                warning = _append_warning(warning, conversion_error)
            else:
                error = conversion_error

        return _DripperPostResult(
            main_html=main_html,
            main_content=main_content,
            postprocess_time_s=time.perf_counter() - started,
            error=error,
            warning=warning,
        )

    def _postprocess_prepare_case(
        self,
        case: object,
        *,
        raw_response: str,
        needs_llm: bool,
        primary_error: str,
        warning: str,
    ) -> tuple[object, str, str]:
        if needs_llm and raw_response:
            try:
                case.generate_output = self._bindings.generate_output_cls(response=raw_response)
                case = self._bindings.parse_result(case)
                case = self._bindings.extract_main_html_single(case)
            except Exception as exc:  # noqa: BLE001
                primary_error = _append_warning(primary_error, str(exc))
                logger.debug("Dripper parse/extract failed, applying {} fallback: {}", self.fallback, primary_error)
                fallback_result = _apply_fallback_extraction(
                    self._bindings, self._fallback_handler, case, primary_error
                )
                warning = _append_warning(warning, fallback_result[1])
                return fallback_result[0], warning, fallback_result[2]
            return case, warning, ""
        if needs_llm and not primary_error:
            primary_error = "empty Dripper response"
        fallback_result = _apply_fallback_extraction(self._bindings, self._fallback_handler, case, primary_error)
        warning = _append_warning(warning, fallback_result[1])
        return fallback_result[0], warning, fallback_result[2]

    def _build_case(self, *, html: str, url: str | None, simplified_html: str, mapped_html: str) -> object:
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html, url=url))
        if simplified_html or mapped_html:
            case.process_data = self._bindings.process_data_cls(simpled_html=simplified_html, map_html=mapped_html)
        return case
