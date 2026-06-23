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

"""DripperHTMLLayoutTemplateStage: layout clustering + LBP template propagation for CC-scale HTML extraction."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from loguru import logger

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper._layout_planning import (
    _LAYOUT_PAGE_SIGNATURE_MODES,
    DripperLayoutAdvancedConfig,
    _build_failed_layout_fallback_groups,
    _build_layout_group_plans,
    _coerce_optional_float,
    _coerce_positive_int,
    _item_id_response,
    _labels_to_webkit_response,
    _LayoutGroupPlan,
    _LayoutPlanningConfig,
    _select_validation_indexes,
    _split_fallback_groups_by_signature,
    _token_f1,
)
from nemo_curator.stages.text.experimental.dripper.stage import (
    _DRIPPER_EMPTY_INPUT_COL,
    _DRIPPER_LAYOUT_FINALIZED_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _DRIPPER_PROMPT_COL,
    _STRUCTURED_OUTPUT_MODES,
    _append_warning,
    _apply_fallback_extraction,
    _coerce_html,
    _coerce_optional_str,
    _coerce_usage_int,
    _DripperInferenceResult,
    _DripperPostResult,
    _is_empty_document_error,
    _item_ids_in_html,
    _LLMWebKitBindings,
    _load_llm_web_kit_bindings,
    _load_mineru_html_bindings,
    _MinerUHTMLBindings,
    _numeric_series_or_zero,
    _query_dripper_model,
    _rebuild_batch,
    _run_dripper_health_check,
    _sanitize_case_output_html,
    _with_structured_output_config,
)
from nemo_curator.stages.text.experimental.translation.utils.async_utils import run_async_safe
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.models.client.llm_client import AsyncLLMClient

_DRIPPER_OUTPUT_HTML_COL = "dripper_html"
_DRIPPER_OUTPUT_CONTENT_COL = "dripper_content"
_DRIPPER_RAW_RESPONSE_COL = "dripper_response"
_DRIPPER_PREPROCESS_TIME_COL = "dripper_preprocess_time_s"
_DRIPPER_INFERENCE_TIME_COL = "dripper_inference_time_s"
_DRIPPER_POSTPROCESS_TIME_COL = "dripper_postprocess_time_s"
_DRIPPER_TOTAL_TIME_COL = "dripper_time_s"
_DRIPPER_ERROR_COL = "dripper_error"
_DRIPPER_WARNING_COL = "dripper_warning"
_DRIPPER_ITEM_COUNT_COL = "dripper_item_count"
_DRIPPER_REQUEST_MAX_TOKENS_COL = "dripper_request_max_tokens"
_DRIPPER_PROMPT_TOKENS_COL = "dripper_prompt_tokens"
_DRIPPER_COMPLETION_TOKENS_COL = "dripper_completion_tokens"
_DRIPPER_TOTAL_TOKENS_COL = "dripper_total_tokens"
_DRIPPER_SIMPLIFIED_HTML_COL = "dripper_simplified_html"
_DRIPPER_MAPPED_HTML_COL = "dripper_mapped_html"

_LAYOUT_TEMPLATE_LARGE_HOST_MODES = {"standalone", "feature_hash", "dom_path_hash"}
_LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES = {"raw_html", "mapped_item_ids"}


@dataclass(frozen=True)
class _LayoutTemplateRowResult:
    raw_response: str = ""
    inference_time_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    main_html: str = ""
    main_content: Any = ""
    postprocess_time_s: float = 0.0
    error: str = ""
    warning: str = ""
    primary_error: str = ""
    deferred_llm: bool = False
    layout_finalized: bool = True
    layout_cluster: str = ""
    layout_representative: bool = False
    layout_propagated: bool = False
    layout_propagation_success: bool = False
    layout_fallback_llm: bool = False
    layout_standalone_llm: bool = False
    layout_pending_propagation: bool = False
    layout_mapping_json: str = ""


@dataclass(frozen=True)
class _LayoutGroupOutcome:
    results: dict[int, _LayoutTemplateRowResult]
    accepted: bool = True
    failure_reason: str = ""


@dataclass(frozen=True)
class _LayoutProcessContext:
    df: pd.DataFrame
    semaphore: asyncio.Semaphore
    propagation_semaphore: asyncio.Semaphore
    inference_cache: _InferenceCache
    inference_cache_lock: asyncio.Lock
    needs_llm: list[bool]


_InferenceCache = dict[tuple[str, int], asyncio.Task[_DripperInferenceResult]]


def _inference_token_fields(r: _DripperInferenceResult) -> dict[str, object]:
    return {
        "raw_response": r.raw_response,
        "inference_time_s": r.inference_time_s,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "total_tokens": r.total_tokens,
    }


@dataclass(kw_only=True)
class DripperHTMLLayoutTemplateStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name: str = "DripperHTMLLayoutTemplateStage"
    client: AsyncLLMClient | None
    model_name: str
    html_col: str = "html"
    url_col: str | None = "url"
    host_col: str | None = None
    layout_id_col: str | None = None
    generation_config: GenerationConfig | None = None
    structured_output_mode: Literal["none", "structured_outputs", "guided_regex"] = "none"
    max_concurrent_requests: int = 64
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    output_format: str = "mm_md"
    keep_intermediate: bool = False
    layout_cluster_threshold: float = 0.95
    layout_template_min_cluster_size: int = 2
    layout_template_fallback_llm: bool = True
    layout_template_require_success: bool = True
    layout_template_max_selected_item_ratio: float | None = 0.50
    layout_template_more_noise_enable: bool = True
    layout_template_validation_rows: int = 0
    layout_template_validation_min_content_f1: float = 0.98
    layout_template_large_cluster_validation_rows: int = 0
    layout_template_large_cluster_min_size: int = 0
    layout_template_propagation_target: Literal["raw_html", "mapped_item_ids"] = "raw_html"
    layout_template_min_main_html_sim: float | None = None
    layout_template_min_content_length_ratio: float | None = None
    layout_template_max_content_length_ratio: float | None = None
    dynamic_classid_similarity_threshold: float = 0.85
    layout_host_single_cluster_min_pages: int = 0
    layout_host_single_cluster_max_pages: int = 0
    layout_max_exact_host_pages: int = 0
    layout_large_host_mode: Literal["standalone", "feature_hash", "dom_path_hash"] = "standalone"
    layout_propagation_concurrency: int = 32
    layout_representative_candidates: int = 1
    layout_defer_fallback_llm: bool = False
    layout_defer_propagation: bool = False
    layout_failed_host_fallback_signature_mode: str = "none"
    layout_failed_layout_fallback_signature_mode: str = "none"
    layout_page_signature_mode: str = "none"
    layout_validation_signature_mode: str = "none"
    health_check: bool = False
    worker_count: int | None = None

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _web_bindings: _LLMWebKitBindings | None = field(init=False, repr=False, default=None)
    _fallback_handler: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    @property
    def _planning_cfg(self) -> _LayoutPlanningConfig:
        adv = DripperLayoutAdvancedConfig(host_single_cluster_min_pages=self.layout_host_single_cluster_min_pages, host_single_cluster_max_pages=self.layout_host_single_cluster_max_pages, max_exact_host_pages=self.layout_max_exact_host_pages, large_host_mode=self.layout_large_host_mode, propagation_concurrency=self.layout_propagation_concurrency, representative_candidates=self.layout_representative_candidates, defer_fallback_llm=self.layout_defer_fallback_llm, defer_propagation=self.layout_defer_propagation, failed_host_fallback_signature_mode=self.layout_failed_host_fallback_signature_mode, failed_layout_fallback_signature_mode=self.layout_failed_layout_fallback_signature_mode, page_signature_mode=self.layout_page_signature_mode, validation_signature_mode=self.layout_validation_signature_mode)  # fmt: skip
        return _LayoutPlanningConfig(html_col=self.html_col, url_col=self.url_col, host_col=self.host_col, layout_id_col=self.layout_id_col, layout_cluster_threshold=self.layout_cluster_threshold, min_cluster_size=self.layout_template_min_cluster_size, adv=adv, web_bindings=self._web_bindings)  # fmt: skip

    def __post_init__(self) -> None:
        def _req(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)

        def _enum(val: object, valid: set, name: str) -> None:
            if val not in valid:
                msg = f"{name} must be one of {sorted(valid)}"
                raise ValueError(msg)

        _req(self.client is not None, "DripperHTMLLayoutTemplateStage requires a non-None 'client' (AsyncLLMClient)")
        self.model_name = self.model_name.strip()
        _req(bool(self.model_name), "DripperHTMLLayoutTemplateStage requires a non-empty 'model_name'")
        _req(self.max_concurrent_requests > 0, "max_concurrent_requests must be positive")
        min_r = self.layout_template_min_content_length_ratio
        max_r = self.layout_template_max_content_length_ratio
        _req(0.0 < self.layout_cluster_threshold <= 1.0, "layout_cluster_threshold must be in (0, 1]")
        _req(self.layout_template_min_cluster_size > 1, "layout_template_min_cluster_size must be greater than 1")
        _max_sir = self.layout_template_max_selected_item_ratio
        _req(_max_sir is None or 0.0 < _max_sir <= 1.0, "layout_template_max_selected_item_ratio must be in (0, 1] when set")  # fmt: skip
        _req(self.layout_representative_candidates > 0, "advanced.representative_candidates must be positive")
        _min_sim = self.layout_template_min_main_html_sim
        _req(_min_sim is None or 0.0 <= _min_sim <= 1.0, "layout_template_min_main_html_sim must be in [0, 1] when set")  # fmt: skip
        _f1 = self.layout_template_validation_min_content_f1
        _req(0.0 <= _f1 <= 1.0, "layout_template_validation_min_content_f1 must be in [0, 1]")
        _req(self.dynamic_classid_similarity_threshold > 0, "dynamic_classid_similarity_threshold must be positive")
        _req(self.layout_template_validation_rows >= 0, "layout_template_validation_rows must be non-negative")
        _lcvr = self.layout_template_large_cluster_validation_rows
        _req(_lcvr >= 0, "layout_template_large_cluster_validation_rows must be non-negative")
        _lcms = self.layout_template_large_cluster_min_size
        _req(_lcms >= 0, "layout_template_large_cluster_min_size must be non-negative")
        _req(min_r is None or min_r >= 0, "layout_template_min_content_length_ratio must be non-negative when set")
        _req(max_r is None or max_r >= 0, "layout_template_max_content_length_ratio must be non-negative when set")
        _req(min_r is None or max_r is None or min_r <= max_r, "layout_template_min_content_length_ratio must be <= layout_template_max_content_length_ratio")  # fmt: skip
        _enum(self.layout_template_propagation_target, _LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES, "layout_template_propagation_target")  # fmt: skip
        for _val, _name in [
            (self.layout_validation_signature_mode, "advanced.validation_signature_mode"),
            (self.layout_page_signature_mode, "advanced.page_signature_mode"),
            (self.layout_failed_host_fallback_signature_mode, "advanced.failed_host_fallback_signature_mode"),
            (self.layout_failed_layout_fallback_signature_mode, "advanced.failed_layout_fallback_signature_mode"),
        ]:
            _enum(_val, _LAYOUT_PAGE_SIGNATURE_MODES, _name)
        _enum(self.layout_large_host_mode, _LAYOUT_TEMPLATE_LARGE_HOST_MODES, "advanced.large_host_mode")
        _enum(self.structured_output_mode, _STRUCTURED_OUTPUT_MODES, "structured_output_mode")
        _min_p, _max_p = self.layout_host_single_cluster_min_pages, self.layout_host_single_cluster_max_pages
        _req(_min_p >= 0, "advanced.host_single_cluster_min_pages must be non-negative")
        _req(_max_p >= 0, "advanced.host_single_cluster_max_pages must be non-negative")
        _req(_max_p == 0 or _min_p <= _max_p, "advanced.host_single_cluster_min_pages must be <= max_pages when max is set")  # fmt: skip
        _req(self.layout_max_exact_host_pages >= 0, "advanced.max_exact_host_pages must be non-negative")
        _req(self.layout_propagation_concurrency > 0, "advanced.propagation_concurrency must be positive")
        _req(self.worker_count is None or self.worker_count > 0, "worker_count must be positive when set")

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.html_col,
            _DRIPPER_RAW_RESPONSE_COL,
            _DRIPPER_PREPROCESS_TIME_COL,
            _DRIPPER_WARNING_COL,
            _DRIPPER_ITEM_COUNT_COL,
            _DRIPPER_REQUEST_MAX_TOKENS_COL,
            _DRIPPER_SIMPLIFIED_HTML_COL,
            _DRIPPER_MAPPED_HTML_COL,
            _DRIPPER_PROMPT_COL,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = [
            _DRIPPER_OUTPUT_HTML_COL,
            _DRIPPER_OUTPUT_CONTENT_COL,
            _DRIPPER_RAW_RESPONSE_COL,
            _DRIPPER_INFERENCE_TIME_COL,
            _DRIPPER_POSTPROCESS_TIME_COL,
            _DRIPPER_TOTAL_TIME_COL,
            _DRIPPER_ERROR_COL,
            _DRIPPER_WARNING_COL,
            _DRIPPER_PROMPT_TOKENS_COL,
            _DRIPPER_COMPLETION_TOKENS_COL,
            _DRIPPER_TOTAL_TOKENS_COL,
            "dripper_layout_cluster",
            "dripper_layout_representative",
            "dripper_layout_propagated",
            "dripper_layout_propagation_success",
            "dripper_layout_fallback_llm",
            "dripper_layout_standalone_llm",
            _DRIPPER_LAYOUT_FINALIZED_COL,
        ]
        if self.layout_defer_propagation:
            columns.extend(["dripper_layout_pending_propagation", "dripper_layout_mapping_json"])
        if self.layout_defer_fallback_llm:
            columns += [_DRIPPER_SIMPLIFIED_HTML_COL, _DRIPPER_MAPPED_HTML_COL, _DRIPPER_PROMPT_COL, _DRIPPER_NEEDS_LLM_COL, _DRIPPER_PRIMARY_ERROR_COL, _DRIPPER_EMPTY_INPUT_COL]  # fmt: skip
        if self.keep_intermediate and not self.layout_defer_fallback_llm:
            columns.extend([_DRIPPER_SIMPLIFIED_HTML_COL, _DRIPPER_MAPPED_HTML_COL])
        return ["data"], columns

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._web_bindings = _load_llm_web_kit_bindings()
        self._fallback_handler = self._bindings.get_fallback_handler(self.fallback)
        self.client.setup()  # type: ignore[union-attr]
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
        results = run_async_safe(lambda: self._process_all_async(df))
        preprocess_times = _numeric_series_or_zero(df, _DRIPPER_PREPROCESS_TIME_COL)
        inference_times = pd.Series([r.inference_time_s for r in results], index=df.index)
        postprocess_times = pd.Series([r.postprocess_time_s for r in results], index=df.index)
        for _col, _attr in [
            (_DRIPPER_OUTPUT_HTML_COL, "main_html"),
            (_DRIPPER_OUTPUT_CONTENT_COL, "main_content"),
            (_DRIPPER_RAW_RESPONSE_COL, "raw_response"),
            (_DRIPPER_ERROR_COL, "error"),
            (_DRIPPER_PROMPT_TOKENS_COL, "prompt_tokens"),
            (_DRIPPER_COMPLETION_TOKENS_COL, "completion_tokens"),
            (_DRIPPER_TOTAL_TOKENS_COL, "total_tokens"),
            ("dripper_layout_cluster", "layout_cluster"),
            ("dripper_layout_representative", "layout_representative"),
            ("dripper_layout_propagated", "layout_propagated"),
            ("dripper_layout_propagation_success", "layout_propagation_success"),
            ("dripper_layout_fallback_llm", "layout_fallback_llm"),
            ("dripper_layout_standalone_llm", "layout_standalone_llm"),
            (_DRIPPER_LAYOUT_FINALIZED_COL, "layout_finalized"),
        ]:
            df[_col] = [getattr(r, _attr) for r in results]
        df[_DRIPPER_INFERENCE_TIME_COL] = inference_times
        df[_DRIPPER_POSTPROCESS_TIME_COL] = postprocess_times
        df[_DRIPPER_TOTAL_TIME_COL] = preprocess_times + inference_times + postprocess_times
        _existing_w = df.get(_DRIPPER_WARNING_COL, pd.Series([""] * len(df))).tolist()
        df[_DRIPPER_WARNING_COL] = [_append_warning(str(e or ""), r.warning) for e, r in zip(_existing_w, results, strict=True)]  # fmt: skip
        if self.layout_defer_propagation:
            df["dripper_layout_pending_propagation"] = [r.layout_pending_propagation for r in results]
            df["dripper_layout_mapping_json"] = [r.layout_mapping_json for r in results]
        if self.layout_defer_fallback_llm:
            existing_primary_errors = df[_DRIPPER_PRIMARY_ERROR_COL].astype(str).tolist()
            df[_DRIPPER_NEEDS_LLM_COL] = [r.deferred_llm for r in results]
            df[_DRIPPER_PRIMARY_ERROR_COL] = [_append_warning(e, r.primary_error) for e, r in zip(existing_primary_errors, results, strict=True)]  # fmt: skip
        _base = [_DRIPPER_PROMPT_COL, _DRIPPER_NEEDS_LLM_COL, _DRIPPER_PRIMARY_ERROR_COL, _DRIPPER_EMPTY_INPUT_COL]
        drop_cols = [] if self.layout_defer_fallback_llm else [*_base, _DRIPPER_LAYOUT_FINALIZED_COL]
        if not self.keep_intermediate and not self.layout_defer_fallback_llm:
            drop_cols.extend([_DRIPPER_SIMPLIFIED_HTML_COL, _DRIPPER_MAPPED_HTML_COL])
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        _ma = [("layout_template_representative_rows", "layout_representative"), ("layout_template_propagated_rows", "layout_propagated"), ("layout_template_success_rows", "layout_propagation_success"), ("layout_template_fallback_llm_rows", "layout_fallback_llm"), ("layout_template_standalone_llm_rows", "layout_standalone_llm"), ("layout_template_deferred_llm_rows", "deferred_llm"), ("layout_template_finalized_rows", "layout_finalized")]  # fmt: skip
        self._log_metrics({"layout_template_rows": float(len(df))} | {k: float(sum(getattr(r, a) for r in results)) for k, a in _ma})  # fmt: skip
        return _rebuild_batch(batch, df)

    async def _process_all_async(self, df: pd.DataFrame) -> list[_LayoutTemplateRowResult]:
        propagation_semaphore = asyncio.Semaphore(
            min(self.max_concurrent_requests, self.layout_propagation_concurrency)
        )
        ctx = _LayoutProcessContext(df=df, semaphore=asyncio.Semaphore(self.max_concurrent_requests), propagation_semaphore=propagation_semaphore, inference_cache={}, inference_cache_lock=asyncio.Lock(), needs_llm=df[_DRIPPER_NEEDS_LLM_COL].astype(bool).tolist())  # fmt: skip
        layout_plans = _build_layout_group_plans(self._planning_cfg, df)
        grouped_indexes = {idx for plan in layout_plans for idx in plan.indexes}

        async def _handle_plan(plan_index: int, plan: _LayoutGroupPlan) -> dict[int, _LayoutTemplateRowResult]:
            return await self._handle_group_attempt_async(
                ctx,
                plan.indexes,
                f"layout-{plan_index:06d}",
                plan.host_key,
                plan.fallback_groups,
                split_failed_host_fallback=True,
            )

        tasks: list[Any] = [_handle_plan(plan_index, plan) for plan_index, plan in enumerate(layout_plans)]
        tasks.extend(self._handle_standalone_async(ctx, idx) for idx in range(len(df)) if idx not in grouped_indexes)
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results_by_index: dict[int, _LayoutTemplateRowResult] = {}
        for raw_result in raw_results:
            if isinstance(raw_result, BaseException):
                logger.error("Dripper layout-template task failed: {}", raw_result)
                continue
            if isinstance(raw_result, tuple):
                idx, result = raw_result
                results_by_index[idx] = result
            else:
                results_by_index.update(raw_result)
        _no_result_err = "layout template task produced no result"
        return [results_by_index[idx] if idx in results_by_index else (self._defer_row(df.iloc[idx], primary_error=_no_result_err, layout_fallback_llm=True) if self.layout_defer_fallback_llm else self._fallback_row(df.iloc[idx], primary_error=_no_result_err)) for idx in range(len(df))]  # fmt: skip

    async def _handle_standalone_async(
        self, ctx: _LayoutProcessContext, idx: int
    ) -> tuple[int, _LayoutTemplateRowResult]:
        if self.layout_defer_fallback_llm:
            return idx, self._defer_row(
                ctx.df.iloc[idx],
                layout_standalone_llm=ctx.needs_llm[idx],
                primary_error="layout template standalone row",
            )
        if ctx.needs_llm[idx]:
            result = await self._infer_and_postprocess_row(ctx.df.iloc[idx], semaphore=ctx.semaphore, cache=ctx.inference_cache, cache_lock=ctx.inference_cache_lock, layout_standalone_llm=True)  # fmt: skip
        else:
            result = self._fallback_row(ctx.df.iloc[idx])
        return idx, result

    async def _handle_group_attempt_async(  # noqa: PLR0913
        self,
        ctx: _LayoutProcessContext,
        indexes: list[int],
        cluster_id: str,
        host_key: str,
        fallback_groups: tuple[list[int], ...],
        *,
        split_failed_host_fallback: bool,
    ) -> dict[int, _LayoutTemplateRowResult]:
        outcome = await self._process_layout_group_with_status(
            ctx,
            indexes,
            cluster_id,
            emit_failure_fallback=not fallback_groups,
        )
        if outcome.accepted or not fallback_groups:
            return outcome.results
        child_groups = list(fallback_groups)
        if split_failed_host_fallback and self.layout_failed_host_fallback_signature_mode != "none":
            child_groups = _split_fallback_groups_by_signature(
                self._planning_cfg, ctx.df, child_groups, self.layout_failed_host_fallback_signature_mode
            )
        fallback_results: dict[int, _LayoutTemplateRowResult] = {}
        fallback_grouped_indexes: set[int] = set()
        fallback_tasks = [self._handle_group_attempt_async(ctx, fallback_indexes, f"{cluster_id}-fallback-{fallback_index:06d}", host_key, tuple(_build_failed_layout_fallback_groups(self._planning_cfg, ctx.df, fallback_indexes)), split_failed_host_fallback=False) for fallback_index, fallback_indexes in enumerate(child_groups)]  # fmt: skip
        if fallback_tasks:
            [fallback_results.update(gr) for gr in await asyncio.gather(*fallback_tasks)]
            fallback_grouped_indexes = {idx for group in child_groups for idx in group}
        standalone_tasks = [self._handle_standalone_async(ctx, idx) for idx in indexes if idx not in fallback_grouped_indexes]  # fmt: skip
        if standalone_tasks:
            fallback_results.update(dict(await asyncio.gather(*standalone_tasks)))
        return fallback_results

    async def _process_layout_group_with_status(
        self,
        ctx: _LayoutProcessContext,
        indexes: list[int],
        cluster_id: str,
        *,
        emit_failure_fallback: bool,
    ) -> _LayoutGroupOutcome:
        df = ctx.df
        representative_idx, mapping_data, results, mapping_failures = await self._infer_representative_candidates(
            ctx, indexes, cluster_id
        )
        if mapping_data is None:
            return await self._handle_mapping_failure(ctx, indexes, cluster_id, results, mapping_failures, emit_failure_fallback)  # fmt: skip
        if representative_idx is None:
            msg = "representative_idx must not be None"
            raise RuntimeError(msg)
        sibling_indexes = [idx for idx in indexes if idx not in results]
        validation_rows = self.layout_template_validation_rows
        if (
            self.layout_template_large_cluster_validation_rows > 0
            and self.layout_template_large_cluster_min_size > 0
            and len(indexes) >= self.layout_template_large_cluster_min_size
        ):
            validation_rows = max(validation_rows, self.layout_template_large_cluster_validation_rows)
        validation_indexes = _select_validation_indexes(df, sibling_indexes, validation_rows, (self.url_col, _DRIPPER_ITEM_COUNT_COL), signature_mode=self.layout_validation_signature_mode)  # fmt: skip
        remaining_indexes = [idx for idx in sibling_indexes if idx not in set(validation_indexes)]
        validation_failed, validation_error = False, ""
        if validation_indexes:
            validation_failed, validation_error = await self._run_validation_rows_async(
                ctx, validation_indexes, mapping_data, cluster_id, results
            )
            if validation_failed:
                logger.debug("Dripper layout validation failed for {}: {}", cluster_id, validation_error)
                if not emit_failure_fallback:
                    return _LayoutGroupOutcome(results=results, accepted=False, failure_reason=validation_error)
        sibling_outcome = await self._propagate_sibling_rows_async(ctx, remaining_indexes, mapping_data, cluster_id, results, validation_failed, validation_error)  # fmt: skip
        if sibling_outcome is not None:
            return sibling_outcome
        return _LayoutGroupOutcome(results=results)

    async def _handle_mapping_failure(  # noqa: PLR0913
        self,
        ctx: _LayoutProcessContext,
        indexes: list[int],
        cluster_id: str,
        results: dict[int, _LayoutTemplateRowResult],
        mapping_failures: list[str],
        emit_failure_fallback: bool,
    ) -> _LayoutGroupOutcome:
        df = ctx.df
        warning = "layout template mapping failed"
        if mapping_failures:
            warning = f"{warning}: {'; '.join(mapping_failures[:3])}"
        if not emit_failure_fallback:
            return _LayoutGroupOutcome(results=results, accepted=False, failure_reason=warning)
        fallback_indexes = [idx for idx in indexes if idx not in results]
        if self.layout_defer_fallback_llm:
            for idx in fallback_indexes:
                results[idx] = self._defer_row(df.iloc[idx], primary_error=warning, layout_cluster=cluster_id, layout_fallback_llm=True)  # fmt: skip
        elif self.layout_template_fallback_llm:
            _fbs = [self._infer_and_postprocess_row(df.iloc[idx], semaphore=ctx.semaphore, cache=ctx.inference_cache, cache_lock=ctx.inference_cache_lock, layout_cluster=cluster_id, layout_fallback_llm=True, primary_error=warning) for idx in fallback_indexes]  # fmt: skip
            results.update(zip(fallback_indexes, await asyncio.gather(*_fbs), strict=True))
        else:
            for idx in fallback_indexes:
                results[idx] = replace(
                    self._fallback_row(df.iloc[idx], primary_error=warning), layout_cluster=cluster_id
                )
        return _LayoutGroupOutcome(results=results, accepted=False, failure_reason=warning)

    async def _run_validation_rows_async(
        self,
        ctx: _LayoutProcessContext,
        validation_indexes: list[int],
        mapping_data: dict[str, Any],
        cluster_id: str,
        results: dict[int, _LayoutTemplateRowResult],
    ) -> tuple[bool, str]:
        _prop_coros = (self._propagate_layout_template_async(ctx.df.iloc[i], mapping_data, cluster_id, ctx.propagation_semaphore) for i in validation_indexes)  # fmt: skip
        _llm_coros = (self._infer_and_postprocess_row(ctx.df.iloc[i], semaphore=ctx.semaphore, cache=ctx.inference_cache, cache_lock=ctx.inference_cache_lock, layout_cluster=cluster_id, layout_fallback_llm=True, primary_error="layout template validation LLM") for i in validation_indexes)  # fmt: skip
        validation_propagated, validation_llm_results = await asyncio.gather(asyncio.gather(*_prop_coros), asyncio.gather(*_llm_coros))  # fmt: skip
        failed, error = False, ""
        for idx, propagated, llm_result in zip(
            validation_indexes, validation_propagated, validation_llm_results, strict=True
        ):
            results[idx] = llm_result
            content_f1 = _token_f1(propagated.main_content, llm_result.main_content)
            failure_reasons = []
            if propagated.error:
                failure_reasons.append(f"propagation_error={propagated.error[:160]}")
            if content_f1 < self.layout_template_validation_min_content_f1:
                failure_reasons.append(f"content_f1={content_f1:.3f}")
            if failure_reasons:
                failed = True
                error = f"layout template validation failed: {' '.join(failure_reasons)} min={self.layout_template_validation_min_content_f1:.3f}"
        return failed, error

    async def _propagate_sibling_rows_async(  # noqa: PLR0913
        self,
        ctx: _LayoutProcessContext,
        remaining_indexes: list[int],
        mapping_data: dict[str, Any],
        cluster_id: str,
        results: dict[int, _LayoutTemplateRowResult],
        validation_failed: bool,
        validation_error: str,
    ) -> _LayoutGroupOutcome | None:
        df = ctx.df
        propagated_results: list[_LayoutTemplateRowResult] = []
        if remaining_indexes and not validation_failed:
            if self.layout_defer_propagation:
                for idx in remaining_indexes:
                    results[idx] = _LayoutTemplateRowResult(
                        layout_cluster=cluster_id, layout_pending_propagation=True, layout_finalized=False
                    )
                return _LayoutGroupOutcome(results=results)
            propagated_results = list(await asyncio.gather(*(self._propagate_layout_template_async(df.iloc[idx], mapping_data, cluster_id, ctx.propagation_semaphore) for idx in remaining_indexes)))  # fmt: skip
        fallback_tasks: list[Any] = []
        fallback_indexes: list[int] = []
        for i, idx in enumerate(remaining_indexes):
            error = (
                validation_error
                if validation_failed
                else (propagated_results[i].error if not validation_failed else "")
            )
            propagated = None if validation_failed else propagated_results[i]
            if validation_failed or (propagated is not None and propagated.error):
                if self.layout_defer_fallback_llm:
                    results[idx] = self._defer_row(df.iloc[idx], primary_error=error, layout_cluster=cluster_id, layout_fallback_llm=True)  # fmt: skip
                elif self.layout_template_fallback_llm:
                    fallback_indexes.append(idx)
                    fallback_tasks.append(self._infer_and_postprocess_row(df.iloc[idx], semaphore=ctx.semaphore, cache=ctx.inference_cache, cache_lock=ctx.inference_cache_lock, layout_cluster=cluster_id, layout_fallback_llm=True, primary_error=error))  # fmt: skip
                else:
                    results[idx] = replace(
                        self._fallback_row(df.iloc[idx], primary_error=error), layout_cluster=cluster_id
                    )
            elif propagated is not None:
                results[idx] = propagated
        if fallback_tasks:
            fallback_results_list = await asyncio.gather(*fallback_tasks)
            results.update(zip(fallback_indexes, fallback_results_list, strict=True))
        return None

    async def _infer_representative_candidates(
        self, ctx: _LayoutProcessContext, indexes: list[int], cluster_id: str
    ) -> tuple[int | None, dict[str, Any] | None, dict[int, _LayoutTemplateRowResult], list[str]]:
        df = ctx.df
        representative_indexes = self._select_representative_indexes(df, indexes)
        representative_idx: int | None = None
        mapping_data: dict[str, Any] | None = None
        candidate_results: dict[int, _LayoutTemplateRowResult] = {}
        mapping_failures: list[str] = []
        for candidate_idx in representative_indexes:
            candidate_result, candidate_mapping = await self._infer_representative_and_mapping(
                df.iloc[candidate_idx], ctx.semaphore, cluster_id, ctx.inference_cache, ctx.inference_cache_lock
            )
            candidate_results[candidate_idx] = candidate_result
            if candidate_mapping is not None:
                representative_idx = candidate_idx
                mapping_data = candidate_mapping
                break
            mapping_failures.append(f"{candidate_idx}:{candidate_result.primary_error or candidate_result.warning or 'mapping failed'}")  # fmt: skip
        results: dict[int, _LayoutTemplateRowResult] = {}
        mapping_json_for_representative = json.dumps(mapping_data, default=str) if self.layout_defer_propagation and mapping_data is not None else ""  # fmt: skip
        for candidate_idx, candidate_result in candidate_results.items():
            is_rep = candidate_idx == representative_idx
            results[candidate_idx] = replace(candidate_result, layout_cluster=cluster_id, layout_representative=is_rep, layout_fallback_llm=not is_rep, layout_mapping_json=mapping_json_for_representative if is_rep else "")  # fmt: skip
        return representative_idx, mapping_data, results, mapping_failures

    def _select_representative_indexes(self, df: pd.DataFrame, indexes: list[int]) -> list[int]:
        candidates = [{"track_id": str(idx), "html": _coerce_html(df.iloc[idx].get(self.html_col, ""))} for idx in indexes]  # fmt: skip
        try:
            rep = self._web_bindings.select_representative_html(candidates)
            selected = int(rep["track_id"]) if rep is not None else indexes[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dripper representative selection failed: {}", exc)
            selected = indexes[0]
        if selected not in indexes:
            selected = indexes[0]
        result = [selected]
        if self.layout_representative_candidates > 1:
            result.extend(_select_validation_indexes(df, [idx for idx in indexes if idx != selected], self.layout_representative_candidates - 1, (self.url_col, _DRIPPER_ITEM_COUNT_COL)))  # fmt: skip
        return result

    async def _infer_representative_and_mapping(
        self,
        row: pd.Series,
        semaphore: asyncio.Semaphore,
        cluster_id: str,
        inference_cache: _InferenceCache,
        inference_cache_lock: asyncio.Lock,
    ) -> tuple[_LayoutTemplateRowResult, dict[str, Any] | None]:
        inference_result = await self._infer_row_cached(row, semaphore, inference_cache, inference_cache_lock)
        started = time.perf_counter()

        def _make_fallback_result(primary_error: str, *, elapsed: float | None = None) -> _LayoutTemplateRowResult:
            fb = self._fallback_and_convert(row, primary_error=primary_error)
            return _LayoutTemplateRowResult(**_inference_token_fields(inference_result), main_html=fb.main_html, main_content=fb.main_content, postprocess_time_s=elapsed if elapsed is not None else fb.postprocess_time_s, error=fb.error, warning=fb.warning, primary_error=primary_error, layout_cluster=cluster_id)  # fmt: skip

        if inference_result.primary_error:
            return _make_fallback_result(_append_warning("", inference_result.primary_error)), None
        html_text = _coerce_html(row.get(self.html_col, ""))
        mapped_html = str(row.get(_DRIPPER_MAPPED_HTML_COL, "") or "")
        case = self._build_case(row)
        mapping_failure_reason = ""
        try:
            case.generate_output = self._bindings.generate_output_cls(response=inference_result.raw_response)
            case = self._bindings.parse_result(case)
            webkit_response = _labels_to_webkit_response(getattr(case.parse_result, "item_label", {}))
            case = self._bindings.extract_main_html_single(case)
            mapping_data = self._web_bindings.map_parser_cls({}).parse({"typical_raw_tag_html": mapped_html, "typical_raw_html": html_text, "llm_response": webkit_response})  # fmt: skip
            if self.layout_template_require_success and mapping_data.get("typical_main_html_success") is False:
                mapping_failure_reason = "typical_main_html_success=false"
                mapping_data = None
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper representative mapping failed: {}", primary_error)
            return _make_fallback_result(primary_error, elapsed=time.perf_counter() - started), None
        post_result = self._convert_case(case)
        warning = post_result.warning
        if mapping_data is None:
            primary_error = f"layout template mapping failed: {mapping_failure_reason or 'template unusable'}"
            warning = _append_warning(warning, primary_error)
        else:
            primary_error = ""
            mapping_data = dict(mapping_data)
            mapping_data["_dripper_representative_content_len"] = len(str(post_result.main_content or ""))
        return _LayoutTemplateRowResult(**_inference_token_fields(inference_result), main_html=post_result.main_html, main_content=post_result.main_content, postprocess_time_s=time.perf_counter() - started, error=post_result.error, warning=warning, primary_error=primary_error, layout_cluster=cluster_id), mapping_data  # fmt: skip

    async def _propagate_layout_template_async(
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
        cluster_id: str,
        semaphore: asyncio.Semaphore,
    ) -> _LayoutTemplateRowResult:
        async with semaphore:
            return await asyncio.to_thread(self._propagate_layout_template, row, mapping_data, cluster_id)

    def _propagate_layout_template(
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
        cluster_id: str,
    ) -> _LayoutTemplateRowResult:
        started = time.perf_counter()
        html_text = _coerce_html(row.get(self.html_col, ""))
        mapped_html = str(row.get(_DRIPPER_MAPPED_HTML_COL, "") or "")
        use_mapped_item_ids = (
            self.layout_template_propagation_target == "mapped_item_ids" and "_item_id" in mapped_html
        )
        html_source = mapped_html if use_mapped_item_ids else html_text
        try:
            task_data = dict(mapping_data) | {
                "html_source": html_source,
                "dynamic_id_enable": True,
                "dynamic_classid_enable": True,
                "more_noise_enable": self.layout_template_more_noise_enable,
                "dynamic_classid_similarity_threshold": self.dynamic_classid_similarity_threshold,
            }
            parts = self._web_bindings.layout_parser_cls({}).parse(task_data)
            if self.layout_template_require_success and parts.get("main_html_success") is False:
                raise RuntimeError(f"layout propagation similarity below threshold: {parts.get('main_html_sim')}")  # noqa: TRY301, EM102
            if self.layout_template_min_main_html_sim is not None:
                main_html_sim = _coerce_optional_float(parts.get("main_html_sim"))
                if main_html_sim is not None and main_html_sim < self.layout_template_min_main_html_sim:
                    msg = f"layout propagation main_html_sim {main_html_sim:.3f} below {self.layout_template_min_main_html_sim:.3f}"
                    raise RuntimeError(msg)  # noqa: TRY301
            main_html = str(parts.get("main_html_body") or "")
            raw_response = ""
            if use_mapped_item_ids:
                all_item_ids = _item_ids_in_html(mapped_html)
                main_item_ids = set(_item_ids_in_html(main_html))
                if not all_item_ids:
                    raise RuntimeError("layout propagation target mapped HTML has no item ids")  # noqa: TRY301, EM101
                if not main_item_ids:
                    raise RuntimeError("layout propagation produced no target item ids")  # noqa: TRY301, EM101
                selected_item_ratio = len(main_item_ids) / len(all_item_ids)
                if (
                    self.layout_template_max_selected_item_ratio is not None
                    and selected_item_ratio > self.layout_template_max_selected_item_ratio
                ):
                    msg = f"layout propagation selected item ratio {selected_item_ratio:.3f} exceeds {self.layout_template_max_selected_item_ratio:.3f}"
                    raise RuntimeError(msg)  # noqa: TRY301
                raw_response = _item_id_response(all_item_ids, main_item_ids)
                post_result = self._postprocess_raw_response(row, raw_response)
            else:
                _case = self._build_case(row)
                _case.output_data = self._bindings.output_cls(main_html=main_html)
                post_result = self._convert_case(_case)
            content_ratio_error = self._propagated_content_length_ratio_error(post_result.main_content, mapping_data)
            if content_ratio_error:
                raise RuntimeError(content_ratio_error)  # noqa: TRY301
            return _LayoutTemplateRowResult(raw_response=raw_response, main_html=post_result.main_html, main_content=post_result.main_content, postprocess_time_s=time.perf_counter() - started, error=post_result.error, warning=post_result.warning, layout_cluster=cluster_id, layout_propagated=True, layout_propagation_success=not bool(post_result.error))  # fmt: skip
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper layout propagation failed: {}", primary_error)
            fallback_result = self._fallback_and_convert(row, primary_error=primary_error)
            return _LayoutTemplateRowResult(main_html=fallback_result.main_html, main_content=fallback_result.main_content, postprocess_time_s=time.perf_counter() - started, error=fallback_result.error or primary_error, warning=fallback_result.warning, primary_error=primary_error, layout_cluster=cluster_id, layout_propagated=True)  # fmt: skip

    def _propagated_content_length_ratio_error(self, propagated_content: object, mapping_data: dict[str, Any]) -> str:
        min_r, max_r = self.layout_template_min_content_length_ratio, self.layout_template_max_content_length_ratio
        if min_r is None and max_r is None:
            return ""
        rep_len = _coerce_positive_int(mapping_data.get("_dripper_representative_content_len"))
        if rep_len <= 0:
            return ""
        ratio = len(str(propagated_content or "")) / rep_len
        if min_r is not None and ratio < min_r:
            return f"layout propagation content length ratio {ratio:.3f} below {min_r:.3f}"
        if max_r is not None and ratio > max_r:
            return f"layout propagation content length ratio {ratio:.3f} exceeds {max_r:.3f}"
        return ""

    async def _infer_and_postprocess_row(  # noqa: PLR0913
        self,
        row: pd.Series,
        *,
        semaphore: asyncio.Semaphore | None = None,
        cache: _InferenceCache | None = None,
        cache_lock: asyncio.Lock | None = None,
        layout_cluster: str = "",
        layout_fallback_llm: bool = False,
        layout_standalone_llm: bool = False,
        primary_error: str = "",
    ) -> _LayoutTemplateRowResult:
        if cache is None or cache_lock is None:
            prompt = str(row.get(_DRIPPER_PROMPT_COL, "") or "")
            row_max_tokens = _coerce_usage_int(row.get(_DRIPPER_REQUEST_MAX_TOKENS_COL, 0))
            inference_result = await self._infer_prompt(prompt, row_max_tokens, semaphore)
        else:
            inference_result = await self._infer_row_cached(row, semaphore, cache, cache_lock)
        if inference_result.primary_error:
            merged_primary = _append_warning(primary_error, inference_result.primary_error)
            fb = self._fallback_and_convert(row, primary_error=merged_primary)
            return _LayoutTemplateRowResult(**_inference_token_fields(inference_result), main_html=fb.main_html, main_content=fb.main_content, postprocess_time_s=fb.postprocess_time_s, error=fb.error, warning=fb.warning, primary_error=merged_primary, layout_cluster=layout_cluster, layout_fallback_llm=layout_fallback_llm, layout_standalone_llm=layout_standalone_llm)  # fmt: skip
        post_result = self._postprocess_raw_response(row, inference_result.raw_response)
        return _LayoutTemplateRowResult(**_inference_token_fields(inference_result), main_html=post_result.main_html, main_content=post_result.main_content, postprocess_time_s=post_result.postprocess_time_s, error=post_result.error, warning=_append_warning(primary_error, post_result.warning), layout_cluster=layout_cluster, layout_fallback_llm=layout_fallback_llm, layout_standalone_llm=layout_standalone_llm)  # fmt: skip

    async def _infer_row_cached(
        self,
        row: pd.Series,
        semaphore: asyncio.Semaphore,
        inference_cache: _InferenceCache,
        inference_cache_lock: asyncio.Lock,
    ) -> _DripperInferenceResult:
        prompt = str(row.get(_DRIPPER_PROMPT_COL, "") or "")
        row_max_tokens = _coerce_usage_int(row.get(_DRIPPER_REQUEST_MAX_TOKENS_COL, 0))
        if not prompt.strip():
            return _DripperInferenceResult(primary_error="empty Dripper prompt", warning="empty Dripper prompt")
        key = (prompt, row_max_tokens)
        async with inference_cache_lock:
            task = inference_cache.get(key)
            owns_request = task is None
            if task is None:
                task = asyncio.create_task(self._infer_prompt(prompt, row_max_tokens, semaphore))
                inference_cache[key] = task
        result = await task
        if owns_request:
            return result
        return replace(result, inference_time_s=0.0, prompt_tokens=0, completion_tokens=0, total_tokens=0)

    async def _infer_prompt(
        self,
        prompt: str,
        row_max_tokens: int,
        semaphore: asyncio.Semaphore,
    ) -> _DripperInferenceResult:
        if not prompt.strip():
            return _DripperInferenceResult(primary_error="empty Dripper prompt", warning="empty Dripper prompt")
        async with semaphore:
            started = time.perf_counter()
            try:
                generation_config = self.generation_config or GenerationConfig()
                if row_max_tokens > 0 and generation_config.max_tokens != row_max_tokens:
                    generation_config = replace(generation_config, max_tokens=row_max_tokens)
                generation_config = _with_structured_output_config(generation_config, prompt, self.structured_output_mode)  # fmt: skip
                raw_response, prompt_tokens, completion_tokens, total_tokens = await _query_dripper_model(self.client, self.model_name, [{"role": "user", "content": prompt}], generation_config)  # fmt: skip
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                logger.debug("Dripper inference failed; postprocess stage will apply fallback: {}", error)
                return _DripperInferenceResult(inference_time_s=time.perf_counter() - started, primary_error=error, warning=error)  # fmt: skip
            return _DripperInferenceResult(raw_response=raw_response, inference_time_s=time.perf_counter() - started, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)  # fmt: skip

    def _postprocess_raw_response(self, row: pd.Series, raw_response: str) -> _DripperPostResult:
        started = time.perf_counter()
        case = self._build_case(row)
        try:
            case.generate_output = self._bindings.generate_output_cls(response=raw_response)
            case = self._bindings.parse_result(case)
            case = self._bindings.extract_main_html_single(case)
        except Exception as exc:  # noqa: BLE001
            pe = str(exc)
            logger.debug("Dripper parse/extract failed, applying {} fallback: {}", self.fallback, pe)
            return replace(
                self._fallback_and_convert(row, primary_error=pe), postprocess_time_s=time.perf_counter() - started
            )
        return replace(self._convert_case(case), postprocess_time_s=time.perf_counter() - started)

    def _fallback_row(self, row: pd.Series, *, primary_error: str = "") -> _LayoutTemplateRowResult:
        r = self._fallback_and_convert(row, primary_error=_append_warning(primary_error, str(row.get(_DRIPPER_PRIMARY_ERROR_COL, "") or "")))  # fmt: skip
        return _LayoutTemplateRowResult(main_html=r.main_html, main_content=r.main_content, postprocess_time_s=r.postprocess_time_s, error=r.error, warning=r.warning, primary_error=primary_error)  # fmt: skip

    def _defer_row(self, row: pd.Series, *, primary_error: str = "", layout_cluster: str = "", layout_fallback_llm: bool = False, layout_standalone_llm: bool = False) -> _LayoutTemplateRowResult:  # fmt: skip
        nlm = bool(row.get(_DRIPPER_NEEDS_LLM_COL, False))
        return _LayoutTemplateRowResult(raw_response=str(row.get(_DRIPPER_RAW_RESPONSE_COL, "") or ""), inference_time_s=float(row.get(_DRIPPER_INFERENCE_TIME_COL, 0.0) or 0.0), prompt_tokens=_coerce_usage_int(row.get(_DRIPPER_PROMPT_TOKENS_COL, 0)), completion_tokens=_coerce_usage_int(row.get(_DRIPPER_COMPLETION_TOKENS_COL, 0)), total_tokens=_coerce_usage_int(row.get(_DRIPPER_TOTAL_TOKENS_COL, 0)), error=str(row.get(_DRIPPER_ERROR_COL, "") or ""), warning=_append_warning(str(row.get(_DRIPPER_WARNING_COL, "") or ""), primary_error), primary_error=primary_error, deferred_llm=nlm, layout_finalized=False, layout_cluster=layout_cluster, layout_fallback_llm=layout_fallback_llm and nlm, layout_standalone_llm=layout_standalone_llm and nlm)  # fmt: skip

    def _build_case(self, row: pd.Series) -> object:
        html_text = _coerce_html(row.get(self.html_col, ""))
        url = _coerce_optional_str(row.get(self.url_col) if self.url_col else None)
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html_text, url=url))
        simplified_html = str(row.get(_DRIPPER_SIMPLIFIED_HTML_COL, "") or "")
        mapped_html = str(row.get(_DRIPPER_MAPPED_HTML_COL, "") or "")
        if simplified_html or mapped_html:
            case.process_data = self._bindings.process_data_cls(simpled_html=simplified_html, map_html=mapped_html)
        return case

    def _fallback_and_convert(self, row: pd.Series, *, primary_error: str = "") -> _DripperPostResult:
        started = time.perf_counter()
        case = self._build_case(row)
        if bool(row.get(_DRIPPER_EMPTY_INPUT_COL, False)) or not _coerce_html(row.get(self.html_col, "")).strip():
            return _DripperPostResult(postprocess_time_s=time.perf_counter() - started, warning=_append_warning(primary_error, "empty HTML input"))  # fmt: skip
        fallback_result = _apply_fallback_extraction(self._bindings, self._fallback_handler, case, primary_error)
        case = fallback_result[0]
        if fallback_result[2]:
            return _DripperPostResult(postprocess_time_s=time.perf_counter() - started, error=fallback_result[2], warning=fallback_result[1])  # fmt: skip
        result = self._convert_case(case, warning=fallback_result[1])
        return replace(result, postprocess_time_s=time.perf_counter() - started)

    def _convert_case(self, case: object, *, warning: str = "") -> _DripperPostResult:
        conversion_error = ""
        try:
            _sanitize_case_output_html(case)
            case = self._bindings.convert2content(case, output_format=self.output_format)
        except (TypeError, AttributeError, ValueError, RuntimeError) as exc:
            conversion_error = str(exc)
            logger.debug("Dripper content conversion failed: {}", conversion_error)
        output_data = getattr(case, "output_data", None)
        main_html = getattr(output_data, "main_html", "") if output_data is not None else ""
        main_content = getattr(output_data, "main_content", "") if output_data is not None else ""
        main_content = "" if main_content is None else main_content
        error = ""
        if conversion_error:
            if _is_empty_document_error(conversion_error) and not str(main_html).strip():
                warning = _append_warning(warning, conversion_error)
            else:
                error = conversion_error
        return _DripperPostResult(main_html=main_html, main_content=main_content, error=error, warning=warning)
