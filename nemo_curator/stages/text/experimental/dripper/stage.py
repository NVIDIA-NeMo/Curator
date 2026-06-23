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


from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.models.client.llm_client import AsyncLLMClient


@dataclass(frozen=True)
class _MinerUHTMLBindings:
    input_cls: type
    case_cls: type
    output_cls: type
    process_data_cls: type
    generate_output_cls: type
    simplify_single_input: Callable[[Any], Any]
    build_prompt: Callable[..., Any]
    parse_result: Callable[[Any], Any]
    extract_main_html_single: Callable[[Any], Any]
    extract_main_html_fallback: Callable[..., Any]
    convert2content: Callable[..., Any]
    get_fallback_handler: Callable[[str], Any]


def _always_similar(_left: object, _right: object, _max_layer_n: int) -> float:
    return 1.0


@dataclass(frozen=True)
class _LLMWebKitBindings:
    get_feature: Callable[[str], Any]
    cluster_html_struct: Callable[..., Any]
    select_representative_html: Callable[[list[dict[str, str]]], dict[str, str] | None]
    map_parser_cls: type
    layout_parser_cls: type
    similarity: Callable[..., float] = _always_similar


@dataclass(frozen=True)
class _DripperRowResult:
    main_html: str = ""
    main_content: Any = ""
    raw_response: str = ""
    preprocess_time_s: float = 0.0
    inference_time_s: float = 0.0
    postprocess_time_s: float = 0.0
    total_time_s: float = 0.0
    error: str = ""
    warning: str = ""
    simplified_html: str = ""
    mapped_html: str = ""
    item_count: int = 0
    prompt_chars: int = 0
    request_max_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class _DripperInferenceResult:
    raw_response: str = ""
    inference_time_s: float = 0.0
    primary_error: str = ""
    warning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class _DripperPostResult:
    main_html: str = ""
    main_content: Any = ""
    postprocess_time_s: float = 0.0
    error: str = ""
    warning: str = ""


@dataclass(frozen=True)
class _DripperPrepResult:
    empty_input: bool = False
    needs_llm: bool = False
    preprocess_time_s: float = 0.0
    warning: str = ""
    primary_error: str = ""
    simplified_html: str = ""
    mapped_html: str = ""
    item_count: int = 0
    prompt: str = ""
    prompt_chars: int = 0
    request_max_tokens: int = 0


_DRIPPER_PROMPT_COL = "_dripper_prompt"
_DRIPPER_NEEDS_LLM_COL = "_dripper_needs_llm"
_DRIPPER_PRIMARY_ERROR_COL = "_dripper_primary_error"
_DRIPPER_EMPTY_INPUT_COL = "_dripper_empty_input"
_DRIPPER_LAYOUT_FINALIZED_COL = "_dripper_layout_finalized"


def _load_mineru_html_bindings() -> _MinerUHTMLBindings:
    from mineru_html.base import (
        MinerUHTMLCase,
        MinerUHTMLGenerateOutput,
        MinerUHTMLInput,
        MinerUHTMLOutput,
        MinerUHTMLProcessData,
    )
    from mineru_html.process import (
        build_prompt,
        convert2content,
        extract_main_html_fallback,
        extract_main_html_single,
        get_fallback_handler,
        parse_result,
        simplify_single_input,
    )

    return _MinerUHTMLBindings(
        input_cls=MinerUHTMLInput,
        case_cls=MinerUHTMLCase,
        output_cls=MinerUHTMLOutput,
        process_data_cls=MinerUHTMLProcessData,
        generate_output_cls=MinerUHTMLGenerateOutput,
        simplify_single_input=simplify_single_input,
        build_prompt=build_prompt,
        parse_result=parse_result,
        extract_main_html_single=extract_main_html_single,
        extract_main_html_fallback=extract_main_html_fallback,
        convert2content=convert2content,
        get_fallback_handler=get_fallback_handler,
    )


def _load_llm_web_kit_bindings() -> _LLMWebKitBindings:
    from llm_web_kit.html_layout.html_layout_cosin import get_feature, similarity
    from llm_web_kit.main_html_parser.parser.layout_batch_parser import LayoutBatchParser
    from llm_web_kit.main_html_parser.parser.tag_mapping import MapItemToHtmlTagsParser
    from llm_web_kit.main_html_parser.typical_html.typical_html import select_representative_html

    # Use GPU-accelerated DBSCAN when available (cuML + cupy), falls back to sklearn
    from nemo_curator.stages.text.experimental.dripper.gpu_layout_clustering import (
        cluster_html_struct_gpu,
    )

    return _LLMWebKitBindings(
        get_feature=get_feature,
        cluster_html_struct=cluster_html_struct_gpu,
        select_representative_html=select_representative_html,
        map_parser_cls=MapItemToHtmlTagsParser,
        layout_parser_cls=LayoutBatchParser,
        similarity=similarity,
    )


async def _run_dripper_health_check(
    client: AsyncLLMClient,
    model_name: str,
    generation_config: GenerationConfig | None,
) -> None:
    extra_kwargs = generation_config.extra_kwargs if generation_config is not None else None
    hc_config = GenerationConfig(max_tokens=8, temperature=0.0, top_p=1.0, extra_kwargs=extra_kwargs)
    try:
        response = await client.query_model(
            model=model_name,
            generation_config=hc_config,
            messages=[{"role": "user", "content": 'Return exactly: "1main"'}],
        )
    except RuntimeError:
        raise
    except Exception as exc:
        msg = f"Dripper LLM health check failed: {exc}. Ensure the inference server is reachable."
        raise RuntimeError(msg) from exc
    result = response[0] if response else ""
    if not result:
        msg = "Dripper LLM health check returned an empty response"
        raise RuntimeError(msg)
    logger.info("Dripper LLM health check passed")


async def _query_dripper_model(
    client: AsyncLLMClient,
    model_name: str,
    messages: list[dict[str, str]],
    generation_config: GenerationConfig,
) -> tuple[str, int, int, int]:
    query_model_with_usage = getattr(client, "query_model_with_usage", None)
    if callable(query_model_with_usage):
        response = await query_model_with_usage(
            model=model_name,
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

    response = await client.query_model(
        model=model_name,
        messages=messages,
        generation_config=generation_config,
    )
    return response[0] if response else "", 0, 0, 0


def _rebuild_batch(batch: DocumentBatch, df: pd.DataFrame) -> DocumentBatch:
    new_batch = DocumentBatch(
        dataset_name=batch.dataset_name,
        data=df,
        _metadata=batch._metadata,
        _stage_perf=batch._stage_perf,
    )
    new_batch.task_id = batch.task_id
    return new_batch


def _sanitize_case_output_html(case: object) -> None:
    output_data = getattr(case, "output_data", None)
    if output_data is None:
        return
    main_html = getattr(output_data, "main_html", None)
    if isinstance(main_html, str):
        output_data.main_html = _strip_xml_incompatible_chars(main_html)


def _get_processed_attr(case: object, attr: str) -> str:
    process_data = getattr(case, "process_data", None)
    value = getattr(process_data, attr, "") if process_data is not None else ""
    return value if isinstance(value, str) else ""


def _case_has_item_ids(case: object) -> bool:
    return "_item_id" in _get_processed_attr(case, "simpled_html") or "_item_id" in _get_processed_attr(
        case, "map_html"
    )


def _count_item_ids(case: object) -> int:
    html = _get_processed_attr(case, "simpled_html") or _get_processed_attr(case, "map_html")
    return len(set(_ITEM_ID_RE.findall(html)))


def _coerce_html(value: object) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, bytes | bytearray):
        raw_bytes = bytes(value)
        decoded: str | None = None
        try:
            decoded = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                from charset_normalizer import detect as _detect

                enc = _detect(raw_bytes)["encoding"]
                if enc and enc != "utf-8":
                    decoded = raw_bytes.decode(enc)
            except Exception:  # noqa: BLE001
                decoded = None
        if decoded is None:
            decoded = raw_bytes.decode("utf-8", errors="replace")
        return _strip_xml_incompatible_chars(decoded or "")
    return _strip_xml_incompatible_chars(str(value))


def _coerce_optional_str(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value)
    return text if text else None


def _is_empty_document_error(error: str) -> bool:
    normalized = error.lower()
    return "document is empty" in normalized or "empty html tree" in normalized or "empty html input" in normalized


def _generation_config_for_item_count(stage: Any, item_count: int) -> GenerationConfig:  # noqa: ANN401
    base = stage.generation_config or GenerationConfig()
    if not stage.dynamic_max_tokens or base.max_tokens is None or item_count <= 0:
        return base
    dynamic_max_tokens = max(
        stage.dynamic_min_max_tokens,
        item_count * stage.dynamic_max_tokens_per_item + stage.dynamic_max_token_padding,
    )
    return replace(base, max_tokens=min(base.max_tokens, dynamic_max_tokens))


def _apply_fallback_extraction(
    bindings: object, fallback_handler: object, case: object, primary_error: str
) -> tuple[object, str, str]:
    try:
        case = bindings.extract_main_html_fallback(case, fallback_handler=fallback_handler)
    except Exception as fallback_exc:  # noqa: BLE001
        if primary_error:
            return case, primary_error, f"{primary_error}; fallback failed: {fallback_exc}"
        return case, "", f"fallback failed: {fallback_exc}"
    else:
        return case, primary_error, ""


def _numeric_series_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _append_warning(existing: str, new_warning: str) -> str:
    if not existing:
        return new_warning
    if not new_warning:
        return existing
    return f"{existing}; {new_warning}"


def _convert_main_html(bindings: _MinerUHTMLBindings, main_html: str, url: object) -> str:
    case = bindings.case_cls(bindings.input_cls(raw_html="", url=_coerce_optional_str(url)))
    case.output_data = bindings.output_cls(main_html=main_html)
    _sanitize_case_output_html(case)
    case = bindings.convert2content(case, output_format="mm_md")
    output_data = getattr(case, "output_data", None)
    return str(getattr(output_data, "main_content", "") or "") if output_data else ""


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


_XML_CHAR_SINGLE = {0x09, 0x0A, 0x0D}
_XML_CHAR_RANGES = ((0x20, 0xD7FF), (0xE000, 0xFFFD), (0x10000, 0x10FFFF))


def _strip_xml_incompatible_chars(value: str) -> str:
    return "".join(
        c for c in value if (cp := ord(c)) in _XML_CHAR_SINGLE or any(lo <= cp <= hi for lo, hi in _XML_CHAR_RANGES)
    )


def _coerce_usage_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _with_structured_output_config(
    generation_config: GenerationConfig,
    prompt: str,
    mode: str,
) -> GenerationConfig:
    if mode == "none":
        return generation_config
    item_ids = _item_ids_in_html(prompt)
    if not item_ids or not all(item_id.isdigit() for item_id in item_ids):
        return generation_config

    item_pattern = "".join(f"{re.escape(i)}(main|other)" for i in item_ids)
    regex = f"<answer>\\s*{item_pattern}\\s*</answer>"
    extra_kwargs = dict(generation_config.extra_kwargs or {})
    raw_extra_body = extra_kwargs.get("extra_body")
    if raw_extra_body is not None and not isinstance(raw_extra_body, dict):
        logger.warning("Skipping Dripper structured output because extra_body is not a dict")
        return generation_config
    extra_body: dict[str, Any] = dict(raw_extra_body) if isinstance(raw_extra_body, dict) else {}

    if mode == "structured_outputs":
        extra_body["structured_outputs"] = {"regex": regex}
    elif mode == "guided_regex":
        extra_body["guided_regex"] = regex
    else:
        return generation_config
    extra_kwargs["extra_body"] = extra_body
    return replace(generation_config, extra_kwargs=extra_kwargs)


def _item_ids_in_html(html: str) -> list[str]:
    # dict.fromkeys preserves insertion order and deduplicates
    return list(dict.fromkeys(_ITEM_ID_RE.findall(html)))


_ITEM_ID_RE = re.compile(r"""_item_id\s*=\s*["']?([^"'\s>]+)""")

_STRUCTURED_OUTPUT_MODES = {"none", "structured_outputs", "guided_regex"}
