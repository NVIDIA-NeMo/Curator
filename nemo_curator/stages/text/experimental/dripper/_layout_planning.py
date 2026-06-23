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

"""Layout-group planning and URL/DOM helpers for DripperHTMLLayoutTemplateStage."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlparse

import pandas as pd  # noqa: TC002 — used at runtime (df.iterrows, df.iloc, etc.)
from loguru import logger

from nemo_curator.stages.text.experimental.dripper.stage import (
    _DRIPPER_NEEDS_LLM_COL,
    _coerce_html,
    _is_missing,
)

_LAYOUT_RE_MD5 = re.compile(r"^[0-9a-f]{32}$")
_LAYOUT_RE_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_LAYOUT_RE_UUID = re.compile(r"^[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}$")
_LAYOUT_RE_TIMESTAMP = re.compile(r"^\d{10,13}$")
_LAYOUT_RE_NUM = re.compile(r"\d+")

_LAYOUT_SEMANTIC_QUERY_VALUE_KEYS = {"hl", "lang", "language", "locale"}
_LAYOUT_EXACT_QUERY_VALUE_KEYS = {"id"}

_LAYOUT_PAGE_SIGNATURE_MODES = {
    "none",
    "url_shape",
    "url_low_card_query_shape",
    "url_semantic_shape",
    "item_count_bucket",
    "item_count_exact",
    "url_shape_item_count_bucket",
    "url_shape_item_count_exact",
    "url_low_card_query_shape_item_count_bucket",
    "url_low_card_query_shape_item_count_exact",
    "url_semantic_shape_item_count_bucket",
    "url_semantic_shape_item_count_exact",
}


def _parse_url(value: object) -> tuple[str, object]:
    text = "" if _is_missing(value) else str(value).strip()
    if not text:
        return "", None
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    return text, parsed


def _url_host_key(value: object) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def _normalize_url_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        segment, extension = segment.rsplit(".", 1)
        suffix = f".{extension}"
    if re.search(r"\d", segment):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


def _url_shape_key(value: object) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    query_keys = ",".join(sorted({key for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}))
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [_normalize_url_path_segment(segment) for segment in raw_segments]
    return f"path={'/'.join(normalized_segments)}|q={query_keys}"


def _url_low_card_query_shape_key(value: object, low_card_query_keys: set[str]) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [_normalize_url_path_segment(segment) for segment in raw_segments]

    include_all_query_values = bool(parsed.query) and not low_card_query_keys
    query_parts = []
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.strip().lower()
        if not lowered_key:
            continue
        if (
            include_all_query_values
            or lowered_key in low_card_query_keys
            or lowered_key in _LAYOUT_EXACT_QUERY_VALUE_KEYS
        ):
            query_parts.append(f"{lowered_key}={query_value.strip().lower()}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _url_semantic_shape_key(value: object) -> str:
    def _norm_seg(seg: str) -> str:
        seg = seg.lower()
        suffix = ""
        if "." in seg:
            seg, ext = seg.rsplit(".", 1)
            suffix = f".{ext}"
        if (
            seg.isdigit()
            or _LAYOUT_RE_MD5.fullmatch(seg)
            or _LAYOUT_RE_SHA1.fullmatch(seg)
            or _LAYOUT_RE_UUID.fullmatch(seg)
            or _LAYOUT_RE_TIMESTAMP.fullmatch(seg)
        ):
            return f"#num{suffix}"
        return f"{seg}{suffix}"

    def _norm_qval(v: str) -> str:
        t = v.strip().lower()
        if not t:
            return ""
        if (
            t.isdigit()
            or _LAYOUT_RE_MD5.fullmatch(t)
            or _LAYOUT_RE_SHA1.fullmatch(t)
            or _LAYOUT_RE_UUID.fullmatch(t)
            or _LAYOUT_RE_TIMESTAMP.fullmatch(t)
        ):
            return "#num"
        return t

    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    normalized_segments = [_norm_seg(segment) for segment in raw_segments]
    query_parts = []
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.lower()
        if lowered_key in _LAYOUT_SEMANTIC_QUERY_VALUE_KEYS:
            query_parts.append(f"{lowered_key}={_norm_qval(query_value)}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _coerce_item_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _coerce_positive_int(value: object) -> int:
    return max(0, _coerce_item_count(value))


# (threshold, label) — label=None → use str(count); count > 128 → "129+"
_ITEM_COUNT_BUCKETS: tuple[tuple[int, str | None], ...] = (
    (8, None),
    (16, "9-16"),
    (32, "17-32"),
    (64, "33-64"),
    (128, "65-128"),
)


def _item_count_bucket(value: object) -> str:
    count = _coerce_item_count(value)
    if count <= 0:
        return "0"
    for threshold, label in _ITEM_COUNT_BUCKETS:
        if count <= threshold:
            return str(count) if label is None else label
    return "129+"


def _layout_page_signature_key(url_value: object, item_count_value: object, mode: str) -> str:
    return _layout_page_signature_key_with_low_card_queries(url_value, item_count_value, mode, set())


def _layout_page_signature_key_with_low_card_queries(
    url_value: object,
    item_count_value: object,
    mode: str,
    low_card_query_keys: set[str],
) -> str:
    if not mode or mode == "none":
        return ""
    parts: list[str] = []
    if "url_low_card_query_shape" in mode:
        parts.append(f"url={_url_low_card_query_shape_key(url_value, low_card_query_keys)}")
    elif "url_semantic_shape" in mode:
        parts.append(f"url={_url_semantic_shape_key(url_value)}")
    elif "url_shape" in mode:
        parts.append(f"url={_url_shape_key(url_value)}")
    if "item_count_exact" in mode:
        parts.append(f"items={_coerce_item_count(item_count_value)}")
    elif "item_count_bucket" in mode:
        parts.append(f"items={_item_count_bucket(item_count_value)}")
    return "|".join(parts)


def _validation_query_values(url_text: str) -> list[tuple[str, str]]:
    _text, parsed = _parse_url(url_text)
    if parsed is None:
        return []
    return [
        (key.strip().lower(), value.strip().lower())
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.strip()
    ]


def _low_card_query_value_keys(url_values: list[Any], max_distinct: int = 16) -> set[str]:
    values_by_key: dict[str, set[str]] = defaultdict(set)
    for url_value in url_values:
        url_text = "" if _is_missing(url_value) else str(url_value)
        for key, value in _validation_query_values(url_text):
            values_by_key[key].add(value)
    return {key for key, values in values_by_key.items() if 1 < len(values) <= max_distinct}


_LAYOUT_TAGS_TO_IGNORE = {"script", "style", "meta", "link", "br", "noscript"}
_LAYOUT_TAGS_IGNORE_ATTR = {"a", "i", "b", "li", "tr", "td", "img", "p", "body"}
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize_attr_tokens(value: str | None) -> str:
    if not value:
        return ""
    tokens = value.split()
    if len(tokens) > 1:
        normalized = [token.lower() for token in tokens if not _LAYOUT_RE_NUM.search(token)]
    else:
        lowered = tokens[0].strip().lower()
        normalized_tok = next(
            (
                label
                for pat, label in (
                    (_LAYOUT_RE_MD5, "[MD5]"),
                    (_LAYOUT_RE_SHA1, "[SHA1]"),
                    (_LAYOUT_RE_UUID, "[UUID]"),
                    (_LAYOUT_RE_TIMESTAMP, "[TIMESTAMP]"),
                )
                if pat.fullmatch(lowered)
            ),
            _LAYOUT_RE_NUM.sub("", lowered),
        )
        normalized = [normalized_tok] if normalized_tok else []
    return " ".join(token for token in normalized if token)


def _walk_dom_element(element: object) -> object:
    raw_tag = getattr(element, "tag", None)
    if not isinstance(raw_tag, str):
        return None
    tag = raw_tag.lower()
    if tag in _LAYOUT_TAGS_TO_IGNORE:
        return None
    attrs: list[tuple[str, str]] = []
    if tag not in _LAYOUT_TAGS_IGNORE_ATTR:
        class_attr = _normalize_attr_tokens(element.get("class"))
        id_attr = _normalize_attr_tokens(element.get("id"))
        if class_attr:
            attrs.append(("class", class_attr))
        if id_attr:
            attrs.append(("id", id_attr))
    children = [child for child in (_walk_dom_element(child) for child in element) if child is not None]
    return [tag, attrs, children]


def _layout_dom_path_fingerprint(html_text: str) -> str:
    try:
        from lxml.html import HTMLParser, fromstring
    except ModuleNotFoundError:
        return ""
    try:
        parser = HTMLParser(collect_ids=False, encoding="utf-8", remove_comments=True, remove_pis=True)
        root = fromstring(html_text.encode("utf-8", errors="ignore"), parser=parser)
        body_nodes = root.xpath("//body")
        root = body_nodes[0] if body_nodes else root
    except Exception:  # noqa: BLE001
        return ""
    return json.dumps(_walk_dom_element(root), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _layout_feature_fingerprint(feature: object) -> str:
    if not isinstance(feature, dict):
        return ""

    def normalize_part(part: str) -> dict[str, list[tuple[str, int]]]:
        raw = feature.get(part, {})
        if not isinstance(raw, dict):
            return {}
        return {
            str(layer): sorted(Counter(str(v) for v in vals).items())
            for layer, vals in raw.items()
            if isinstance(vals, list)
        }

    payload = {"tags": normalize_part("tags"), "attrs": normalize_part("attrs")}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _labels_to_webkit_response(labels: object) -> dict[str, int]:
    if not isinstance(labels, dict):
        return {}
    response: dict[str, int] = {}
    for item_id, label in labels.items():
        normalized = str(label).strip().lower()
        response[f"item_id {item_id}"] = 1 if normalized in {"main", "1", "true"} else 0
    return response


def _item_id_response(all_item_ids: list[str], main_item_ids: set[str]) -> str:
    labels = {item_id: ("main" if item_id in main_item_ids else "other") for item_id in all_item_ids}
    if all(item_id.isdigit() for item_id in all_item_ids):
        return "".join(f"{item_id}{label}" for item_id, label in labels.items())
    return json.dumps(labels, ensure_ascii=False, separators=(",", ":"))


def _token_f1(candidate: object, reference: object) -> float:
    candidate_tokens = Counter(_TOKEN_RE.findall(str(candidate or "").lower()))
    reference_tokens = Counter(_TOKEN_RE.findall(str(reference or "").lower()))
    if not candidate_tokens and not reference_tokens:
        return 1.0
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = sum((candidate_tokens & reference_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(candidate_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.stages.text.experimental.dripper.stage import (
        _LLMWebKitBindings,
    )

# Column name duplicated here to avoid a circular import with layout_template.py.
_DRIPPER_ITEM_COUNT_COL = "dripper_item_count"
_MAX_EXEMPLARS_PER_LAYOUT = 3


@dataclass(kw_only=True)
class DripperLayoutAdvancedConfig:
    host_single_cluster_min_pages: int = 0
    host_single_cluster_max_pages: int = 0
    max_exact_host_pages: int = 0
    large_host_mode: Literal["standalone", "feature_hash", "dom_path_hash"] = "standalone"
    propagation_concurrency: int = 32
    representative_candidates: int = 1
    defer_fallback_llm: bool = False
    defer_propagation: bool = False
    failed_host_fallback_signature_mode: str = "none"
    failed_layout_fallback_signature_mode: str = "none"
    page_signature_mode: str = "none"
    validation_signature_mode: str = "none"


@dataclass(frozen=True)
class _LayoutGroupPlan:
    indexes: list[int]
    host_key: str = ""
    source: str = "dom"
    fallback_groups: tuple[list[int], ...] = ()


@dataclass(frozen=True)
class _LayoutPlanningConfig:
    html_col: str
    url_col: str | None
    host_col: str | None
    layout_id_col: str | None
    layout_cluster_threshold: float
    min_cluster_size: int
    adv: DripperLayoutAdvancedConfig
    web_bindings: _LLMWebKitBindings | None


def _build_layout_group_plans(cfg: _LayoutPlanningConfig, df: pd.DataFrame) -> list[_LayoutGroupPlan]:
    if len(df) < cfg.min_cluster_size:
        return []
    precomputed_plans = _build_precomputed_layout_group_plans(cfg, df)
    if precomputed_plans is not None:
        return precomputed_plans

    samples_by_host = _build_host_samples(cfg, df)
    return _build_plans_from_host_samples(cfg, df, samples_by_host)


def _build_host_samples(cfg: _LayoutPlanningConfig, df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in df.iterrows():
        if not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
            continue
        html_text = _coerce_html(row.get(cfg.html_col, ""))
        if not html_text.strip():
            continue
        try:
            feature = cfg.web_bindings.get_feature(html_text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dripper layout feature extraction failed for row {}: {}", idx, exc)
            continue
        if feature is None:
            continue
        samples_by_host[_row_host_key(cfg, row)].append({"track_id": str(idx), "html": html_text, "feature": feature})
    return samples_by_host


def _build_plans_from_host_samples(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    samples_by_host: dict[str, list[dict[str, Any]]],
) -> list[_LayoutGroupPlan]:
    plans: list[_LayoutGroupPlan] = []
    adv = cfg.adv
    for host_key, samples in samples_by_host.items():
        if len(samples) < cfg.min_cluster_size:
            continue
        host_indexes = sorted(int(sample["track_id"]) for sample in samples)
        fallback_groups = _build_layout_groups_for_host_samples(cfg, df, host_key, samples)
        n = len(samples)
        try_single = (
            adv.host_single_cluster_min_pages > 0
            and n >= adv.host_single_cluster_min_pages
            and not (adv.host_single_cluster_max_pages > 0 and n > adv.host_single_cluster_max_pages)
        )
        if try_single:
            plans.append(
                _LayoutGroupPlan(
                    indexes=host_indexes,
                    host_key=host_key,
                    source="host_single_cluster",
                    fallback_groups=tuple(fallback_groups),
                )
            )
            continue
        for indexes in fallback_groups:
            plans.append(
                _LayoutGroupPlan(
                    indexes=indexes,
                    host_key=host_key,
                    source="dom",
                    fallback_groups=tuple(_build_failed_layout_fallback_groups(cfg, df, indexes)),
                )
            )
    return plans


def _build_precomputed_layout_group_plans(
    cfg: _LayoutPlanningConfig, df: pd.DataFrame
) -> list[_LayoutGroupPlan] | None:
    if not cfg.layout_id_col or cfg.layout_id_col not in df.columns:
        return None

    by_layout: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        if not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
            continue
        html_text = _coerce_html(row.get(cfg.html_col, ""))
        if not html_text.strip():
            continue
        layout_key = _row_layout_id_key(cfg, row)
        if not layout_key:
            continue
        by_layout[(_row_host_key(cfg, row), layout_key)].append(int(idx))

    plans: list[_LayoutGroupPlan] = []
    for (host_key, layout_key), indexes in sorted(by_layout.items(), key=lambda item: (min(item[1]), item[0])):
        sorted_indexes = sorted(indexes)
        if len(sorted_indexes) < cfg.min_cluster_size:
            continue
        plan_groups = _split_large_precomputed_layout_group(cfg, df, host_key, layout_key, sorted_indexes)
        for plan_indexes in plan_groups:
            if len(plan_indexes) < cfg.min_cluster_size:
                continue
            plans.append(
                _LayoutGroupPlan(
                    indexes=plan_indexes,
                    host_key=host_key,
                    source=f"precomputed_layout:{layout_key}",
                    fallback_groups=tuple(_build_failed_layout_fallback_groups(cfg, df, plan_indexes)),
                )
            )
    return plans


def _split_large_precomputed_layout_group(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    host_key: str,
    _layout_key: str,
    indexes: list[int],
) -> list[list[int]]:
    adv = cfg.adv
    if not adv.max_exact_host_pages or len(indexes) <= adv.max_exact_host_pages:
        return [indexes]
    if adv.large_host_mode == "standalone":
        return []

    samples: list[dict[str, Any]] = []
    for idx in indexes:
        html_text = _coerce_html(df.iloc[idx].get(cfg.html_col, ""))
        if not html_text.strip():
            continue
        sample: dict[str, Any] = {"track_id": str(idx), "html": html_text}
        if adv.large_host_mode == "feature_hash":
            try:
                feature = cfg.web_bindings.get_feature(html_text) if cfg.web_bindings else None
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dripper precomputed layout feature extraction failed for row {}: {}", idx, exc)
                continue
            if feature is None:
                continue
            sample["feature"] = feature
        samples.append(sample)
    fingerprint_fn = (
        (lambda sample: _layout_feature_fingerprint(sample.get("feature")))
        if adv.large_host_mode == "feature_hash"
        else (lambda sample: _layout_dom_path_fingerprint(str(sample.get("html") or "")))
    )
    return _build_fingerprint_groups(cfg, df, host_key, samples, fingerprint_fn=fingerprint_fn)


def _row_host_key(cfg: _LayoutPlanningConfig, row: pd.Series) -> str:
    if cfg.host_col and cfg.host_col in row:
        host_key = _url_host_key(row.get(cfg.host_col))
        if host_key:
            return host_key
    return _url_host_key(row.get(cfg.url_col) if cfg.url_col else None)


def _row_layout_id_key(cfg: _LayoutPlanningConfig, row: pd.Series) -> str:
    if not cfg.layout_id_col:
        return ""
    value = row.get(cfg.layout_id_col)
    text = "" if _is_missing(value) else str(value).strip()
    if not text or text in {"-1", "-2"} or text.endswith(("_-1", "_-2")):
        return ""
    return text


def _build_layout_groups_for_host_samples(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    host_key: str,
    samples: list[dict[str, Any]],
) -> list[list[int]]:
    if len(samples) < cfg.min_cluster_size:
        return []

    # Large-host fast path: skip clustering, use fingerprint bucketing instead.
    adv = cfg.adv
    if adv.max_exact_host_pages and len(samples) > adv.max_exact_host_pages:
        if adv.large_host_mode == "feature_hash":
            fingerprint_fn = lambda sample: _layout_feature_fingerprint(sample.get("feature"))  # noqa: E731
        elif adv.large_host_mode == "dom_path_hash":
            fingerprint_fn = lambda sample: _layout_dom_path_fingerprint(str(sample.get("html") or ""))  # noqa: E731
        else:
            return []
        return _build_fingerprint_groups(cfg, df, host_key, samples, fingerprint_fn=fingerprint_fn)

    try:
        clustered_samples, _layout_ids = cfg.web_bindings.cluster_html_struct(
            samples,
            threshold=cfg.layout_cluster_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Dripper layout clustering failed for host {}: {}", host_key, exc)
        return []

    if not clustered_samples:
        return []
    return _build_clustered_host_groups(cfg, df, host_key, clustered_samples)


def _build_clustered_host_groups(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    _host_key: str,
    clustered_samples: list[dict[str, Any]],
) -> list[list[int]]:
    max_layer_n = int(
        next((s.get("max_layer_n") for s in clustered_samples if int(s.get("layout_id", -1)) >= 0), None) or 5
    )
    exemplars_by_layout: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in clustered_samples:
        layout_id = int(sample.get("layout_id", -1))
        if layout_id < 0:
            continue
        if len(exemplars_by_layout[layout_id]) < _MAX_EXEMPLARS_PER_LAYOUT:
            exemplars_by_layout[layout_id].append(sample)

    by_layout: dict[tuple[int, str], list[int]] = defaultdict(list)
    for sample in clustered_samples:
        layout_id = _assign_layout_by_exemplar_similarity(cfg, sample.get("feature"), exemplars_by_layout, max_layer_n)
        if layout_id < 0:
            continue
        row_idx = int(sample["track_id"])
        _row = df.iloc[row_idx]
        signature_key = _layout_page_signature_key(
            _row.get(cfg.url_col) if cfg.url_col else None,
            _row.get(_DRIPPER_ITEM_COUNT_COL),
            cfg.adv.page_signature_mode,
        )
        by_layout[(layout_id, signature_key)].append(row_idx)
    groups: list[list[int]] = []
    for (_layout_id, _signature_key), indexes in sorted(by_layout.items()):
        if len(indexes) >= cfg.min_cluster_size:
            groups.append(sorted(indexes))
    return groups


def _build_failed_layout_fallback_groups(
    cfg: _LayoutPlanningConfig, df: pd.DataFrame, indexes: list[int]
) -> list[list[int]]:
    mode = cfg.adv.failed_layout_fallback_signature_mode
    if mode == "none" or len(indexes) < cfg.min_cluster_size:
        return []

    children = _split_fallback_groups_by_signature(cfg, df, [indexes], mode)
    parent_set = set(indexes)
    return [child for child in children if set(child) != parent_set]


def _assign_layout_by_exemplar_similarity(
    cfg: _LayoutPlanningConfig,
    feature: object,
    exemplars_by_layout: dict[int, list[dict[str, Any]]],
    max_layer_n: int,
) -> int:
    for layout_id, exemplars in sorted(exemplars_by_layout.items()):
        for exemplar in exemplars:
            try:
                score = cfg.web_bindings.similarity(feature, exemplar.get("feature"), max_layer_n)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dripper layout similarity failed for layout {}: {}", layout_id, exc)
                continue
            if score is not None and score >= cfg.layout_cluster_threshold:
                return layout_id
    return -2


def _build_fingerprint_groups(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    _host_key: str,
    samples: list[dict[str, Any]],
    *,
    fingerprint_fn: Callable[[dict[str, Any]], str],
) -> list[list[int]]:
    by_fingerprint: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        by_fingerprint[fingerprint_fn(sample)].append(int(sample["track_id"]))

    groups: list[list[int]] = []
    for _fingerprint, indexes in sorted(by_fingerprint.items(), key=lambda item: (min(item[1]), item[0])):
        by_signature: dict[str, list[int]] = defaultdict(list)
        for row_idx in indexes:
            _row = df.iloc[row_idx]
            signature_key = _layout_page_signature_key(
                _row.get(cfg.url_col) if cfg.url_col else None,
                _row.get(_DRIPPER_ITEM_COUNT_COL),
                cfg.adv.page_signature_mode,
            )
            by_signature[signature_key].append(row_idx)
        for _signature_key, signature_indexes in sorted(by_signature.items()):
            if len(signature_indexes) < cfg.min_cluster_size:
                continue
            groups.append(sorted(signature_indexes))
    return groups


def _split_fallback_groups_by_signature(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    groups: list[list[int]],
    mode: str,
) -> list[list[int]]:
    split_groups: list[list[int]] = []
    for group in groups:
        low_card_query_keys: set[str] = set()
        if "url_low_card_query_shape" in mode and cfg.url_col:
            low_card_query_keys = _low_card_query_value_keys([df.iloc[row_idx].get(cfg.url_col) for row_idx in group])
        by_signature: dict[str, list[int]] = defaultdict(list)
        use_low_card = "url_low_card_query_shape" in mode
        for row_idx in group:
            row = df.iloc[row_idx]
            url = row.get(cfg.url_col) if cfg.url_col else None
            if use_low_card:
                signature_key = _layout_page_signature_key_with_low_card_queries(
                    url, row.get(_DRIPPER_ITEM_COUNT_COL), mode, low_card_query_keys
                )
            else:
                signature_key = _layout_page_signature_key(url, row.get(_DRIPPER_ITEM_COUNT_COL), mode)
            by_signature[signature_key].append(row_idx)
        for _signature, indexes in sorted(by_signature.items(), key=lambda item: (min(item[1]), item[0])):
            if len(indexes) >= cfg.min_cluster_size:
                split_groups.append(sorted(indexes))
    return split_groups


_QUERY_POSITIONS_THRESHOLD = 8
_QUERY_POSITIONS_HIGH = 4
_QUERY_POSITIONS_LOW = 3

_ColSpec = tuple[str | None, str]


@dataclass
class _SelectorState:
    selected: list[int]
    selected_set: set[int]
    count: int
    url_col: str | None
    item_count_col: str

    def add(self, idx: int) -> None:
        if len(self.selected) >= self.count or idx in self.selected_set:
            return
        self.selected.append(idx)
        self.selected_set.add(idx)

    def is_full(self) -> bool:
        return len(self.selected) >= self.count


def _select_by_signature(
    df: pd.DataFrame,
    indexes: list[int],
    *,
    signature_mode: str,
    state: _SelectorState,
) -> bool:
    url_col = state.url_col
    item_count_col = state.item_count_col
    low_card_query_keys: set[str] = set()
    if "url_low_card_query_shape" in signature_mode and url_col:
        low_card_query_keys = _low_card_query_value_keys([df.iloc[idx].get(url_col) for idx in indexes])
    by_signature: dict[str, list[int]] = defaultdict(list)
    for idx in indexes:
        row = df.iloc[idx]
        signature_key = _layout_page_signature_key_with_low_card_queries(
            row.get(url_col) if url_col else None,
            row.get(item_count_col) if item_count_col in row else None,
            signature_mode,
            low_card_query_keys,
        )
        by_signature[signature_key].append(idx)
    signature_groups = sorted(
        by_signature.values(),
        key=lambda group: (-len(group), _validation_sample_key(df.iloc[group[0]], group[0], url_col, item_count_col)),
    )
    for group in signature_groups:
        for idx in _select_validation_indexes(df, sorted(group), 1, (url_col, item_count_col), signature_mode="none"):
            state.add(idx)
            break
        if state.is_full():
            return True
    return False


def _select_by_url(
    df: pd.DataFrame,
    indexes: list[int],
    *,
    state: _SelectorState,
) -> None:
    url_col = state.url_col
    count = state.count
    query_value_rows: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for idx in indexes:
        url_text = str(df.iloc[idx].get(url_col) or "")
        for key, value in _validation_query_values(url_text):
            query_value_rows[key].append((value, idx))
    for key in sorted(query_value_rows):
        entries = sorted(query_value_rows[key])
        query_positions = _QUERY_POSITIONS_HIGH if count >= _QUERY_POSITIONS_THRESHOLD else _QUERY_POSITIONS_LOW
        for position in _spread_positions(len(entries), min(count, query_positions)):
            state.add(entries[position][1])
        if state.is_full():
            return

    url_sorted = sorted(indexes, key=lambda idx: (str(df.iloc[idx].get(url_col) or ""), idx))
    for position in _spread_positions(len(url_sorted), count):
        state.add(url_sorted[position])
        if state.is_full():
            return


def _select_validation_indexes(
    df: pd.DataFrame,
    indexes: list[int],
    count: int,
    cols: _ColSpec,
    *,
    signature_mode: str = "none",
) -> list[int]:
    url_col, item_count_col = cols
    if count <= 0 or not indexes:
        return []
    if count >= len(indexes):
        return list(indexes)
    if count == 1:
        return [indexes[-1]]

    state = _SelectorState(
        selected=[], selected_set=set(), count=count, url_col=url_col, item_count_col=item_count_col
    )

    if (
        signature_mode
        and signature_mode != "none"
        and _select_by_signature(df, indexes, signature_mode=signature_mode, state=state)
    ):
        return sorted(state.selected)

    state.add(indexes[0])
    state.add(indexes[-1])

    item_sorted = sorted(indexes, key=lambda idx: (_coerce_item_count(df.iloc[idx].get(item_count_col)), idx))
    state.add(item_sorted[0])
    state.add(item_sorted[-1])

    if url_col:
        _select_by_url(df, indexes, state=state)
        if state.is_full():
            return sorted(state.selected)

    remaining = [idx for idx in indexes if idx not in state.selected_set]
    remaining.sort(key=lambda idx: _validation_sample_key(df.iloc[idx], idx, url_col, item_count_col))
    for idx in remaining:
        state.add(idx)
        if state.is_full():
            break
    return sorted(state.selected)


def _spread_positions(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({round(slot * (length - 1) / (count - 1)) for slot in range(count)})


def _validation_sample_key(
    row: pd.Series,
    row_index: int,
    url_col: str | None,
    item_count_col: str,
) -> tuple[int, int]:
    import hashlib

    url_text = str(row.get(url_col) or "") if url_col else ""
    item_count = str(row.get(item_count_col) or "")
    payload = f"{url_text}\0{item_count}\0{row_index}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False), row_index
