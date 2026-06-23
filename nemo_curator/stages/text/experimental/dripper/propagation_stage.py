from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper._layout_planning import _token_f1
from nemo_curator.stages.text.experimental.dripper.stage import (
    _coerce_html,
    _convert_main_html,
    _load_llm_web_kit_bindings,
    _load_mineru_html_bindings,
    _MinerUHTMLBindings,
    _rebuild_batch,
    _strip_xml_incompatible_chars,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    import pandas as pd


_PENDING_COL = "dripper_layout_pending_propagation"
_MAPPING_COL = "dripper_layout_mapping_json"
_CLUSTER_COL = "dripper_layout_cluster"
_REPRESENTATIVE_COL = "dripper_layout_representative"

# Number of siblings sampled to validate static-LBP trustworthiness per cluster.
_K_SAMPLE_SIBLINGS = 3

# Maximum HTML bytes forwarded to the content converter (guards against OOM).
_MAX_CONTENT_HTML_BYTES = 200_000


@dataclass
class _StaticTrustConfig:
    memo: dict[str, bool]
    lbp_fn: Any  # (html, mapping_data, dynamic) -> (str, str)
    content_fn: Any  # (main_html, url) -> (str, str)
    threshold: float


@dataclass
class _PropagationConfig:
    lbp_fn: Any  # (html, mapping_data, dynamic) -> (str, str)
    content_fn: Any  # (main_html, url) -> (str, str)
    min_ratio: float
    max_ratio: float


def _run_lbp(
    params: dict[str, Any],
    html: str,
    mapping_data: dict[str, Any],
    dynamic: bool,
    _parser_cache: dict | None = None,
) -> tuple[str, str]:
    html_source = html.strip()
    if not html_source:
        return "", "empty_html"
    try:
        from llm_web_kit.main_html_parser.parser.layout_batch_parser import LayoutBatchParser

        task_data = dict(mapping_data)
        if "_parsed_element_dict" in task_data:
            task_data["html_element_dict"] = task_data.pop("_parsed_element_dict")
        task_data["html_source"] = html_source
        task_data["dynamic_id_enable"] = task_data["dynamic_classid_enable"] = dynamic
        task_data["more_noise_enable"] = params.get("more_noise_enable", True)
        task_data["dynamic_classid_similarity_threshold"] = params.get("dynamic_classid_similarity_threshold", 0.70)
        element_dict = task_data.get("html_element_dict")
        cache_key = id(element_dict) if element_dict is not None else None
        if _parser_cache is not None and cache_key is not None:
            if cache_key not in _parser_cache:
                _parser_cache[cache_key] = LayoutBatchParser({})
            parser = _parser_cache[cache_key]
        else:
            parser = LayoutBatchParser({})
        parts = parser.parse(task_data)
    except Exception as exc:  # noqa: BLE001
        return "", f"layout_parser_error={exc!s:.200}"
    main_html = str(parts.get("main_html_body") or "")
    if not main_html.strip():
        if parts.get("main_html_success") is False:
            return "", f"main_html_success_false sim={parts.get('main_html_sim', 'n/a')}"
        return "", "layout_parser_empty_output"
    return main_html, ""


def _run_content_convert(
    bindings: _MinerUHTMLBindings,
    main_html: str,
    url: str,
) -> tuple[str, str]:
    if len(main_html) > _MAX_CONTENT_HTML_BYTES:
        main_html = main_html[:_MAX_CONTENT_HTML_BYTES]
    try:
        sanitized = _strip_xml_incompatible_chars(main_html)
        content = _convert_main_html(bindings, sanitized, url)
        return str(content or ""), ""
    except Exception as exc:  # noqa: BLE001
        return "", f"content_conversion_error={exc!s:.150}"


def _cluster_static_trustworthy(
    cluster_id: object,
    sample_rows: list[dict[str, Any]],
    mapping_data: dict[str, Any],
    cfg: _StaticTrustConfig,
) -> bool:
    if mapping_data is None:
        return False
    key = str(cluster_id)
    if key in cfg.memo:
        return cfg.memo[key]
    f1s: list[float] = []
    for row in sample_rows[:_K_SAMPLE_SIBLINGS]:
        html = _coerce_html(row.get("html", ""))
        if not html.strip():
            continue
        sh, se = cfg.lbp_fn(html, mapping_data, False)
        dh, de = cfg.lbp_fn(html, mapping_data, True)
        if not dh or de:
            continue
        url = row.get("url", "")
        if not sh or se:
            f1s.append(0.0)
        else:
            sc, _ = cfg.content_fn(sh, url)
            dc, _ = cfg.content_fn(dh, url)
            f1s.append(_token_f1(sc, dc))
    ok = bool(f1s) and (sum(f1s) / len(f1s) >= cfg.threshold)
    cfg.memo[key] = ok
    return ok


def _lbp_once(
    html: str,
    url: str,
    mapping_data: dict[str, Any],
    dynamic: bool,
    prop_cfg: _PropagationConfig,
) -> tuple[str, str, str]:
    lh, le = prop_cfg.lbp_fn(html, mapping_data, dynamic)
    if not lh or le:
        return "", "", le
    rc, ce = prop_cfg.content_fn(lh, url)
    if ce:
        return "", "", ce
    rep_len = (mapping_data or {}).get("_dripper_representative_content_len")
    if rep_len and rep_len > 0:
        ratio = len(rc) / rep_len
        if ratio < prop_cfg.min_ratio:
            return "", "", f"content_length_ratio_low={ratio:.3f}"
        if ratio > prop_cfg.max_ratio:
            return "", "", f"content_length_ratio_high={ratio:.3f}"
    return lh, rc, ""


def _sibling_propagate(
    row: dict[str, Any],
    mapping_data: dict[str, Any] | None,
    use_static: bool,
    prop_cfg: _PropagationConfig,
) -> tuple[str, str, str, str]:
    url = row.get("url", "")
    html = _coerce_html(row.get("html", ""))
    method, main_html, content, error = "fallback", "", "", ""

    if mapping_data is not None:
        if use_static:
            main_html, content, error = _lbp_once(html, url, mapping_data, False, prop_cfg)
            if main_html:
                method = "lbp_static"
        if not main_html:
            dh, dc, de = _lbp_once(html, url, mapping_data, True, prop_cfg)
            if dh:
                main_html, method, content, error = dh, "layout_batch_parser", dc, ""
            elif de:
                error = f"static_failed({error}); dynamic_failed({de})" if error else de

    if not main_html:
        method = "fallback"
        error = error or "no_template_available"

    return main_html, content, error, method


@dataclass(kw_only=True)
class DripperHTMLLayoutPropagationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    html_col: str = "html"
    output_html_col: str = "dripper_html"
    output_content_col: str = "dripper_content"
    postprocess_time_col: str = "dripper_postprocess_time_s"
    error_col: str = "dripper_error"
    url_col: str = "url"

    dynamic_classid_similarity_threshold: float = 0.85
    more_noise_enable: bool = True
    layout_template_validation_min_content_f1: float = 0.95
    layout_template_min_content_length_ratio: float | None = 0.25
    layout_template_max_content_length_ratio: float | None = 4.0
    propagation_target: str = "raw_html"
    use_static_lbp: bool = True
    static_validation_min_f1: float = 0.97

    _bindings: Any = field(init=False, repr=False, default=None)
    _web_bindings: Any = field(init=False, repr=False, default=None)
    _cluster_static_ok: dict = field(init=False, repr=False, default_factory=dict)

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.output_html_col,
            self.output_content_col,
            self.postprocess_time_col,
            self.error_col,
            "dripper_layout_propagated",
            "dripper_layout_propagation_success",
            "dripper_layout_propagation_method",
            _PENDING_COL,
        ]

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ANN401, ARG002
        if self._bindings is not None:
            return
        self._bindings = _load_mineru_html_bindings()
        self._web_bindings = _load_llm_web_kit_bindings()
        self._cluster_static_ok = {}

    def _make_lbp_fn(self, parser_cache: dict | None = None) -> Any:  # noqa: ANN401  # returns Callable[[str, dict, bool], tuple[str, str]]
        params = {
            "more_noise_enable": self.more_noise_enable,
            "dynamic_classid_similarity_threshold": self.dynamic_classid_similarity_threshold,
        }

        def _lbp(html: str, mapping_data: dict, dynamic: bool = True) -> tuple[str, str]:
            return _run_lbp(params, html, mapping_data, dynamic, _parser_cache=parser_cache)

        return _lbp

    def _make_content_fn(self) -> Any:  # noqa: ANN401  # returns Callable[[str, str], tuple[str, str]]
        bindings = self._bindings

        def _content(main_html: str, url: str) -> tuple[str, str]:
            return _run_content_convert(bindings, main_html, url)

        return _content

    def _make_prop_cfg(self, parser_cache: dict | None = None) -> _PropagationConfig:
        return _PropagationConfig(
            lbp_fn=self._make_lbp_fn(parser_cache),
            content_fn=self._make_content_fn(),
            min_ratio=self.layout_template_min_content_length_ratio or 0.0,
            max_ratio=self.layout_template_max_content_length_ratio or float("inf"),
        )

    def _make_trust_cfg(self, parser_cache: dict | None = None) -> _StaticTrustConfig:
        return _StaticTrustConfig(
            memo=self._cluster_static_ok,
            lbp_fn=self._make_lbp_fn(parser_cache),
            content_fn=self._make_content_fn(),
            threshold=self.static_validation_min_f1,
        )

    def process(self, batch: DocumentBatch) -> DocumentBatch:  # noqa: C901, PLR0912, PLR0915
        if self._bindings is None:
            self.setup()

        df = batch.to_pandas()

        if _PENDING_COL not in df.columns:
            return batch

        pending_mask = df[_PENDING_COL].astype(bool)
        if not pending_mask.any():
            return batch

        mapping_by_cluster: dict[str, dict[str, Any]] = {}
        if _MAPPING_COL in df.columns and _REPRESENTATIVE_COL in df.columns:
            rep_rows = df[df[_REPRESENTATIVE_COL].astype(bool)]
            for _, row in rep_rows.iterrows():
                mapping_json = str(row.get(_MAPPING_COL) or "")
                cluster = str(row.get(_CLUSTER_COL) or "")
                if mapping_json and cluster:
                    with contextlib.suppress(Exception):
                        mapping_by_cluster[cluster] = json.loads(mapping_json)

        cluster_pending: dict[str, list] = {}
        for idx in df.index[pending_mask]:
            cid = str(df.loc[idx, _CLUSTER_COL] if _CLUSTER_COL in df.columns else "")
            cluster_pending.setdefault(cid, []).append(idx)

        for cid, idxs in cluster_pending.items():
            mapping_data = mapping_by_cluster.get(cid)
            parser_cache: dict = {}
            prop_cfg = self._make_prop_cfg(parser_cache)

            # memoised: validate static-LBP trustworthiness once per cluster
            use_static = False
            if self.use_static_lbp and mapping_data is not None:
                sample_rows = [df.loc[i].to_dict() for i in idxs[:_K_SAMPLE_SIBLINGS]]
                trust_cfg = self._make_trust_cfg(parser_cache)
                use_static = _cluster_static_trustworthy(cid, sample_rows, mapping_data, trust_cfg)

            for idx in idxs:
                row = df.loc[idx]
                t0 = time.perf_counter()
                propagated_html = ""
                propagated_content = ""
                error = ""
                success = False
                method = "fallback"

                if mapping_data is None:
                    error = f"no_mapping_data_for_cluster={cid}"
                else:
                    try:
                        row_dict = row.to_dict()
                        propagated_html, propagated_content, error, method = _sibling_propagate(
                            row_dict, mapping_data, use_static, prop_cfg
                        )
                        if propagated_html and not error:
                            success = True
                    except Exception as exc:  # noqa: BLE001
                        error = f"propagation_exception={exc!s:.200}"

                elapsed = time.perf_counter() - t0
                df.loc[idx, self.output_html_col] = propagated_html
                df.loc[idx, self.output_content_col] = propagated_content
                df.loc[idx, self.postprocess_time_col] = elapsed
                df.loc[idx, self.error_col] = error
                df.loc[idx, "dripper_layout_propagated"] = True
                df.loc[idx, "dripper_layout_propagation_success"] = success
                df.loc[idx, "dripper_layout_propagation_method"] = method
                df.loc[idx, _PENDING_COL] = False  # consumed

        n_pending = int(pending_mask.sum())
        n_success = (
            int(df["dripper_layout_propagation_success"].sum())
            if "dripper_layout_propagation_success" in df.columns
            else 0
        )
        logger.info(
            "DripperHTMLLayoutPropagationStage: propagated {}/{} rows in batch",
            n_success,
            n_pending,
        )
        return _rebuild_batch(batch, df)

    def _run_propagation(
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
    ) -> tuple[str, str, str]:
        if self._bindings is None:
            self.setup()
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        prop_cfg = self._make_prop_cfg()
        main_html, content, error, _ = _sibling_propagate(row_dict, mapping_data, False, prop_cfg)
        return main_html, content, error
