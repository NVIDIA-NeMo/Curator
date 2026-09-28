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

"""Build the extraction-only NRL graph used by the Curator Lance bridge.

This module runs in the NeMo Retriever environment.  Its terminal CPU
operator turns page-level Nemotron Parse results into a small, flat contract
that the separate Curator process can validate and publish.  It deliberately
has no NeMo Curator dependency.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import nrl_lance_contract as contract
import pandas as pd
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator

if TYPE_CHECKING:
    from nemo_retriever.common.ray_resource_hueristics import ClusterResources
    from nemo_retriever.graph.pipeline_graph import Graph

MIN_PICTURE_CROP_PX = 10
_NEMOTRON_ELEMENT_RE = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)"
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.DOTALL,
)


class IncompleteModelOutputError(ValueError):
    """The tagged response contains content outside complete elements."""


def _raw_output_sha256(raw_output: str) -> str:
    return hashlib.sha256(raw_output.encode("utf-8", errors="strict")).hexdigest()


def _row_raw_output_sha256(row: Mapping[str, Any]) -> str | None:
    parser_metadata = row.get("nemotron_parse_v1_2")
    raw_output = parser_metadata.get("raw_output") if isinstance(parser_metadata, Mapping) else None
    return _raw_output_sha256(raw_output) if isinstance(raw_output, str) else None


def _validate_complete_raw_output(raw_output: str) -> int:
    """Return the number of elements only when the entire response parses."""

    if not isinstance(raw_output, str):
        msg = f"raw model output must be text, got {type(raw_output).__name__}"
        raise TypeError(msg)
    matches = list(_NEMOTRON_ELEMENT_RE.finditer(raw_output))
    count = len(matches)
    if (
        raw_output.count("<x_") != count * 2
        or raw_output.count("<y_") != count * 2
        or raw_output.count("<class_") != count
    ):
        msg = "raw model output contains incomplete or unbalanced element tags"
        raise IncompleteModelOutputError(msg)

    cursor = 0
    for match in matches:
        if raw_output[cursor : match.start()].strip():
            msg = f"raw model output has unparsed content at offset {cursor}"
            raise IncompleteModelOutputError(msg)
        cursor = match.end()
    if raw_output[cursor:].strip():
        msg = f"raw model output has unparsed content at offset {cursor}"
        raise IncompleteModelOutputError(msg)
    return count


def _parse_raw_elements(raw_output: str) -> list[dict[str, Any]]:
    expected_count = _validate_complete_raw_output(raw_output)

    from nemo_retriever.common.modality.parse.nemotron_parse_postprocessing import (
        extract_classes_bboxes,
        postprocess_text,
    )

    classes, bboxes, texts = extract_classes_bboxes(raw_output)
    if not (len(classes) == len(bboxes) == len(texts) == expected_count):
        msg = "NRL parser did not return every validated model element"
        raise IncompleteModelOutputError(msg)

    elements: list[dict[str, Any]] = []
    for element_class, bbox, text in zip(classes, bboxes, texts, strict=True):
        processed = postprocess_text(
            text,
            cls=element_class,
            text_format="markdown",
            table_format="latex",  # Preserve native table bodies and merged-cell spans for Curator.
            blank_text_in_figures=False,
        ).strip()
        elements.append({"class": element_class, "text": processed, "bbox": list(bbox)})
    return elements


def _crop_picture_bytes(page_image: object, bbox: Sequence[float]) -> bytes | None:  # noqa: PLR0911
    from nemo_retriever.common.modality.ocr.shared import _crop_b64_image_by_norm_bbox
    from nemo_retriever.common.modality.parse.nemotron_parse_postprocessing import transform_bbox_to_original

    if not isinstance(page_image, dict) or not isinstance(page_image.get("image_b64"), str):
        return None
    shape = page_image.get("orig_shape_hw")
    tolist = getattr(shape, "tolist", None)
    if callable(tolist):
        shape = tolist()
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        height, width = (int(item) for item in shape)
    except (TypeError, ValueError, OverflowError):
        return None
    if height <= 0 or width <= 0:
        return None

    left, top, right, bottom = transform_bbox_to_original(tuple(float(item) for item in bbox), width, height)
    normalized = [left / width, top / height, right / width, bottom / height]
    cropped_b64, cropped_shape = _crop_b64_image_by_norm_bbox(
        page_image["image_b64"],
        bbox_xyxy_norm=normalized,
        image_format="png",
    )
    if cropped_b64 is None or cropped_shape is None or min(cropped_shape) < MIN_PICTURE_CROP_PX:
        return None
    try:
        payload = base64.b64decode(cropped_b64, validate=True)
    except (TypeError, ValueError):
        return None
    return payload or None


def _source_path(row: Mapping[str, Any]) -> str:
    value = row.get("path")
    metadata = row.get("metadata")
    if not value and isinstance(metadata, Mapping):
        value = metadata.get("source_path")
    return os.fspath(value) if isinstance(value, os.PathLike) else str(value or "")


def _native_page_number(row: Mapping[str, Any]) -> int:
    value = row.get("page_number")
    if isinstance(value, bool):
        return 0
    try:
        page_number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, page_number)


def _compact_error(error: object) -> dict[str, Any]:
    if isinstance(error, Mapping):
        compact = {key: error[key] for key in ("stage", "type", "message") if key in error and error[key] is not None}
        return compact or {"message": str(dict(error))}
    return {"message": str(error)}


def _page_outcome_row(  # noqa: PLR0913
    *,
    source_path: str,
    native_page_number: int,
    outcome: str,
    element_count: int,
    issues: Sequence[Mapping[str, Any]],
    raw_output_sha256: str | None,
) -> dict[str, Any]:
    return {
        "record_type": "page_outcome",
        "source_path": source_path,
        "native_page_number": native_page_number,
        "page_outcome": outcome,
        "element_count": element_count,
        "issues_json": contract._canonical_json(list(issues)),
        "raw_output_sha256": raw_output_sha256,
        "element_index": None,
        "element_class": None,
        "modality": None,
        "content_type": None,
        "text_content": None,
        "binary_content": None,
        "bbox_xyxy_norm_json": None,
        "bbox_coordinate_space": None,
    }


def _failed_page_row(
    row: Mapping[str, Any],
    issue: Mapping[str, Any],
    *,
    raw_output_sha256: str | None = None,
) -> dict[str, Any]:
    return _page_outcome_row(
        source_path=_source_path(row),
        native_page_number=_native_page_number(row),
        outcome="failed",
        element_count=0,
        issues=[issue],
        raw_output_sha256=raw_output_sha256,
    )


def _project_page(row: Mapping[str, Any]) -> list[dict[str, Any]]:  # noqa: C901, PLR0911, PLR0912
    source_path = _source_path(row)
    native_page_number = _native_page_number(row)
    metadata = row.get("metadata")
    extraction_error = metadata.get("error") if isinstance(metadata, Mapping) else None
    parser_metadata = row.get("nemotron_parse_v1_2")
    parse_error = parser_metadata.get("error") if isinstance(parser_metadata, Mapping) else None
    raw_output = parser_metadata.get("raw_output") if isinstance(parser_metadata, Mapping) else None
    raw_sha256 = _raw_output_sha256(raw_output) if isinstance(raw_output, str) else None

    if extraction_error is not None or parse_error is not None:
        return [
            _failed_page_row(
                row,
                {"kind": "page_stage_error", "error": _compact_error(extraction_error or parse_error)},
                raw_output_sha256=raw_sha256,
            )
        ]
    if native_page_number == 0:
        return [
            _failed_page_row(
                row,
                {"kind": "document_or_split_failure", "error": _compact_error(row.get("error") or "invalid page")},
                raw_output_sha256=raw_sha256,
            )
        ]
    if not isinstance(row.get("page_image"), Mapping):
        return [_failed_page_row(row, {"kind": "missing_page_image"}, raw_output_sha256=raw_sha256)]
    if raw_output is None:
        return [_failed_page_row(row, {"kind": "missing_model_output"})]
    if not isinstance(raw_output, str):
        return [
            _failed_page_row(
                row,
                {"kind": "invalid_model_output", "type": type(raw_output).__name__},
            )
        ]
    if not raw_output.strip():
        return [
            _page_outcome_row(
                source_path=source_path,
                native_page_number=native_page_number,
                outcome="empty",
                element_count=0,
                issues=[],
                raw_output_sha256=raw_sha256,
            )
        ]

    try:
        elements = _parse_raw_elements(raw_output)
    except (IncompleteModelOutputError, TypeError, ValueError) as exc:
        return [
            _failed_page_row(
                row,
                {"kind": "truncated_or_unparseable_model_output", "detail": str(exc)},
                raw_output_sha256=raw_sha256,
            )
        ]
    if not elements:
        return [
            _failed_page_row(
                row,
                {"kind": "unparseable_model_output"},
                raw_output_sha256=raw_sha256,
            )
        ]

    projected: list[dict[str, Any]] = []
    for element_index, element in enumerate(elements):
        element_class = str(element.get("class") or "").strip()
        bbox = contract._normalize_bbox(element.get("bbox"))
        if not element_class or bbox is None:
            return [
                _failed_page_row(
                    row,
                    {
                        "kind": "invalid_element",
                        "element_index": element_index,
                        "element_class": element_class or None,
                    },
                    raw_output_sha256=raw_sha256,
                )
            ]

        binary_content: bytes | None = None
        if element_class == "Picture":
            binary_content = _crop_picture_bytes(row["page_image"], bbox)
            if binary_content is None:
                return [
                    _failed_page_row(
                        row,
                        {"kind": "picture_crop_failure", "element_index": element_index},
                        raw_output_sha256=raw_sha256,
                    )
                ]
            modality, content_type = "image", "image/png"
        elif element_class == "Table":
            modality, content_type = "table", "text/markdown"
        else:
            modality, content_type = "text", "text/markdown"

        projected.append(
            {
                "record_type": "element",
                "source_path": source_path,
                "native_page_number": native_page_number,
                "page_outcome": None,
                "element_count": None,
                "issues_json": "[]",
                "raw_output_sha256": None,
                "element_index": element_index,
                "element_class": element_class,
                "modality": modality,
                "content_type": content_type,
                "text_content": str(element.get("text", "")),
                "binary_content": binary_content,
                "bbox_xyxy_norm_json": contract._canonical_json(bbox),
                "bbox_coordinate_space": contract.COORDINATE_SPACE,
            }
        )

    outcome = _page_outcome_row(
        source_path=source_path,
        native_page_number=native_page_number,
        outcome="parsed",
        element_count=len(projected),
        issues=[],
        raw_output_sha256=raw_sha256,
    )
    return [outcome, *projected]


def project_nrl_pages(data: object) -> pd.DataFrame:
    """Project an NRL page batch into page outcomes and ordered elements."""

    if not isinstance(data, pd.DataFrame):
        msg = f"projection input must be a pandas DataFrame, got {type(data).__name__}"
        raise TypeError(msg)
    rows: list[dict[str, Any]] = []
    for row in data.to_dict("records"):
        try:
            rows.extend(_project_page(row))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                _failed_page_row(
                    row,
                    {"kind": "projection_error", "type": type(exc).__name__, "message": str(exc)},
                    raw_output_sha256=_row_raw_output_sha256(row),
                )
            )
    return contract.validate_projection_envelope(pd.DataFrame(rows, columns=contract.PROJECTION_COLUMNS))


class NRLCuratorProjectionOperator(AbstractOperator, CPUOperator):
    """Terminal CPU operator producing Curator-neutral page records."""

    PRESERVE_PANDAS_OUTPUT = True

    def preprocess(self, data: object, **kwargs) -> object:  # noqa: ARG002
        return data

    def process(self, data: object, **kwargs) -> pd.DataFrame:  # noqa: ARG002
        return project_nrl_pages(data)

    def postprocess(self, data: object, **kwargs) -> object:  # noqa: ARG002
        return data


def build_projection_graph() -> Graph:
    """Build the existing NRL PDF graph plus one terminal CPU projection."""

    from nemo_retriever.operators.extract.parse.nemotron_parse import NEMOTRON_PARSE_DEFAULT_TASK_PROMPT

    if NEMOTRON_PARSE_DEFAULT_TASK_PROMPT != contract.PARSE_TASK_PROMPT:
        msg = "NRL's default Nemotron Parse prompt no longer matches the pinned v1.2 contract"
        raise RuntimeError(msg)

    extract_params = ExtractParams(
        method="nemotron_parse",
        extract_text=False,
        extract_images=False,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
        extract_page_as_image=True,
        use_page_elements=False,
        use_table_structure=False,
        dpi=200,
        image_format="png",
        render_mode="full_dpi",
        nemotron_parse_model=contract.PARSE_MODEL,
    )
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=extract_params,
        split_config={},
        stage_order=(),
    )
    return graph >> NRLCuratorProjectionOperator()


def _require_local_parse_resolution(resources: ClusterResources) -> None:
    """Fail closed unless NRL resolves Nemotron Parse to its local GPU actor."""

    from nemo_retriever.operators.extract.parse.nemotron_parse import (
        NemotronParseActor,
        NemotronParseGPUActor,
    )

    resolved = NemotronParseActor.resolve_operator_class(
        resources,
        operator_kwargs={
            "nemotron_parse_model": contract.PARSE_MODEL,
            "nemotron_parse_invoke_url": None,
            "invoke_url": None,
        },
    )
    if resolved is not NemotronParseGPUActor:
        msg = (
            "The NRL-to-Curator recipe requires a Ray runtime with a visible GPU so "
            "Nemotron Parse v1.2 resolves to the local GPU actor; remote CPU/NIM fallback is disabled"
        )
        raise RuntimeError(msg)


def _prepare_executor_for_local_parse(executor: RayDataExecutor) -> None:
    """Pin the Ray resource snapshot that resolves Nemotron Parse to the local GPU actor."""

    from nemo_retriever.common.ray_resource_hueristics import gather_cluster_resources
    from nemo_retriever.common.ray_runtime import ensure_local_ray_runtime
    from nemo_retriever.graph.executor import preflight_executors

    resources = gather_cluster_resources(ensure_local_ray_runtime())
    _require_local_parse_resolution(resources)
    preflight_executors([executor], resources)


def build_projection_executor(
    graph: Graph,
    *,
    projection_workers: int = contract.MAX_PROJECTION_WORKERS,
    projection_block_rows: int | None = None,
    parse_batch_size: int = contract.DEFAULT_PARSE_BATCH_SIZE,
    parse_cpus: int = contract.DEFAULT_PARSE_CPUS,
) -> RayDataExecutor:
    """Schedule one Parse actor and a bounded projection pool without changing model settings."""

    contract.validate_projection_workers(projection_workers)
    contract.validate_parse_scheduling(parse_batch_size, parse_cpus)
    contract.validate_projection_block_rows(projection_block_rows)
    projection_overrides: dict[str, Any] = {"concurrency": projection_workers, "num_cpus": 1}
    if projection_block_rows is not None:
        projection_overrides["target_num_rows_per_block"] = projection_block_rows
    return RayDataExecutor(
        graph,
        node_overrides={
            "NemotronParseActor": {"batch_size": parse_batch_size, "num_cpus": parse_cpus},
            NRLCuratorProjectionOperator.__name__: projection_overrides,
        },
        auto_concurrency_nodes={NRLCuratorProjectionOperator.__name__},
        source_cpu_reservation=1,
    )


def run_nrl_graph(
    paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    projection_workers: int = contract.MAX_PROJECTION_WORKERS,
    projection_block_rows: int | None = None,
    parse_batch_size: int = contract.DEFAULT_PARSE_BATCH_SIZE,
    parse_cpus: int = contract.DEFAULT_PARSE_CPUS,
) -> pd.DataFrame:
    """Run extraction once and return the validated projection envelope."""

    normalized_paths = [os.fspath(paths)] if isinstance(paths, (str, os.PathLike)) else [os.fspath(p) for p in paths]
    executor = build_projection_executor(
        build_projection_graph(),
        projection_workers=projection_workers,
        projection_block_rows=projection_block_rows,
        parse_batch_size=parse_batch_size,
        parse_cpus=parse_cpus,
    )
    if not normalized_paths:
        return pd.DataFrame(columns=contract.PROJECTION_COLUMNS, dtype=object)
    _prepare_executor_for_local_parse(executor)
    return contract.validate_projection_envelope(executor.ingest(normalized_paths))
