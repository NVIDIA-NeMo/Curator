# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reassemble extracted page ranges without shuffling raw rendered images."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import identity_digest
from nemo_curator.stages.interleaved.pdf.nemotron_parse.schema import PAGE_OUTPUT_SCHEMA, align_page_output
from nemo_curator.tasks import InterleavedBatch


def validate_document_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Check persisted page outcomes against actual extracted content rows."""
    metadata = [row for row in rows if row["modality"] == "metadata"]
    content = [row for row in rows if row["modality"] != "metadata"]
    if len(metadata) != 1 or metadata[0]["position"] != -1:
        msg = "Each document/range must have exactly one metadata row at position -1"
        raise ValueError(msg)
    info = json.loads(metadata[0]["text_content"])
    provenance = info["pdf_provenance"]
    outcomes = info["page_outcomes"]
    indexes = [outcome["page_number"] for outcome in outcomes]
    if (
        not indexes
        or any(isinstance(page, bool) or not isinstance(page, int) or page < 0 for page in indexes)
        or len(indexes) != len(set(indexes))
        or info["num_pages"] != len(indexes)
        or [row["position"] for row in content] != list(range(len(content)))
    ):
        msg = "Invalid page outcomes or document reading order"
        raise ValueError(msg)
    counts = Counter(row["page_number"] for row in content)
    if set(counts) - set(indexes):
        msg = "Extracted content contains an unexpected physical page"
        raise ValueError(msg)
    ranks = {page: rank for rank, page in enumerate(indexes)}
    content_ranks = [ranks[row["page_number"]] for row in content]
    if any(left > right for left, right in pairwise(content_ranks)):
        msg = "Extracted content does not follow the declared physical-page selection order"
        raise ValueError(msg)
    failed = {row["page_number"] for row in content if row.get("materialize_error")}
    for outcome in outcomes:
        page = outcome["page_number"]
        if (
            outcome["status"] not in {"success", "failed"}
            or isinstance(outcome["element_count"], bool)
            or not isinstance(outcome["element_count"], int)
            or outcome["element_count"] < 0
            or counts[page] != outcome["element_count"]
            or (outcome["status"] == "failed") != (page in failed)
        ):
            msg = f"Page outcome does not match delivered elements/failure for page {page}"
            raise ValueError(msg)
    for row in rows:
        if (
            row.get("document_id") != provenance["document_id"]
            or row["sample_id"] != metadata[0]["sample_id"]
            or row["pdf_name"] != metadata[0]["pdf_name"]
        ):
            msg = "Mixed document identities in extracted output"
            raise ValueError(msg)
    return info, content


@dataclass
class PDFDocumentMergeStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Group extracted ranges by document. Fan-in is not checkpoint resumable."""

    name: str = "pdf_document_merge"
    is_resumable: bool = False
    required_finalizer: ClassVar[str] = "pdf_page_budget_validation"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"group_by_metadata": "pdf_document_id"}

    def process(self, _task: InterleavedBatch) -> InterleavedBatch:
        msg = "PDFDocumentMergeStage requires grouped execution with RayDataExecutor"
        raise RuntimeError(msg)

    def process_batch(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        if len(tasks) == 0:
            return []
        ordered = sorted(tasks, key=lambda task: task._metadata["pdf_chunk"]["offset"])
        first = ordered[0]._metadata["pdf_chunk"]
        offset, content, outcomes = 0, [], []
        position = 0
        metadata_row, document_info = None, None
        for task in ordered:
            chunk = task._metadata["pdf_chunk"]
            if (
                any(
                    chunk[key] != first[key]
                    for key in (
                        "document_id",
                        "source_locator",
                        "source_version",
                        "selection_sha256",
                        "total_pages",
                        "sample_id",
                        "pdf_name",
                    )
                )
                or chunk["offset"] != offset
            ):
                msg = "PDF chunks must cover consecutive, nonoverlapping ranges of one versioned document"
                raise ValueError(msg)
            table = align_page_output(task.to_pyarrow())
            rows = table.drop(["binary_content"]).to_pylist()
            info, elements = validate_document_rows(rows)
            if [outcome["page_number"] for outcome in info["page_outcomes"]] != chunk["page_indices"]:
                msg = "Chunk outcomes do not match the selected physical pages"
                raise ValueError(msg)
            if document_info is None:
                document_info = info
                metadata_row = next(row for row in rows if row["modality"] == "metadata")
            elif info["pdf_provenance"] != document_info["pdf_provenance"]:
                msg = "Inconsistent source or parser versions across PDF chunks"
                raise ValueError(msg)
            element_table = table.filter(pc.not_equal(table["modality"], "metadata"))
            element_table = element_table.set_column(
                element_table.schema.get_field_index("position"),
                PAGE_OUTPUT_SCHEMA.field("position"),
                pa.array(range(position, position + len(elements)), type=pa.int32()),
            )
            content.append(element_table)
            position += len(elements)
            outcomes.extend(info["page_outcomes"])
            offset += len(chunk["page_indices"])
        indexes = [outcome["page_number"] for outcome in outcomes]
        if offset != first["total_pages"] or identity_digest(indexes) != first["selection_sha256"]:
            msg = "Incomplete or substituted PDF page selection; refusing to emit a partial document"
            raise ValueError(msg)
        document_info.update(num_pages=offset, page_outcomes=outcomes)
        metadata_row["text_content"] = json.dumps(document_info)
        metadata_table = pa.Table.from_pylist([metadata_row], schema=PAGE_OUTPUT_SCHEMA)
        merged = pa.concat_tables([metadata_table, *content])
        validate_document_rows(merged.drop(["binary_content"]).to_pylist())
        output_id = identity_digest([first["document_id"], first["selection_sha256"], document_info["pdf_provenance"]])
        self._log_metrics({"merged_chunks": float(len(ordered)), "merged_pages": float(offset)})
        return [
            InterleavedBatch(
                dataset_name=ordered[0].dataset_name,
                data=merged,
                _metadata={
                    **{
                        key: value
                        for key, value in ordered[0]._metadata.items()
                        if key not in {"pdf_page_chunks", "pdf_chunk", "partition_index"}
                    },
                    "source_files": [first["source_locator"]],
                    "output_file_id": output_id,
                },
                _stage_perf=[stat for task in ordered for stat in task._stage_perf],
            )
        ]
