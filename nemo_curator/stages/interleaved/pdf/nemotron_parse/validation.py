# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent manifest-to-Parquet reconciliation for bounded PDF experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar

import pyarrow.parquet as pq

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import (
    SHA256_HEX_LENGTH,
    document_identity,
    identity_digest,
    selected_pages,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import validate_document_rows
from nemo_curator.tasks import FileGroupTask


def _check_output_version(expected_output_version: str) -> None:
    if (
        not isinstance(expected_output_version, str)
        or len(expected_output_version) != SHA256_HEX_LENGTH
        or any(c not in "0123456789abcdef" for c in expected_output_version)
    ):
        msg = "Final validation requires the admitted expected_output_version SHA256"
        raise ValueError(msg)


def _expected_documents(manifest_path: str, max_pdfs: int | None, file_name_field: str) -> dict[str, dict[str, Any]]:
    expected, samples = {}, set()
    with open(manifest_path) as manifest:
        for line in manifest:
            if not line.strip():
                continue
            entry = json.loads(line)
            entry["file_name"] = entry[file_name_field]
            identity = document_identity(entry)
            key = identity["document_id"]
            sample = entry["file_name"].rsplit(".", 1)[0]
            if key in expected or sample in samples:
                msg = "Duplicate document or ambiguous sample_id in expected inventory"
                raise ValueError(msg)
            samples.add(sample)
            expected[key] = {**identity, "entry": entry, "pages": selected_pages(entry), "sample": sample}
            if max_pdfs is not None and len(expected) == max_pdfs:
                break
    return expected


def validate_page_budget_outputs(  # noqa: PLR0913
    manifest_path: str,
    output_paths: list[str],
    *,
    expected_output_version: str,
    max_pdfs: int | None = None,
    file_name_field: str = "file_name",
    url_field: str = "url",
) -> dict[str, Any]:
    """Reconcile all expected documents, even those with no surviving chunks.

    Read output columns without binary payloads. Coverage is a separate gate
    from model/extraction quality: retained failed pages are accounted for,
    never converted into successful pages.
    """
    _check_output_version(expected_output_version)
    expected = _expected_documents(manifest_path, max_pdfs, file_name_field)
    seen, failed, successful = set(), 0, 0
    if len(output_paths) != len(set(output_paths)):
        msg = "Duplicate output paths in delivered inventory"
        raise ValueError(msg)
    columns = [
        "sample_id",
        "url",
        "pdf_name",
        "document_id",
        "position",
        "modality",
        "page_number",
        "text_content",
        "materialize_error",
    ]
    for path in output_paths:
        rows = pq.read_table(path, columns=columns).to_pylist()
        info, _ = validate_document_rows(rows)
        provenance = info["pdf_provenance"]
        if provenance.get("output_version") != expected_output_version:
            msg = "Delivered PDF extraction version differs from the admitted expected_output_version"
            raise ValueError(msg)
        key = provenance["document_id"]
        if key not in expected or key in seen:
            msg = f"Unexpected or duplicate delivered PDF: {key}"
            raise ValueError(msg)
        wanted = expected[key]
        if (
            any(provenance[field] != wanted[field] for field in ("source_locator", "source_version"))
            or provenance["selection_sha256"] != identity_digest(wanted["pages"])
            or [outcome["page_number"] for outcome in info["page_outcomes"]] != wanted["pages"]
            or rows[0]["sample_id"] != wanted["sample"]
            or rows[0]["pdf_name"] != wanted["entry"]["file_name"]
            or info["url"] != wanted["entry"].get(url_field, "")
            or any(row["url"] != wanted["entry"].get(url_field, "") for row in rows)
        ):
            msg = f"Delivered PDF does not match original expected inventory: {key}"
            raise ValueError(msg)
        seen.add(key)
        failed += sum(outcome["status"] == "failed" for outcome in info["page_outcomes"])
        successful += sum(outcome["status"] == "success" for outcome in info["page_outcomes"])
    if seen != set(expected):
        msg = f"Missing whole PDFs in delivered inventory: {sorted(set(expected) - seen)!r}"
        raise ValueError(msg)
    return {
        "coverage_passed": True,
        "documents": len(expected),
        "pages": successful + failed,
        "successful_pages": successful,
        "failed_pages": failed,
        "all_pages_succeeded": failed == 0,
        "extraction_quality_validated": False,
        "output_version": expected_output_version,
    }


@dataclass
class PDFPageBudgetValidationStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Reconcile all document file references on the driver, including no results."""

    manifest_path: str
    expected_output_version: str
    max_pdfs: int | None = None
    file_name_field: str = "file_name"
    url_field: str = "url"
    name: str = "pdf_page_budget_validation"
    is_resumable: bool = False
    finalizes: ClassVar[str] = "pdf_page_budget_validation"

    def __post_init__(self) -> None:
        _check_output_version(self.expected_output_version)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"run_on_driver": True}

    def process(self, _task: FileGroupTask) -> FileGroupTask:
        msg = "PDFPageBudgetValidationStage requires driver validation with RayDataExecutor"
        raise RuntimeError(msg)

    def process_batch(self, tasks: list[FileGroupTask]) -> list[FileGroupTask]:
        result = validate_page_budget_outputs(
            self.manifest_path,
            [path for task in tasks for path in task.data],
            expected_output_version=self.expected_output_version,
            max_pdfs=self.max_pdfs,
            file_name_field=self.file_name_field,
            url_field=self.url_field,
        )
        self._log_metrics(
            {"validated_documents": float(result["documents"]), "validated_pages": float(result["pages"])}
        )
        for task in tasks:
            task._metadata["pdf_page_budget_validation"] = result
        return tasks
