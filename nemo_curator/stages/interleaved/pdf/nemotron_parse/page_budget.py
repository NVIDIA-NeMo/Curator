# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, serializable provenance contracts for opt-in PDF page budgets."""

from __future__ import annotations

import hashlib
import json
from typing import Any

PAGE_BUDGET_SCHEMA = "pdf-page-budget.v1"
EXTRACTION_CONFIG_SCHEMA = "pdf-page-extraction.v1"
SHA256_HEX_LENGTH = hashlib.sha256().digest_size * 2


class ExtractionConfigMismatchError(ValueError):
    """An admitted extraction contract differs from effective runtime settings."""


def validate_extraction_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Copy and check a resolved contract; task layout is not part of it."""
    keys = {"schema", "parser", "model", "request", "render", "transport", "projection"}
    if not isinstance(config, dict) or set(config) != keys or config["schema"] != EXTRACTION_CONFIG_SCHEMA:
        msg = "Page budgeting requires a resolved pdf-page-extraction.v1 configuration"
        raise ValueError(msg)
    for key in keys - {"schema"}:
        if not isinstance(config[key], dict) or not config[key]:
            msg = f"Missing extraction configuration section: {key}"
            raise ValueError(msg)
    for section, key in (("parser", "source_sha256"), ("model", "artifacts_sha256")):
        value = config[section].get(key)
        if (
            not isinstance(value, str)
            or len(value) != SHA256_HEX_LENGTH
            or any(c not in "0123456789abcdef" for c in value)
        ):
            msg = f"Extraction configuration requires {section}.{key}"
            raise ValueError(msg)
    if not config["parser"].get("version") or not config["model"].get("revision"):
        msg = "Extraction configuration requires parser version and immutable model revision"
        raise ValueError(msg)
    request = config["request"]
    if set(request) != {"model", "stage", "prompt", "generation"} or request["stage"] != "fullpage":
        msg = "Page budgeting requires the resolved served single-mode request"
        raise ValueError(msg)
    if not request["model"] or not request["prompt"] or not isinstance(request["generation"], dict):
        msg = "Extraction request must include model, prompt and generation settings"
        raise ValueError(msg)
    return json.loads(json.dumps(config, sort_keys=True, allow_nan=False))


def identity_digest(value: Any) -> str:  # noqa: ANN401
    """Hash canonical JSON, independent of dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def document_identity(entry: dict[str, Any]) -> dict[str, str]:
    """Require an explicit canonical locator and immutable source version.

    A caller must supply a version that identifies the actual source bytes
    (for example their SHA256). URL, task size and page-count metadata are not
    identities: changing any of those must not rename the document.
    """
    source = {}
    for key in ("source_locator", "source_version"):
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            msg = f"Page budgeting requires a nonempty {key} for {entry.get('file_name')!r}"
            raise ValueError(msg)
        source[key] = value
    return {**source, "document_id": identity_digest(source)}


def selected_pages(entry: dict[str, Any]) -> list[int]:
    """Resolve authoritative page selections without opening or rendering PDFs."""
    count = entry.get("num_pages")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
        msg = f"num_pages must be a positive integer for {entry.get('file_name')!r}"
        raise ValueError(msg)
    selection = entry.get("selected_page_indices")
    if selection is not None:
        if (
            not isinstance(selection, list)
            or not selection
            or any(isinstance(page, bool) or not isinstance(page, int) or page < 0 for page in selection)
            or len(selection) != len(set(selection))
            or (count is not None and max(selection) >= count)
        ):
            msg = f"Invalid selected_page_indices for {entry.get('file_name')!r}"
            raise ValueError(msg)
        return list(selection)
    if count is None:
        msg = f"Page budgeting requires num_pages or selected_page_indices for {entry.get('file_name')!r}"
        raise ValueError(msg)
    return list(range(count))


def validate_chunk_pages(chunk: dict[str, Any], pages: list[dict[str, Any]]) -> None:
    """Verify actual physical pages against a chunk, rather than its row count."""
    expected = chunk["page_indices"]
    actual = [page.get("page_number") for page in pages]
    if len(actual) != len(expected) or set(actual) != set(expected):
        msg = f"Missing, duplicate or unexpected physical pages for {chunk['document_id']}: {actual!r}"
        raise ValueError(msg)
    for page in pages:
        if (
            page.get("document_id") != chunk["document_id"]
            or page.get("pdf_name") != chunk["pdf_name"]
            or page.get("sample_id") != chunk["sample_id"]
        ):
            msg = f"Page provenance does not match document {chunk['document_id']}"
            raise ValueError(msg)


def output_version(metadata: dict[str, Any]) -> str:
    """Identify effective served semantics, independent of ignored/layout knobs."""
    config = validate_extraction_config(metadata.get("pdf_extraction_config"))
    return identity_digest({"schema": PAGE_BUDGET_SCHEMA, "extraction": config})
