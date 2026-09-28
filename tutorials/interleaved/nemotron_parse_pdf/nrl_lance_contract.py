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

"""Publish NRL Nemotron Parse output for native NeMo Curator consumption.

Run ``ingest`` in a NeMo Retriever Library environment and ``consume`` in a
NeMo Curator environment. A Lance dataset is the only process boundary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import pyarrow as pa

PARSE_MODEL = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
PARSE_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
DEFAULT_PARSE_BATCH_SIZE = 64
DEFAULT_PARSE_CPUS = 1
MAX_PROJECTION_WORKERS = 8
ELEMENT_TABLE = "pdf_elements"
RUN_STATE_FILE = "run_state.json"
HANDOFF_MANIFEST_FILE = "handoff_manifest.json"
CONSUME_REPORT_FILE = "consume_validation.json"
COMPLETION_MANIFEST_FILE = "completion_manifest.json"
COORDINATE_SPACE = "normalized_1664x2048_padded_canvas"
_HASH_BLOCK_BYTES = 8 * 1024 * 1024
_HANDOFF_HASH_FIELD = "handoff_sha256"
_REPORT_HASH_FIELD = "report_sha256"
_COMPLETION_HASH_FIELD = "completion_sha256"
PUBLICATION_POLICY = "validated_pages_v1"
_PUBLISHABLE_STATUSES = frozenset({"success", "valid_blank", "partial"})
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PAGE_OUTCOMES = frozenset({"parsed", "empty", "failed"})
_RUN_ID_RE = re.compile(r"(?!\.{1,2}$)[A-Za-z0-9._-]+")

PROJECTION_COLUMNS = (
    "record_type",
    "source_path",
    "native_page_number",
    "page_outcome",
    "element_count",
    "issues_json",
    "raw_output_sha256",
    "element_index",
    "element_class",
    "modality",
    "content_type",
    "text_content",
    "binary_content",
    "bbox_xyxy_norm_json",
    "bbox_coordinate_space",
)


def validate_parse_scheduling(parse_batch_size: int, parse_cpus: int) -> None:
    """Validate recipe-only Parse scheduling without importing the NRL runtime."""

    for name, value in (("parse_batch_size", parse_batch_size), ("parse_cpus", parse_cpus)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            msg = f"{name} must be a positive integer"
            raise ValueError(msg)
    if parse_batch_size == 1:
        msg = "parse_batch_size must be at least 2; the NRL executor promotes batch size 1 to 64"
        raise ValueError(msg)


def validate_projection_block_rows(value: int | None) -> None:
    """Validate optional page-block splitting before the CPU projection pool."""

    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = "projection_block_rows must be a positive integer or None"
        raise ValueError(msg)


def validate_projection_workers(value: int) -> None:
    """Validate the CPU projection actor-pool ceiling."""

    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_PROJECTION_WORKERS:
        msg = f"projection_workers must be an integer from 1 through {MAX_PROJECTION_WORKERS}"
        raise ValueError(msg)


def validate_run_id(run_id: str) -> None:
    """Require a run ID that names exactly one new child of the output root."""

    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        msg = "run-id may contain only letters, numbers, '.', '_', and '-', and cannot be '.' or '..'"
        raise ValueError(msg)


@dataclass
class DocumentBuild:
    """Validated publication result for one canonical PDF."""

    status: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    page_count: int = 0
    blank_page_count: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)
    page_outcomes: list[dict[str, Any]] = field(default_factory=list)


class MarkerDurabilityUnconfirmedError(OSError):
    """A marker is visible and authoritative, but its directory sync failed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Marker is visible at {path}; durability is unconfirmed. Do not overwrite or retry this run."
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    msg = f"Object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return _canonical_json(payload).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(_HASH_BLOCK_BYTES):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a marker atomically without replacing an existing marker."""

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        try:
            _fsync_directory(path.parent)
        except OSError as exc:
            raise MarkerDurabilityUnconfirmedError(path) from exc
    finally:
        # Cleanup must not disguise either a publication failure or visibility.
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        msg = f"Expected a JSON object in {path}"
        raise TypeError(msg)
    return payload


def _seal_payload(payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    if hash_field in payload:
        msg = f"payload already contains reserved field {hash_field!r}"
        raise ValueError(msg)
    sealed = copy.deepcopy(dict(payload))
    sealed[hash_field] = _sha256_bytes(_canonical_json_bytes(sealed))
    return sealed


def _verify_sealed_payload(payload: Mapping[str, Any], hash_field: str, *, label: str) -> str:
    declared = payload.get(hash_field)
    if not isinstance(declared, str) or _SHA256_RE.fullmatch(declared) is None:
        msg = f"{label} has an invalid {hash_field}"
        raise ValueError(msg)
    core = {key: value for key, value in payload.items() if key != hash_field}
    actual = _sha256_bytes(_canonical_json_bytes(core))
    if actual != declared:
        msg = f"{label} SHA-256 is {actual}; expected {declared}"
        raise ValueError(msg)
    return actual


def _load_sealed_json(path: Path, hash_field: str, *, label: str) -> dict[str, Any]:
    payload = _load_json(path)
    _verify_sealed_payload(payload, hash_field, label=label)
    return payload


def _confirm_marker_durability(path: Path, hash_field: str, *, expected_sha256: str) -> dict[str, Any]:
    """Verify an existing sealed marker and sync it without replacing its bytes."""

    with path.open("rb") as stream:
        payload = json.load(stream)
        digest = _verify_sealed_payload(payload, hash_field, label=str(path))
        if digest != expected_sha256:
            msg = f"Marker at {path} differs from the expected SHA-256"
            raise ValueError(msg)
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return payload


def _paths_overlap(first: Path, second: Path) -> bool:
    return first.is_relative_to(second) or second.is_relative_to(first)


def _pdf_page_count(path: Path) -> int:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        count = len(document)
    finally:
        document.close()
    if count <= 0:
        msg = "PDF contains no pages"
        raise ValueError(msg)
    return count


def _source_records_from_directory(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(path.resolve() for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    return [{"path": str(path), "url": None, "valid_blank_pages": []} for path in paths]


def _source_records_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("path"), str):
                msg = f"{manifest_path}:{line_number} must contain an object with a string 'path'"
                raise ValueError(msg)  # noqa: TRY004
            path = Path(value["path"]).expanduser()
            if not path.is_absolute():
                path = manifest_path.parent / path
            path = path.resolve()
            blank_pages = value.get("valid_blank_pages", [])
            if not isinstance(blank_pages, list) or any(
                isinstance(page, bool) or not isinstance(page, int) or page < 0 for page in blank_pages
            ):
                msg = f"{manifest_path}:{line_number} valid_blank_pages must be a list of zero-based integers"
                raise ValueError(msg)
            records.append(
                {
                    "path": str(path),
                    "url": str(value["url"]) if value.get("url") is not None else None,
                    "valid_blank_pages": sorted(set(blank_pages)),
                }
            )
    return records


def inventory_sources(  # noqa: C901, PLR0915
    source_records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hash, preflight, and exact-deduplicate input PDFs."""

    inputs: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    representative_by_hash: dict[str, dict[str, Any]] = {}

    for input_index, source in enumerate(source_records):
        path = Path(str(source["path"])).resolve()
        entry: dict[str, Any] = {
            "input_index": input_index,
            "path": str(path),
            "url": source.get("url"),
            "valid_blank_pages": sorted(set(source.get("valid_blank_pages", []))),
            "content_sha256": None,
            "size_bytes": None,
            "expected_page_count": None,
            "preflight_error": None,
            "representative_input_index": None,
            "representative_path": None,
            "status": "pending",
            "representative_status": None,
            "publication_status": "unpublished",
        }
        try:
            if path.suffix.lower() != ".pdf":
                msg = "input does not have a .pdf extension"
                raise ValueError(msg)  # noqa: TRY301
            digest, size = _sha256_file(path)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "failed"
            entry["preflight_error"] = {"type": type(exc).__name__, "message": str(exc)}
            inputs.append(entry)
            continue

        entry["content_sha256"] = digest
        entry["size_bytes"] = size
        representative = representative_by_hash.get(digest)
        if representative is None:
            entry["representative_input_index"] = input_index
            entry["representative_path"] = str(path)
            try:
                entry["expected_page_count"] = _pdf_page_count(path)
            except Exception as exc:  # noqa: BLE001
                entry["preflight_error"] = {"type": type(exc).__name__, "message": str(exc)}
                entry["status"] = "failed"
            representative_by_hash[digest] = entry
            representatives.append(entry)
        else:
            entry["representative_input_index"] = representative["input_index"]
            entry["representative_path"] = representative["path"]
            entry["expected_page_count"] = representative["expected_page_count"]
            entry["preflight_error"] = copy.deepcopy(representative["preflight_error"])
            entry["status"] = "duplicate"
        inputs.append(entry)

    by_hash: dict[str, list[dict[str, Any]]] = {}
    for entry in inputs:
        digest = entry.get("content_sha256")
        if isinstance(digest, str):
            by_hash.setdefault(digest, []).append(entry)

    for representative in representatives:
        aliases = by_hash[str(representative["content_sha256"])]
        representative["aliases"] = [
            {
                "path": alias["path"],
                "url": alias.get("url"),
                "input_index": alias["input_index"],
                "valid_blank_pages": list(alias["valid_blank_pages"]),
            }
            for alias in aliases
        ]
        declared_blank_pages = sorted({page for alias in aliases for page in alias["valid_blank_pages"]})
        representative["document_valid_blank_pages"] = declared_blank_pages
        expected_pages = representative.get("expected_page_count")
        if isinstance(expected_pages, int) and any(page >= expected_pages for page in declared_blank_pages):
            representative["preflight_error"] = {
                "type": "InvalidBlankPageDeclaration",
                "message": (f"declared blank page is outside zero-based page range 0..{expected_pages - 1}"),
            }
            representative["status"] = "failed"
        for alias in aliases:
            alias["preflight_error"] = copy.deepcopy(representative["preflight_error"])
            if alias["status"] == "duplicate":
                alias["representative_status"] = representative["status"]
    return inputs, representatives


def _rehash_source_inventory(inputs: Sequence[dict[str, Any]]) -> int:
    validated = 0
    for entry in inputs:
        expected_digest = entry.get("content_sha256")
        expected_size = entry.get("size_bytes")
        if expected_digest is None:
            continue
        if not isinstance(expected_digest, str) or _SHA256_RE.fullmatch(expected_digest) is None:
            msg = f"input {entry['input_index']} has an invalid inventory SHA-256"
            raise ValueError(msg)
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            msg = f"input {entry['input_index']} has an invalid inventory size"
            raise ValueError(msg)
        path = Path(str(entry["path"]))
        actual_digest, actual_size = _sha256_file(path)
        if actual_digest != expected_digest or actual_size != expected_size:
            msg = f"source input changed after inventory: {path} (sha256={actual_digest}, size={actual_size})"
            raise RuntimeError(msg)
        validated += 1
    return validated


def _canonical_issues(value: object) -> list[dict[str, Any]]:
    if value is None or value == "":
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        msg = "issues_json must encode an array of objects"
        raise ValueError(msg)
    return [dict(item) for item in parsed]


def _normalize_bbox(value: object) -> list[float] | None:  # noqa: PLR0911
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:  # noqa: PLR2004
        return None
    if any(isinstance(coordinate, bool) for coordinate in value):
        return None
    try:
        bbox = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0 for coordinate in bbox):
        return None
    left, top, right, bottom = bbox
    if left >= right or top >= bottom:
        return None
    return bbox


def _is_missing(value: object) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _normalize_integer(value: object) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return value
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError):
        return value
    return normalized if normalized == value else value


def _normalize_json_field(value: object, *, field: str) -> str | None:
    if _is_missing(value):
        return None
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            msg = f"{field} must contain valid JSON"
            raise ValueError(msg) from exc
    return _canonical_json(decoded)


def normalize_projection_envelope(data: object) -> pd.DataFrame:
    """Return an object-backed DataFrame with the canonical column order."""

    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    unknown = sorted(set(frame.columns) - set(PROJECTION_COLUMNS))
    if unknown:
        msg = f"projection envelope has unknown columns: {unknown}"
        raise ValueError(msg)

    records = []
    for source in frame.to_dict("records"):
        record = {column: source.get(column) for column in PROJECTION_COLUMNS}
        record = {column: None if _is_missing(value) else value for column, value in record.items()}
        if isinstance(record["source_path"], os.PathLike):
            record["source_path"] = os.fspath(record["source_path"])
        for column in ("native_page_number", "element_count", "element_index"):
            record[column] = _normalize_integer(record[column])
        if isinstance(record["binary_content"], (bytearray, memoryview)):
            record["binary_content"] = bytes(record["binary_content"])
        for column in ("issues_json", "bbox_xyxy_norm_json"):
            record[column] = _normalize_json_field(record[column], field=column)
        records.append(record)
    return pd.DataFrame(records, columns=list(PROJECTION_COLUMNS), dtype=object)


def _decoded_json_array(value: object, *, field: str, row_index: int) -> list[Any]:
    if not isinstance(value, str):
        msg = f"row {row_index}: {field} must be a JSON string"
        raise ValueError(msg)  # noqa: TRY004
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        msg = f"row {row_index}: {field} must encode a JSON array"
        raise ValueError(msg)  # noqa: TRY004
    return decoded


def _validated_bbox(value: object, *, row_index: int) -> list[float]:
    decoded = _decoded_json_array(value, field="bbox_xyxy_norm_json", row_index=row_index)
    if any(isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)) for coordinate in decoded):
        msg = f"row {row_index}: bbox coordinates must be JSON numbers"
        raise ValueError(msg)
    bbox = _normalize_bbox(decoded)
    if bbox is None:
        msg = f"row {row_index}: bbox must be four finite ordered floats in [0, 1]"
        raise ValueError(msg)
    return bbox


def validate_projection_envelope(data: object) -> pd.DataFrame:  # noqa: C901, PLR0912, PLR0915
    """Normalize and fail closed on any malformed projection record."""

    frame = normalize_projection_envelope(data)
    outcomes: dict[tuple[str, int], tuple[int, str]] = {}
    elements: dict[tuple[str, int], list[int]] = {}

    for index, row in frame.iterrows():
        record_type = row["record_type"]
        source_path = row["source_path"]
        native_page_number = row["native_page_number"]
        if record_type not in {"page_outcome", "element"}:
            msg = f"row {index}: unsupported record_type {record_type!r}"
            raise ValueError(msg)
        if not isinstance(source_path, str) or not source_path:
            msg = f"row {index}: source_path must be non-empty text"
            raise ValueError(msg)
        if isinstance(native_page_number, bool) or not isinstance(native_page_number, int) or native_page_number < 0:
            msg = f"row {index}: native_page_number must be a non-negative integer"
            raise ValueError(msg)

        issues = _decoded_json_array(row["issues_json"], field="issues_json", row_index=index)
        if any(
            not isinstance(issue, Mapping) or not isinstance(issue.get("kind"), str) or not issue["kind"]
            for issue in issues
        ):
            msg = f"row {index}: every issue must be an object with a non-empty kind"
            raise ValueError(msg)
        key = (source_path, native_page_number)
        if record_type == "page_outcome":
            if key in outcomes:
                msg = f"duplicate page outcome for {source_path!r} page {native_page_number}"
                raise ValueError(msg)
            outcome = row["page_outcome"]
            element_count = row["element_count"]
            if outcome not in _PAGE_OUTCOMES:
                msg = f"row {index}: unsupported page_outcome {outcome!r}"
                raise ValueError(msg)
            if isinstance(element_count, bool) or not isinstance(element_count, int) or element_count < 0:
                msg = f"row {index}: element_count must be a non-negative integer"
                raise ValueError(msg)
            if native_page_number == 0 and outcome != "failed":
                msg = f"row {index}: native page 0 is reserved for document/split failures"
                raise ValueError(msg)
            if outcome == "parsed" and element_count == 0:
                msg = f"row {index}: parsed pages must contain at least one element"
                raise ValueError(msg)
            if outcome != "parsed" and element_count != 0:
                msg = f"row {index}: non-parsed pages cannot declare elements"
                raise ValueError(msg)
            if (outcome == "failed") != bool(issues):
                msg = f"row {index}: only failed outcomes may carry issues"
                raise ValueError(msg)
            raw_sha256 = row["raw_output_sha256"]
            if raw_sha256 is not None and (
                not isinstance(raw_sha256, str) or _SHA256_RE.fullmatch(raw_sha256) is None
            ):
                msg = f"row {index}: raw_output_sha256 must be a lowercase SHA-256"
                raise ValueError(msg)
            if outcome in {"parsed", "empty"} and raw_sha256 is None:
                msg = f"row {index}: successful model outcomes require raw_output_sha256"
                raise ValueError(msg)
            for column in (
                "element_index",
                "element_class",
                "modality",
                "content_type",
                "text_content",
                "binary_content",
                "bbox_xyxy_norm_json",
                "bbox_coordinate_space",
            ):
                if row[column] is not None:
                    msg = f"row {index}: page outcome field {column} must be null"
                    raise ValueError(msg)
            outcomes[key] = (element_count, outcome)
            continue

        if native_page_number == 0:
            msg = f"row {index}: element rows require a positive native page number"
            raise ValueError(msg)
        for column in ("page_outcome", "element_count", "raw_output_sha256"):
            if row[column] is not None:
                msg = f"row {index}: element field {column} must be null"
                raise ValueError(msg)
        if issues:
            msg = f"row {index}: element rows cannot carry issues"
            raise ValueError(msg)
        element_index = row["element_index"]
        if isinstance(element_index, bool) or not isinstance(element_index, int) or element_index < 0:
            msg = f"row {index}: element_index must be a non-negative integer"
            raise ValueError(msg)
        element_class = row["element_class"]
        if not isinstance(element_class, str) or not element_class:
            msg = f"row {index}: element_class must be non-empty text"
            raise ValueError(msg)
        expected_modality = "image" if element_class == "Picture" else "table" if element_class == "Table" else "text"
        expected_content_type = "image/png" if element_class == "Picture" else "text/markdown"
        if row["modality"] != expected_modality or row["content_type"] != expected_content_type:
            msg = f"row {index}: modality/content_type do not match {element_class!r}"
            raise ValueError(msg)
        if not isinstance(row["text_content"], str):
            msg = f"row {index}: text_content must be text, including for textless pictures"
            raise ValueError(msg)  # noqa: TRY004
        if element_class == "Picture":
            if not isinstance(row["binary_content"], bytes) or not row["binary_content"].startswith(_PNG_SIGNATURE):
                msg = f"row {index}: Picture must carry inline PNG bytes"
                raise ValueError(msg)
        elif row["binary_content"] is not None:
            msg = f"row {index}: only Picture may carry binary_content"
            raise ValueError(msg)
        _validated_bbox(row["bbox_xyxy_norm_json"], row_index=index)
        if row["bbox_coordinate_space"] != COORDINATE_SPACE:
            msg = f"row {index}: unexpected bbox coordinate space"
            raise ValueError(msg)
        elements.setdefault(key, []).append(element_index)

    for key, indexes in elements.items():
        if key not in outcomes:
            msg = f"elements have no page outcome for {key[0]!r} page {key[1]}"
            raise ValueError(msg)
        declared_count, outcome = outcomes[key]
        if outcome != "parsed":
            msg = f"non-parsed page {key[0]!r} page {key[1]} has elements"
            raise ValueError(msg)
        if sorted(indexes) != list(range(len(indexes))):
            msg = f"element indexes are not contiguous for {key[0]!r} page {key[1]}"
            raise ValueError(msg)
        if declared_count != len(indexes):
            msg = f"element count mismatch for {key[0]!r} page {key[1]}"
            raise ValueError(msg)
    for key, (declared_count, _outcome) in outcomes.items():
        if declared_count and key not in elements:
            msg = f"declared elements are missing for {key[0]!r} page {key[1]}"
            raise ValueError(msg)
    return frame


def _base_element_row(  # noqa: PLR0913
    document: Mapping[str, Any],
    *,
    run_id: str,
    position: int,
    modality: str,
    content_type: str,
    text_content: str | None,
    binary_content: bytes | None,
    page_number: int | None,
    element_class: str | None,
    bbox: list[float] | None,
) -> dict[str, Any]:
    aliases = list(document["aliases"])
    primary_url = next((alias["url"] for alias in aliases if alias.get("url")), None)
    return {
        "sample_id": document["content_sha256"],
        "position": position,
        "modality": modality,
        "content_type": content_type,
        "text_content": text_content,
        "binary_content": binary_content,
        "source_ref": None,
        "materialize_error": None,
        "url": primary_url,
        "page_number": page_number,
        "pdf_name": Path(str(document["path"])).name,
        "element_class": element_class,
        "source_path": document["path"],
        "source_aliases": json.dumps(aliases, sort_keys=True, separators=(",", ":")),
        "content_sha256": document["content_sha256"],
        "bbox_xyxy_norm": bbox,
        "bbox_coordinate_space": COORDINATE_SPACE if bbox is not None else None,
        "run_id": run_id,
    }


def validate_page_outcomes(
    page_outcomes: Sequence[Mapping[str, Any]],
    *,
    expected_page_count: int,
    extraction_status: str,
    issues: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Validate explicit coverage independently of whether its rows were delivered."""

    if (
        isinstance(expected_page_count, bool)
        or not isinstance(expected_page_count, int)
        or expected_page_count <= 0
        or not isinstance(page_outcomes, list)
        or len(page_outcomes) != expected_page_count
        or not isinstance(issues, list)
        or any(not isinstance(issue, Mapping) for issue in issues)
    ):
        msg = "Invalid document page coverage"
        raise ValueError(msg)
    counts = dict.fromkeys(
        (
            "validated_page_count",
            "content_page_count",
            "blank_page_count",
            "failed_page_count",
            "content_element_count",
        ),
        0,
    )
    for number, page in enumerate(page_outcomes):
        if (
            not isinstance(page, Mapping)
            or isinstance(page.get("page_number"), bool)
            or not isinstance(page.get("page_number"), int)
            or page["page_number"] != number
        ):
            msg = "Page outcomes must enumerate every expected zero-based page exactly once"
            raise ValueError(msg)
        status, count, page_issues = page.get("status"), page.get("element_count"), page.get("issues")
        if (
            not isinstance(status, str)
            or status not in {"success", "valid_blank", "failed"}
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or not isinstance(page_issues, list)
            or any(not isinstance(issue, Mapping) for issue in page_issues)
            or bool(page_issues) != (status == "failed")
            or (count > 0) != (status == "success")
        ):
            msg = f"Invalid page outcome for page {number}"
            raise ValueError(msg)
        counts["content_element_count"] += count
        counts["failed_page_count"] += status == "failed"
        counts["blank_page_count"] += status == "valid_blank"
        counts["content_page_count"] += status == "success"
        counts["validated_page_count"] += status != "failed"
    if counts["failed_page_count"] and not issues:
        msg = "Failed pages require document issues"
        raise ValueError(msg)
    expected_status = (
        "failed"
        if not counts["validated_page_count"]
        else "partial"
        if issues
        else "success"
        if counts["content_page_count"]
        else "valid_blank"
    )
    if extraction_status != expected_status:
        msg = f"Extraction status {extraction_status!r} differs from page coverage {expected_status!r}"
        raise ValueError(msg)
    return counts


def build_document_rows(  # noqa: C901, PLR0912, PLR0915
    document: Mapping[str, Any],
    envelope_rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
) -> DocumentBuild:
    """Publish only whole validated pages, retaining explicit incomplete coverage.

    ``envelope_rows`` are one document's records from :func:`validate_projection_envelope`,
    which already rejects malformed records and inconsistent pages. This gate decides
    document coverage: expected pages, declared blanks, and the extraction status.
    """

    expected_page_count = document.get("expected_page_count")
    if isinstance(expected_page_count, bool) or not isinstance(expected_page_count, int) or expected_page_count <= 0:
        return DocumentBuild(
            status="failed",
            issues=[{"kind": "invalid_expected_page_count", "value": expected_page_count}],
        )

    valid_blank_pages = set(document.get("document_valid_blank_pages", []))
    markers: dict[int, list[Mapping[str, Any]]] = {}
    elements: dict[int, list[Mapping[str, Any]]] = {}
    for row in envelope_rows:
        records = markers if row["record_type"] == "page_outcome" else elements
        records.setdefault(row["native_page_number"], []).append(row)

    issues: list[dict[str, Any]] = []
    if 0 in markers:
        issues.append({"kind": "document_or_split_failure", "native_page_number": 0})
    expected_pages = set(range(1, expected_page_count + 1))
    actual_positive_pages = {page for page in markers if page > 0}
    missing_pages = sorted(expected_pages - actual_positive_pages)
    unexpected_pages = sorted(actual_positive_pages - expected_pages)
    if missing_pages:
        issues.append({"kind": "missing_pages", "native_page_numbers": missing_pages})
    if unexpected_pages:
        issues.append({"kind": "unexpected_pages", "native_page_numbers": unexpected_pages})
    duplicate_pages = sorted(page for page, values in markers.items() if len(values) != 1)
    if duplicate_pages:
        issues.append({"kind": "duplicate_page_outcomes", "native_page_numbers": duplicate_pages})

    blank_pages: set[int] = set()
    content_pages: set[int] = set()
    for native_page in sorted(expected_pages & set(markers)):
        if len(markers[native_page]) != 1:
            continue
        marker = markers[native_page][0]
        issues.extend(
            {**issue, "native_page_number": native_page} for issue in _canonical_issues(marker["issues_json"])
        )
        if marker["page_outcome"] == "failed":
            issues.append({"kind": "page_failed", "native_page_number": native_page})
        elif marker["page_outcome"] == "parsed":
            content_pages.add(native_page)
        elif native_page - 1 in valid_blank_pages:
            blank_pages.add(native_page)
        else:
            issues.append({"kind": "unexpected_empty_output", "page_number": native_page - 1})

    # A page is atomic: elements of a rejected page never reach the delivered document.
    ordered_elements = sorted(
        (element for page in content_pages for element in elements[page]),
        key=lambda element: (element["native_page_number"], element["element_index"]),
    )
    page_issues: dict[int, list[dict[str, Any]]] = {page: [] for page in expected_pages}
    for issue in issues:
        native_page = issue.get("native_page_number")
        if native_page is not None:
            affected = [native_page]
        elif isinstance(issue.get("native_page_numbers"), list):
            affected = issue["native_page_numbers"]
        elif isinstance(issue.get("page_number"), int):
            affected = [issue["page_number"] + 1]
        else:
            affected = []
        for native_page in affected:
            if isinstance(native_page, int) and not isinstance(native_page, bool) and native_page in page_issues:
                scoped = {key: value for key, value in issue.items() if key != "native_page_numbers"}
                page_issues[native_page].append({**scoped, "native_page_number": native_page})
    page_outcomes = [
        {
            "page_number": page - 1,
            "status": "success" if page in content_pages else "valid_blank" if page in blank_pages else "failed",
            "element_count": len(elements[page]) if page in content_pages else 0,
            "issues": page_issues[page],
        }
        for page in sorted(expected_pages)
    ]
    status = (
        "failed"
        if not content_pages and not blank_pages
        else "partial"
        if issues
        else "success"
        if content_pages
        else "valid_blank"
    )
    validate_page_outcomes(
        page_outcomes, expected_page_count=expected_page_count, extraction_status=status, issues=issues
    )
    if status == "failed":
        return DocumentBuild(
            status=status,
            issues=issues,
            page_outcomes=page_outcomes,
        )

    metadata_payload = {
        "content_sha256": document["content_sha256"],
        "pdf_name": Path(str(document["path"])).name,
        "num_pages": expected_page_count,
        "source_path": document["path"],
        "source_aliases": list(document["aliases"]),
        "url": next((alias["url"] for alias in document["aliases"] if alias.get("url")), None),
        "valid_blank_pages": sorted(valid_blank_pages),
        "extraction_status": status,
        "page_outcomes": page_outcomes,
        "issues": issues,
    }
    rows = [
        _base_element_row(
            document,
            run_id=run_id,
            position=-1,
            modality="metadata",
            content_type="application/json",
            text_content=json.dumps(metadata_payload, sort_keys=True, separators=(",", ":")),
            binary_content=None,
            page_number=None,
            element_class=None,
            bbox=None,
        )
    ]
    for position, element in enumerate(ordered_elements):
        rows.append(
            _base_element_row(
                document,
                run_id=run_id,
                position=position,
                modality=str(element["modality"]),
                content_type=str(element["content_type"]),
                text_content=str(element["text_content"]),
                binary_content=element["binary_content"],
                page_number=element["native_page_number"] - 1,
                element_class=str(element["element_class"]),
                bbox=_normalize_bbox(element["bbox_xyxy_norm_json"]),
            )
        )

    return DocumentBuild(
        status=status,
        rows=rows,
        page_count=len(content_pages) + len(blank_pages),
        blank_page_count=len(blank_pages),
        issues=issues,
        page_outcomes=page_outcomes,
    )


def element_schema() -> pa.Schema:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
            pa.field("modality", pa.string(), nullable=False),
            pa.field("content_type", pa.string()),
            pa.field("text_content", pa.string()),
            pa.field("binary_content", pa.large_binary()),
            pa.field("source_ref", pa.string()),
            pa.field("materialize_error", pa.string()),
            pa.field("url", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("pdf_name", pa.string()),
            pa.field("element_class", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("source_aliases", pa.string()),
            pa.field("content_sha256", pa.string(), nullable=False),
            pa.field("bbox_xyxy_norm", pa.list_(pa.float64())),
            pa.field("bbox_coordinate_space", pa.string()),
            pa.field("run_id", pa.string(), nullable=False),
        ]
    )


def _table_names(connection: object) -> set[str]:
    names = connection.list_tables()
    return set(names.tables if hasattr(names, "tables") else names)


class ElementTableWriter:
    """Append exactly one validated document per Lance write with NRL's LanceDB."""

    def __init__(self, uri: Path) -> None:
        import lancedb

        self.uri = uri
        self.connection = lancedb.connect(str(uri))
        if ELEMENT_TABLE in _table_names(self.connection):
            msg = f"Refusing to append to existing table {ELEMENT_TABLE!r} at {uri}"
            raise FileExistsError(msg)
        self.table: Any | None = None

    def add_document(self, rows: Sequence[dict[str, Any]]) -> None:
        import pyarrow as pa

        document = pa.Table.from_pylist(list(rows), schema=element_schema())
        if self.table is None:
            self.table = self.connection.create_table(ELEMENT_TABLE, data=document, mode="create")
        else:
            self.table.add(document, mode="append")


def _table_path(uri: Path, table_name: str = ELEMENT_TABLE) -> Path:
    return uri / f"{table_name}.lance"


def read_element_table(path: Path, *, version: int | None = None) -> tuple[pa.Schema, pa.Table, int]:
    """Return the stored schema, every row with its Lance row ID, and the version read.

    Consume runs in the Curator environment, which provides ``pylance``. Ingest
    runs in the NRL environment, which provides LanceDB instead.
    """

    columns = element_schema().names
    try:
        import lance
    except ModuleNotFoundError:
        import lancedb

        table = lancedb.connect(str(path.parent)).open_table(path.name.removesuffix(".lance"))
        if version is not None:
            table.checkout(version)
        schema = table.schema() if callable(table.schema) else table.schema
        return schema, table.search().select(columns).with_row_id(True).limit(None).to_arrow(), int(table.version)
    dataset = lance.dataset(str(path), version=version)
    return dataset.schema, dataset.to_table(columns=columns, with_row_id=True), int(dataset.version)


def _fragment_id(row_id: int) -> int:
    return int(row_id) >> 32
