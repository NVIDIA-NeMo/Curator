# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Lifecycle and validation for the NRL-to-Curator Lance recipe."""

from __future__ import annotations

import contextlib
import copy
import importlib
import importlib.metadata
import json
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nrl_lance_contract as contract

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from types import ModuleType

    import pyarrow as pa

    from nemo_curator.pipeline import Pipeline


def _expected_document_provenance(
    inputs: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    sample_ids: set[str],
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for entry in inputs:
        digest = entry.get("content_sha256")
        if isinstance(digest, str) and digest in sample_ids:
            grouped.setdefault(digest, []).append(entry)
    if set(grouped) != sample_ids:
        msg = "source inventory and published sample identities differ"
        raise ValueError(msg)

    expected: dict[str, dict[str, Any]] = {}
    for sample_id, entries in grouped.items():
        ordered = sorted(entries, key=lambda entry: int(entry["input_index"]))
        representative_index = ordered[0].get("representative_input_index")
        representatives = [entry for entry in ordered if entry.get("input_index") == representative_index]
        if len(representatives) != 1:
            msg = f"sample {sample_id} does not have exactly one representative"
            raise ValueError(msg)
        representative = representatives[0]
        aliases = [
            {
                "path": str(entry["path"]),
                "url": entry.get("url"),
                "input_index": int(entry["input_index"]),
                "valid_blank_pages": list(entry.get("valid_blank_pages", [])),
            }
            for entry in ordered
        ]
        expected[sample_id] = {
            "source_path": str(representative["path"]),
            "source_name": Path(str(representative["path"])).name,
            "num_pages": int(representative["expected_page_count"]),
            "source_aliases": aliases,
            "url": next((alias["url"] for alias in aliases if alias.get("url")), None),
            "valid_blank_pages": sorted({page for alias in aliases for page in alias["valid_blank_pages"]}),
            "run_id": run_id,
        }
        results = [document for document in documents if document.get("content_sha256") == sample_id]
        if len(results) != 1:
            msg = f"sample {sample_id} must have exactly one extraction result"
            raise ValueError(msg)
        result = results[0]
        coverage = contract.validate_page_outcomes(
            result.get("page_outcomes"),
            expected_page_count=expected[sample_id]["num_pages"],
            extraction_status=result.get("extraction_status"),
            issues=result.get("issues"),
        )
        if (
            result.get("status") != result.get("extraction_status")
            or result.get("expected_page_count") != expected[sample_id]["num_pages"]
            or result.get("element_count") != coverage["content_element_count"] + 1
            or any(result.get(key) != value for key, value in coverage.items() if key != "content_element_count")
        ):
            msg = f"Extraction counts differ from page outcomes for sample {sample_id}"
            raise ValueError(msg)
        expected[sample_id].update({key: result[key] for key in ("extraction_status", "page_outcomes", "issues")})
    return expected


def _validate_element_schema(schema: pa.Schema, *, label: str) -> None:
    expected = contract.element_schema()
    if schema.names != expected.names:
        msg = f"{label} fields are {schema.names}; expected {expected.names}"
        raise ValueError(msg)
    for expected_field, actual_field in zip(expected, schema, strict=True):
        if actual_field.type != expected_field.type or actual_field.nullable != expected_field.nullable:
            msg = f"{label} field {expected_field.name!r} is {actual_field}; expected {expected_field}"
            raise ValueError(msg)


def validate_element_table(  # noqa: C901, PLR0912, PLR0915
    path: Path,
    expected_element_counts: Mapping[str, int],
    *,
    expected_provenance: Mapping[str, Mapping[str, Any]],
    version: int | None = None,
) -> dict[str, Any]:
    """Validate the exact Arrow contract, document order, and fragment ownership."""

    expected_counts = {str(sample_id): int(count) for sample_id, count in expected_element_counts.items()}
    if not expected_counts or any(count <= 0 for count in expected_counts.values()):
        msg = "Every published document must have at least its metadata row"
        raise ValueError(msg)

    schema, projected, table_version = contract.read_element_table(path, version=version)
    _validate_element_schema(schema, label="Element table")
    records = sorted(projected.to_pylist(), key=lambda row: int(row["_rowid"]))
    positions: dict[str, list[int]] = {}
    sample_fragments: dict[str, set[int]] = {}
    fragment_samples: dict[int, set[str]] = {}
    metadata_counts: Counter[str] = Counter()
    image_hashes: dict[str, str] = {}
    page_element_counts: dict[str, Counter[int]] = {}
    last_page: dict[str, int] = {}
    page_outcomes: dict[str, list[dict[str, Any]]] = {}

    for row in records:
        sample_id = row["sample_id"]
        position = int(row["position"])
        positions.setdefault(sample_id, []).append(position)
        fragment_id = contract._fragment_id(row["_rowid"])
        sample_fragments.setdefault(sample_id, set()).add(fragment_id)
        fragment_samples.setdefault(fragment_id, set()).add(sample_id)
        if row["content_sha256"] != sample_id:
            msg = f"Element identity differs for sample {sample_id}"
            raise ValueError(msg)
        if row["source_ref"] is not None or row["materialize_error"] is not None:
            msg = f"Published row has a locator or materialization error for sample {sample_id}"
            raise ValueError(msg)
        provenance = expected_provenance.get(sample_id)
        if provenance is None:
            msg = f"Unexpected sample {sample_id}"
            raise ValueError(msg)
        try:
            aliases = json.loads(row["source_aliases"])
        except (TypeError, json.JSONDecodeError) as exc:
            msg = f"Invalid aliases for sample {sample_id}"
            raise ValueError(msg) from exc
        if (
            row["source_path"] != provenance["source_path"]
            or row["pdf_name"] != provenance["source_name"]
            or row["url"] != provenance["url"]
            or row["run_id"] != provenance["run_id"]
            or aliases != provenance["source_aliases"]
        ):
            msg = f"Element provenance differs for sample {sample_id}"
            raise ValueError(msg)

        modality = row["modality"]
        binary_content = row["binary_content"]
        if position == -1:
            metadata_counts[sample_id] += 1
            if (
                modality != "metadata"
                or row["content_type"] != "application/json"
                or binary_content is not None
                or row["page_number"] is not None
                or row["element_class"] is not None
                or row["bbox_xyxy_norm"] is not None
                or row["bbox_coordinate_space"] is not None
            ):
                msg = f"Malformed metadata row for sample {sample_id}"
                raise ValueError(msg)
            try:
                metadata = json.loads(row["text_content"])
            except (TypeError, json.JSONDecodeError) as exc:
                msg = f"Invalid metadata JSON for sample {sample_id}"
                raise ValueError(msg) from exc
            if (
                metadata.get("content_sha256") != sample_id
                or metadata.get("pdf_name") != provenance["source_name"]
                or metadata.get("num_pages") != provenance["num_pages"]
                or metadata.get("source_path") != provenance["source_path"]
                or metadata.get("source_aliases") != provenance["source_aliases"]
                or metadata.get("url") != provenance["url"]
                or metadata.get("valid_blank_pages") != provenance["valid_blank_pages"]
            ):
                msg = f"Metadata provenance differs for sample {sample_id}"
                raise ValueError(msg)
            contract.validate_page_outcomes(
                metadata.get("page_outcomes"),
                expected_page_count=provenance["num_pages"],
                extraction_status=metadata.get("extraction_status"),
                issues=metadata.get("issues"),
            )
            if any(metadata.get(key) != provenance[key] for key in ("extraction_status", "page_outcomes", "issues")):
                msg = f"Metadata extraction coverage differs for sample {sample_id}"
                raise ValueError(msg)
            page_outcomes[sample_id] = metadata["page_outcomes"]
            if any(
                page["status"] == "valid_blank" and page["page_number"] not in provenance["valid_blank_pages"]
                for page in metadata["page_outcomes"]
            ):
                msg = f"Undeclared blank in page outcomes for sample {sample_id}"
                raise ValueError(msg)
            continue

        if position < 0 or modality not in {"text", "table", "image"}:
            msg = f"Invalid content row for sample {sample_id}"
            raise ValueError(msg)
        page_number = row["page_number"]
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or not 0 <= page_number < provenance["num_pages"]
        ):
            msg = f"Invalid page number for sample {sample_id}"
            raise ValueError(msg)
        if page_number < last_page.get(sample_id, -1):
            msg = f"Content page order differs for sample {sample_id}"
            raise ValueError(msg)
        last_page[sample_id] = page_number
        page_element_counts.setdefault(sample_id, Counter())[page_number] += 1
        if contract._normalize_bbox(row["bbox_xyxy_norm"]) is None:
            msg = f"Invalid bbox for sample {sample_id}"
            raise ValueError(msg)
        if row["bbox_coordinate_space"] != contract.COORDINATE_SPACE:
            msg = f"Invalid bbox coordinate space for sample {sample_id}"
            raise ValueError(msg)
        expected_type = "image/png" if modality == "image" else "text/markdown"
        if row["content_type"] != expected_type:
            msg = f"Invalid content type for sample {sample_id}"
            raise ValueError(msg)
        if modality == "image":
            if not isinstance(binary_content, (bytes, bytearray, memoryview)):
                msg = f"Image row is missing inline bytes for sample {sample_id}"
                raise ValueError(msg)
            image_hashes[f"{sample_id}:{position}"] = contract._sha256_bytes(bytes(binary_content))
        elif binary_content is not None:
            msg = f"Non-image row contains binary data for sample {sample_id}"
            raise ValueError(msg)

    if set(positions) != set(expected_counts):
        msg = f"Element samples differ: actual={sorted(positions)}, expected={sorted(expected_counts)}"
        raise ValueError(msg)
    split_samples = {
        sample_id: sorted(fragment_ids)
        for sample_id, fragment_ids in sample_fragments.items()
        if len(fragment_ids) != 1
    }
    if split_samples:
        msg = f"Documents span multiple Lance fragments: {split_samples}"
        raise ValueError(msg)
    mixed_fragments = {
        fragment_id: sorted(sample_ids) for fragment_id, sample_ids in fragment_samples.items() if len(sample_ids) != 1
    }
    if mixed_fragments:
        msg = f"Lance fragments contain multiple documents: {mixed_fragments}"
        raise ValueError(msg)
    if len(fragment_samples) != len(positions):
        msg = "Lance fragment and published-document counts differ"
        raise ValueError(msg)
    expected_row_count = sum(expected_counts.values())
    if len(records) != expected_row_count:
        msg = "Element row count differs from completed documents"
        raise ValueError(msg)
    for sample_id, expected_count in expected_counts.items():
        expected_positions = [-1, *range(expected_count - 1)]
        if positions[sample_id] != expected_positions:
            msg = f"Element positions for {sample_id} are {positions[sample_id]}; expected {expected_positions}"
            raise ValueError(msg)
        if metadata_counts[sample_id] != 1:
            msg = f"Expected exactly one metadata row for sample {sample_id}"
            raise ValueError(msg)
        expected_pages = {
            page["page_number"]: page["element_count"]
            for page in page_outcomes[sample_id]
            if page["status"] == "success"
        }
        if dict(page_element_counts.get(sample_id, {})) != expected_pages:
            msg = f"Content rows differ from validated page outcomes for sample {sample_id}"
            raise ValueError(msg)

    return {
        "name": contract.ELEMENT_TABLE,
        "path": str(path),
        "version": table_version,
        "row_count": expected_row_count,
        "document_count": len(positions),
        "fragment_count": len(fragment_samples),
        "inline_image_count": len(image_hashes),
        "inline_image_sha256": image_hashes,
        "schema_and_nullability_validated": True,
        "positions_validated": True,
        "page_outcomes_validated": True,
        "documents_per_fragment_validated": True,
        "provenance_validated": True,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _model_revision(model_id: str) -> str | None:
    from nemo_retriever.models.hf_model_registry import get_hf_revision

    return get_hf_revision(model_id)


def _load_graph_module() -> ModuleType:
    # Import by its stable module name so Ray can resolve the terminal operator
    # class in worker processes. The recipe entry point adds this directory to
    # ``sys.path`` before importing the runtime module.
    return importlib.import_module("nrl_graph")


def _counts(inputs: Sequence[Mapping[str, Any]], documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    publication_counts = Counter(str(item.get("publication_status", "unknown")) for item in documents)
    delivered = [item for item in documents if item.get("publication_status") in {"handed_off", "published"}]
    unknown_pages = sum(item.get("failed_page_count") is None for item in documents)
    known_failed_pages = sum(int(item.get("failed_page_count") or 0) for item in documents)
    return {
        "inputs": dict(sorted(Counter(str(item["status"]) for item in inputs).items())),
        "documents": dict(sorted(Counter(str(item["status"]) for item in documents).items())),
        "document_publication": dict(sorted(publication_counts.items())),
        "input_count": len(inputs),
        "representative_count": len(documents),
        "publishable_document_count": sum(
            item["status"] in contract._PUBLISHABLE_STATUSES and int(item.get("element_count", 0)) > 0
            for item in documents
        ),
        "handed_off_document_count": publication_counts["handed_off"],
        "published_document_count": publication_counts["published"],
        "element_row_count": sum(int(item.get("element_count", 0)) for item in documents),
        "complete_document_count": sum(item["status"] in {"success", "valid_blank"} for item in documents),
        "partial_document_count": sum(item["status"] == "partial" for item in documents),
        "failed_document_count": sum(item["status"] == "failed" for item in documents),
        "failed_page_count": None if unknown_pages else known_failed_pages,
        "known_failed_page_count": known_failed_pages,
        "unknown_page_count_document_count": unknown_pages,
        "delivered_document_count": len(delivered),
        "delivered_page_count": sum(int(item["validated_page_count"]) for item in delivered),
        "delivered_content_page_count": sum(
            int(item["validated_page_count"]) - int(item["blank_page_count"]) for item in delivered
        ),
        "delivered_content_element_count": sum(int(item["element_count"]) - 1 for item in delivered),
    }


def _publish_marker_and_update_state(
    marker_path: Path,
    marker: Mapping[str, Any],
    *,
    state_path: Path,
    state_updates: Mapping[str, Any],
) -> None:
    """Publish the authoritative marker before updating diagnostic state."""

    try:
        contract._write_json_exclusive_atomic(marker_path, marker)
    except contract.MarkerDurabilityUnconfirmedError:
        with contextlib.suppress(Exception):
            state = contract._load_json(state_path)
            state.update(state_updates)
            state["marker_durability"] = {"path": str(marker_path), "status": "unconfirmed"}
            contract._write_json_atomic(state_path, state)
        raise
    # The sealed marker is authoritative. A diagnostic-state write
    # failure must not turn a completed publication into an error that a caller
    # might retry against the same immutable run.
    with contextlib.suppress(Exception):
        state = contract._load_json(state_path)
        state.update(state_updates)
        state.pop("marker_durability", None)
        contract._write_json_atomic(state_path, state)


def _update_input_outcomes(
    inputs: list[dict[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    *,
    publication_status: str,
) -> None:
    by_digest = {str(document["content_sha256"]): document for document in documents}
    for entry in inputs:
        digest = entry.get("content_sha256")
        if not isinstance(digest, str):
            entry["publication_status"] = "not_applicable"
            continue
        document = by_digest.get(digest)
        if document is None:
            continue
        if entry["status"] == "duplicate":
            entry["representative_status"] = document["status"]
        else:
            entry["status"] = document["status"]
        entry["publication_status"] = (
            publication_status
            if document["status"] in contract._PUBLISHABLE_STATUSES and document.get("element_count", 0) > 0
            else "not_applicable"
        )


def _document_result(
    document: Mapping[str, Any],
    build: contract.DocumentBuild,
    *,
    publication_status: str,
) -> dict[str, Any]:
    page_outcomes = build.page_outcomes
    expected_pages = document.get("expected_page_count")
    if not page_outcomes and isinstance(expected_pages, int) and expected_pages > 0:
        page_outcomes = [
            {"page_number": page, "status": "failed", "element_count": 0, "issues": build.issues}
            for page in range(expected_pages)
        ]
    return {
        "content_sha256": document["content_sha256"],
        "path": document["path"],
        "status": build.status,
        "extraction_status": build.status,
        "publication_status": (
            publication_status if build.status in contract._PUBLISHABLE_STATUSES else "not_applicable"
        ),
        "expected_page_count": document.get("expected_page_count"),
        "validated_page_count": build.page_count,
        "blank_page_count": build.blank_page_count,
        "content_page_count": build.page_count - build.blank_page_count,
        "failed_page_count": sum(page["status"] == "failed" for page in page_outcomes) if page_outcomes else None,
        "page_outcomes": page_outcomes,
        "element_count": len(build.rows),
        "issues": build.issues,
    }


def run_ingest(  # noqa: C901, PLR0912, PLR0915
    args: argparse.Namespace, *, graph_runner: Callable[..., Any] | None = None
) -> Path:
    """Run the extraction-only custom graph and publish a validated Lance handoff."""

    parse_batch_size = getattr(args, "parse_batch_size", contract.DEFAULT_PARSE_BATCH_SIZE)
    parse_cpus = getattr(args, "parse_cpus", contract.DEFAULT_PARSE_CPUS)
    projection_block_rows = getattr(args, "projection_block_rows", None)
    contract.validate_parse_scheduling(parse_batch_size, parse_cpus)
    contract.validate_projection_block_rows(projection_block_rows)
    contract.validate_projection_workers(args.projection_workers)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    contract.validate_run_id(run_id)

    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
        source_records = contract._source_records_from_directory(input_dir)
        input_spec = {"input_dir": str(input_dir)}
    else:
        manifest_path = Path(args.manifest).expanduser().resolve()
        source_records = contract._source_records_from_manifest(manifest_path)
        input_spec = {"manifest": str(manifest_path)}
    if not source_records:
        msg = "No PDF inputs were found"
        raise ValueError(msg)

    runner = graph_runner
    if runner is None:
        graph_module = _load_graph_module()
        graph_module.check_nrl_compatibility()
        runner = graph_module.run_nrl_graph
    model_revision = _model_revision(contract.PARSE_MODEL)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)

    started_at = contract._utc_now()
    inventory_started = time.perf_counter()
    inputs, representatives = contract.inventory_sources(source_records)
    timings: dict[str, float] = {
        "inventory_seconds": time.perf_counter() - inventory_started,
        "graph_seconds": 0.0,
        "finalize_and_write_seconds": 0.0,
        "adaptation_seconds": 0.0,
        "lance_write_seconds": 0.0,
        "validation_seconds": 0.0,
    }
    documents: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "unpublished",
        "publication_policy": contract.PUBLICATION_POLICY,
        "run_id": run_id,
        "started_at": started_at,
        "input": input_spec,
        "configuration": {
            "projection_workers": args.projection_workers,
            "projection_block_rows": projection_block_rows,
            "parse_batch_size": parse_batch_size,
            "parse_cpus": parse_cpus,
            "render": {"dpi": 200, "image_format": "png", "render_mode": "full_dpi"},
            "parse_model": contract.PARSE_MODEL,
            "task_prompt": contract.PARSE_TASK_PROMPT,
        },
        "inputs": inputs,
        "documents": documents,
        "timings": timings,
    }
    state_path = run_dir / contract.RUN_STATE_FILE
    contract._write_json_atomic(state_path, state)

    try:
        pending: list[dict[str, Any]] = []
        for document in representatives:
            if document.get("preflight_error") is None:
                pending.append(document)
                continue
            build = contract.DocumentBuild(
                status="failed",
                issues=[{"kind": "source_pdf_preflight_failure", "error": document["preflight_error"]}],
            )
            documents.append(_document_result(document, build, publication_status="not_applicable"))

        envelope_records: list[dict[str, Any]] = []
        if pending:
            graph_started = time.perf_counter()
            envelope = runner(
                [str(document["path"]) for document in pending],
                projection_workers=args.projection_workers,
                projection_block_rows=projection_block_rows,
                parse_batch_size=parse_batch_size,
                parse_cpus=parse_cpus,
            )
            timings["graph_seconds"] = time.perf_counter() - graph_started
            if not hasattr(envelope, "to_dict"):
                msg = "custom graph must return a pandas-compatible envelope"
                raise TypeError(msg)  # noqa: TRY301
            envelope_records = list(contract.validate_projection_envelope(envelope).to_dict("records"))

        expected_paths = {str(Path(str(document["path"])).resolve()) for document in pending}
        records_by_path: dict[str, list[dict[str, Any]]] = {path: [] for path in expected_paths}
        for record in envelope_records:
            canonical_path = str(Path(record["source_path"]).resolve())
            if canonical_path not in records_by_path:
                msg = f"projection envelope contains unexpected source path {canonical_path}"
                raise ValueError(msg)  # noqa: TRY301
            record["source_path"] = canonical_path
            records_by_path[canonical_path].append(record)

        writer: contract.ElementTableWriter | None = None
        expected_counts: dict[str, int] = {}
        finalization_started = time.perf_counter()
        for document in pending:
            path = str(Path(str(document["path"])).resolve())
            adaptation_started = time.perf_counter()
            build = contract.build_document_rows(document, records_by_path[path], run_id=run_id)
            timings["adaptation_seconds"] += time.perf_counter() - adaptation_started
            result = _document_result(document, build, publication_status="pending")
            documents.append(result)
            if build.status not in contract._PUBLISHABLE_STATUSES:
                continue
            write_started = time.perf_counter()
            if writer is None:
                writer = contract.ElementTableWriter(run_dir)
            writer.add_document(build.rows)
            timings["lance_write_seconds"] += time.perf_counter() - write_started
            expected_counts[str(document["content_sha256"])] = len(build.rows)
        timings["finalize_and_write_seconds"] = time.perf_counter() - finalization_started

        if writer is None or not expected_counts:
            msg = "No validated pages were available for handoff"
            raise RuntimeError(msg)  # noqa: TRY301

        _update_input_outcomes(inputs, documents, publication_status="pending")
        expected_provenance = _expected_document_provenance(
            inputs, run_id=run_id, sample_ids=set(expected_counts), documents=documents
        )
        validation_started = time.perf_counter()
        table_info = validate_element_table(
            contract._table_path(run_dir),
            expected_counts,
            expected_provenance=expected_provenance,
        )
        rehashed_input_count = contract._rehash_source_inventory(inputs)
        timings["validation_seconds"] = time.perf_counter() - validation_started

        for document in documents:
            if document["status"] in contract._PUBLISHABLE_STATUSES:
                document["publication_status"] = "handed_off"
        _update_input_outcomes(inputs, documents, publication_status="handed_off")

        handoff_core = {
            "schema_version": 1,
            "status": "tables_validated",
            "publication_policy": contract.PUBLICATION_POLICY,
            "run_id": run_id,
            "started_at": started_at,
            "handed_off_at": contract._utc_now(),
            "input": input_spec,
            "configuration": state["configuration"],
            "software": {"nemo_retriever": _package_version("nemo-retriever")},
            "models": {
                "nemotron_parse": {
                    "model_id": contract.PARSE_MODEL,
                    "revision": model_revision,
                    "task_prompt": contract.PARSE_TASK_PROMPT,
                }
            },
            "tables": {"database_uri": str(run_dir), contract.ELEMENT_TABLE: table_info},
            "counts": _counts(inputs, documents),
            "inputs": inputs,
            "documents": documents,
            "timings": timings,
            "source_inventory": {"rehash_status": "validated", "rehashed_input_count": rehashed_input_count},
            "publication": {
                "completion_required": True,
                "restart_policy": "restart into a fresh run directory",
            },
        }
        handoff = contract._seal_payload(handoff_core, contract._HANDOFF_HASH_FIELD)
        handoff_path = run_dir / contract.HANDOFF_MANIFEST_FILE
        _publish_marker_and_update_state(
            handoff_path,
            handoff,
            state_path=state_path,
            state_updates={
                "status": "tables_validated",
                "handed_off_at": handoff["handed_off_at"],
                "inputs": inputs,
                "documents": documents,
                "counts": handoff["counts"],
                "handoff_manifest": str(handoff_path),
                "handoff_sha256": handoff[contract._HANDOFF_HASH_FIELD],
            },
        )
    except contract.MarkerDurabilityUnconfirmedError:
        # Visibility is the publication point; a failed directory sync cannot
        # turn a valid handoff back into an unpublished extraction.
        raise
    except Exception as exc:
        state["status"] = "unpublished"
        state["failed_at"] = contract._utc_now()
        state["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        recorded_hashes = {str(document["content_sha256"]) for document in documents}
        for representative in representatives:
            digest = representative.get("content_sha256")
            if not isinstance(digest, str) or digest in recorded_hashes:
                continue
            documents.append(
                _document_result(
                    representative,
                    contract.DocumentBuild(
                        status="failed",
                        issues=[
                            {
                                "kind": "graph_or_publication_failure",
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                        ],
                    ),
                    publication_status="failed",
                )
            )
        _update_input_outcomes(inputs, documents, publication_status="failed")
        for document in documents:
            if document["status"] in contract._PUBLISHABLE_STATUSES:
                document["publication_status"] = "failed"
        state["inputs"] = inputs
        state["documents"] = documents
        state["counts"] = _counts(inputs, documents)
        with contextlib.suppress(Exception):
            contract._write_json_atomic(state_path, state)
        raise
    return handoff_path


def _load_handoff_manifest(path: Path) -> dict[str, Any]:
    if path.name != contract.HANDOFF_MANIFEST_FILE:
        msg = f"handoff manifest must be named {contract.HANDOFF_MANIFEST_FILE}"
        raise ValueError(msg)
    handoff = contract._load_sealed_json(
        path,
        contract._HANDOFF_HASH_FIELD,
        label="handoff manifest",
    )
    if handoff.get("status") != "tables_validated":
        msg = "Refusing a handoff whose status is not tables_validated"
        raise RuntimeError(msg)
    if handoff.get("publication_policy") != contract.PUBLICATION_POLICY:
        msg = f"Unknown publication policy {handoff.get('publication_policy')!r}"
        raise ValueError(msg)
    tables = handoff.get("tables")
    if not isinstance(tables, dict) or tables.get("database_uri") != str(path.parent):
        msg = "handoff database URI differs from its run directory"
        raise ValueError(msg)
    table = tables.get(contract.ELEMENT_TABLE)
    if not isinstance(table, dict):
        msg = f"handoff is missing {contract.ELEMENT_TABLE}"
        raise ValueError(msg)  # noqa: TRY004
    expected_path = path.parent / f"{contract.ELEMENT_TABLE}.lance"
    table_path = Path(str(table.get("path"))).resolve()
    if table_path != expected_path:
        msg = f"handoff table must be {expected_path}; got {table_path}"
        raise ValueError(msg)
    version = table.get("version")
    row_count = table.get("row_count")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        msg = "handoff table version is invalid"
        raise ValueError(msg)
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        msg = "handoff table row count is invalid"
        raise ValueError(msg)
    return handoff


def _binary_sha256(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (bytes, bytearray, memoryview)):
        msg = f"{label} binary_content is not bytes"
        raise TypeError(msg)
    return contract._sha256_bytes(bytes(value))


def _collect_reconciliation_rows(
    batches: Iterable[Any],
    field_names: Sequence[str],
    *,
    label: str,
) -> tuple[dict[tuple[str, int], tuple[dict[str, Any], str | None]], dict[str, list[int]], int]:
    comparable_fields = [name for name in field_names if name != "binary_content"]
    records: dict[tuple[str, int], tuple[dict[str, Any], str | None]] = {}
    positions: dict[str, list[int]] = {}
    row_count = 0
    for batch in batches:
        for row in batch.to_pylist():
            row_count += 1
            sample_id = row.get("sample_id")
            position = row.get("position")
            if not isinstance(sample_id, str) or not sample_id:
                msg = f"{label} row has an invalid sample_id"
                raise ValueError(msg)
            if isinstance(position, bool) or not isinstance(position, int):
                msg = f"{label} row for {sample_id} has an invalid position"
                raise ValueError(msg)  # noqa: TRY004
            key = (sample_id, position)
            if key in records:
                msg = f"{label} contains duplicate row key {key}"
                raise ValueError(msg)
            comparable = {name: row.get(name) for name in comparable_fields}
            records[key] = (comparable, _binary_sha256(row.get("binary_content"), label=f"{label} {key}"))
            positions.setdefault(sample_id, []).append(position)
    return records, positions, row_count


def _validate_consumed_output(table_path: Path, version: int, output_dir: Path) -> dict[str, Any]:  # noqa: C901
    import lance
    import pyarrow.parquet as pq

    field_names = list(contract.element_schema().names)
    source_dataset = lance.dataset(str(table_path), version=version)
    _validate_element_schema(source_dataset.schema, label="Pinned Lance source")
    expected_records, expected_positions, source_row_count = _collect_reconciliation_rows(
        source_dataset.scanner(columns=field_names).to_batches(),
        field_names,
        label="pinned Lance source",
    )

    parquet_files = sorted(output_dir.rglob("*.parquet"))
    if not parquet_files:
        msg = f"Curator wrote no Parquet files under {output_dir}"
        raise FileNotFoundError(msg)

    def parquet_batches() -> Iterable[Any]:
        for parquet_path in parquet_files:
            parquet_file = pq.ParquetFile(parquet_path)
            _validate_element_schema(parquet_file.schema_arrow, label=f"Parquet {parquet_path}")
            yield from parquet_file.iter_batches()

    native_reader_seconds = 0.0

    def native_reader_batches() -> Iterable[Any]:
        from nemo_curator.backends.ray_data import RayDataExecutor
        from nemo_curator.pipeline import Pipeline
        from nemo_curator.stages.interleaved.io import InterleavedParquetReader

        nonlocal native_reader_seconds
        pipeline = Pipeline(name="nrl_curator_parquet_reopen")
        pipeline.add_stage(
            InterleavedParquetReader(file_paths=[str(path) for path in parquet_files], files_per_partition=1)
        )
        started = time.perf_counter()
        tasks = pipeline.run(RayDataExecutor())
        native_reader_seconds = time.perf_counter() - started
        for task in tasks:
            table = task.to_pyarrow()
            _validate_element_schema(table.schema, label="Curator native Parquet reader")
            yield from table.to_batches()

    # Check stored values first so reader-side normalization cannot conceal corruption.
    for label, read_batches in (
        ("Curator Parquet output", parquet_batches),
        ("Curator native Parquet reader", native_reader_batches),
    ):
        actual_records, actual_positions, actual_row_count = _collect_reconciliation_rows(
            read_batches(), field_names, label=label
        )
        if actual_row_count != source_row_count or set(actual_records) != set(expected_records):
            missing = sorted(set(expected_records) - set(actual_records))[:10]
            unexpected = sorted(set(actual_records) - set(expected_records))[:10]
            msg = (
                f"{label} row keys differ: source={source_row_count}, output={actual_row_count}, "
                f"missing={missing}, unexpected={unexpected}"
            )
            raise ValueError(msg)
        if actual_positions != expected_positions:
            msg = f"{label} changed per-document positions or order"
            raise ValueError(msg)
        for key, expected in expected_records.items():
            if actual_records[key] != expected:
                msg = f"{label} row {key} differs from the pinned Lance source"
                raise ValueError(msg)
        del actual_records

    return {
        "schema_version": 1,
        "status": "validated",
        "validated_at": contract._utc_now(),
        "source": {
            "table_path": str(table_path),
            "version": version,
            "row_count": source_row_count,
            "document_count": len(expected_positions),
        },
        "output": {
            "directory": str(output_dir),
            "parquet_files": [str(path) for path in parquet_files],
            "row_count": actual_row_count,
            "document_count": len(actual_positions),
        },
        "reconciliation": {
            "key_fields": ["sample_id", "position"],
            "fields_compared": [name for name in field_names if name != "binary_content"],
            "binary_comparison": "sha256",
            "positions_by_sample": actual_positions,
            "schema_and_nullability_validated": True,
            "native_parquet_reader_validated": True,
            "native_parquet_reader_seconds": native_reader_seconds,
        },
    }


def _expected_counts_from_handoff(handoff: Mapping[str, Any]) -> dict[str, int]:
    documents = handoff.get("documents")
    if not isinstance(documents, list):
        msg = "handoff is missing document results"
        raise ValueError(msg)  # noqa: TRY004
    if any(
        isinstance(document, dict)
        and document.get("publication_status") == "handed_off"
        and document.get("status") not in contract._PUBLISHABLE_STATUSES
        for document in documents
    ):
        msg = "handoff contains a published document without validated pages"
        raise ValueError(msg)
    counts = {
        str(document["content_sha256"]): int(document["element_count"])
        for document in documents
        if isinstance(document, dict) and document.get("publication_status") == "handed_off"
    }
    if not counts:
        msg = "handoff contains no published documents"
        raise ValueError(msg)
    return counts


def _build_consume_pipeline(
    table_path: Path,
    *,
    version: int,
    output_dir: Path,
) -> Pipeline:
    """Build the exact native Curator compatibility pipeline."""

    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.interleaved import InterleavedAspectRatioFilterStage
    from nemo_curator.stages.interleaved.filter.blur_filter import InterleavedBlurFilterStage
    from nemo_curator.stages.interleaved.io import InterleavedLanceReader, InterleavedParquetWriterStage

    pipeline = Pipeline(
        name="nrl_lance_pdf_consume",
        description="Pinned NRL Lance elements -> image decode validation -> Parquet",
    )
    pipeline.add_stage(
        InterleavedLanceReader(
            path=str(table_path),
            fragments_per_partition=1,
            read_kwargs={"version": version},
            include_lance_metadata=False,
        )
    )
    pipeline.add_stage(
        InterleavedAspectRatioFilterStage(
            min_aspect_ratio=0.0,
            max_aspect_ratio=float("inf"),
            drop_invalid_rows=False,
            preserve_metadata_only_samples=True,
        )
    )
    pipeline.add_stage(
        InterleavedBlurFilterStage(
            score_threshold=0.0,
            drop_invalid_rows=False,
            preserve_metadata_only_samples=True,
        )
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=str(output_dir),
            materialize_on_write=False,
            mode="error",
            schema=contract.element_schema(),
            write_kwargs={"schema": contract.element_schema()},
        )
    )
    return pipeline


def run_consume(args: argparse.Namespace) -> Path:  # noqa: C901, PLR0912, PLR0915
    """Run native Curator validation and atomically publish completion."""

    handoff_path = Path(args.handoff_manifest).expanduser().resolve()
    handoff = _load_handoff_manifest(handoff_path)
    completion_path = handoff_path.parent / contract.COMPLETION_MANIFEST_FILE
    if completion_path.exists():
        msg = f"completion manifest already exists: {completion_path}"
        raise FileExistsError(msg)

    table_info = handoff["tables"][contract.ELEMENT_TABLE]
    table_path = Path(str(table_info["path"]))
    version = int(table_info["version"])
    expected_row_count = int(table_info["row_count"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    if contract._paths_overlap(output_dir, handoff_path.parent):
        msg = "consumer output directory must not overlap the handoff run directory"
        raise ValueError(msg)
    if output_dir.exists():
        msg = f"consumer output directory must be fresh: {output_dir}"
        raise FileExistsError(msg)
    contract._confirm_marker_durability(
        handoff_path,
        contract._HANDOFF_HASH_FIELD,
        expected_sha256=handoff[contract._HANDOFF_HASH_FIELD],
    )
    source_inputs = handoff.get("inputs")
    if not isinstance(source_inputs, list):
        msg = "handoff is missing source inventory"
        raise ValueError(msg)  # noqa: TRY004
    for source_input in source_inputs:
        if not isinstance(source_input, dict) or not isinstance(source_input.get("path"), str):
            msg = "handoff contains an invalid source input"
            raise ValueError(msg)  # noqa: TRY004
        source_path = Path(source_input["path"]).resolve()
        if contract._paths_overlap(output_dir, source_path):
            msg = f"consumer output directory must not contain source PDF {source_path}"
            raise ValueError(msg)

    expected_counts = _expected_counts_from_handoff(handoff)
    expected_provenance = _expected_document_provenance(
        source_inputs,
        run_id=str(handoff["run_id"]),
        sample_ids=set(expected_counts),
        documents=handoff["documents"],
    )

    import lance

    current_dataset = lance.dataset(str(table_path))
    if int(current_dataset.version) != version:
        msg = f"Current Lance version is {current_dataset.version}; handoff pins version {version}"
        raise RuntimeError(msg)
    if int(current_dataset.count_rows()) != expected_row_count:
        msg = "Current Lance row count differs from the handoff"
        raise RuntimeError(msg)

    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.tasks.utils import TaskPerfUtils

    pipeline = _build_consume_pipeline(table_path, version=version, output_dir=output_dir)

    consume_started = time.perf_counter()
    pipeline_started = time.perf_counter()
    tasks = pipeline.run(RayDataExecutor())
    pipeline_seconds = time.perf_counter() - pipeline_started
    stage_metrics = {
        stage: {name: values.tolist() for name, values in metrics.items()}
        for stage, metrics in TaskPerfUtils.collect_stage_metrics(tasks).items()
    }
    reconciliation_started = time.perf_counter()
    reconciliation = _validate_consumed_output(table_path, version, output_dir)
    reconciliation_seconds = time.perf_counter() - reconciliation_started
    if reconciliation["source"]["row_count"] != expected_row_count:
        msg = "Reconciled source count differs from the handoff"
        raise RuntimeError(msg)

    report_core = {
        **reconciliation,
        "kind": "curator_consume",
        "handoff": {
            "path": str(handoff_path),
            "sha256": handoff[contract._HANDOFF_HASH_FIELD],
        },
        "table": copy.deepcopy(table_info),
        "stage_metrics": stage_metrics,
        "timings": {
            "pipeline_seconds": pipeline_seconds,
            "reconciliation_seconds": reconciliation_seconds,
            "pipeline_and_reconciliation_seconds": time.perf_counter() - consume_started,
        },
    }
    current_dataset = lance.dataset(str(table_path))
    if int(current_dataset.version) != version or int(current_dataset.count_rows()) != expected_row_count:
        msg = "Lance table changed during Curator consumption"
        raise RuntimeError(msg)
    revalidated_table = validate_element_table(
        table_path,
        expected_counts,
        expected_provenance=expected_provenance,
        version=version,
    )
    for key in ("path", "version", "row_count", "document_count", "inline_image_sha256"):
        if revalidated_table.get(key) != table_info.get(key):
            msg = f"Lance table {key} differs from its handoff contract"
            raise RuntimeError(msg)
    rehashed_input_count = contract._rehash_source_inventory(source_inputs)
    report = contract._seal_payload(report_core, contract._REPORT_HASH_FIELD)
    report_path = output_dir / contract.CONSUME_REPORT_FILE
    contract._write_json_exclusive_atomic(report_path, report)

    published_inputs = copy.deepcopy(source_inputs)
    for entry in published_inputs:
        if entry.get("publication_status") == "handed_off":
            entry["publication_status"] = "published"
    published_documents = copy.deepcopy(handoff["documents"])
    for document in published_documents:
        if document.get("publication_status") == "handed_off":
            document["publication_status"] = "published"
    completion_core = {
        "schema_version": 1,
        "status": "published",
        "publication_policy": handoff["publication_policy"],
        "run_id": handoff["run_id"],
        "published_at": contract._utc_now(),
        "handoff": {
            "path": str(handoff_path),
            "sha256": handoff[contract._HANDOFF_HASH_FIELD],
        },
        "curator_consume": {
            "path": str(report_path),
            "sha256": report[contract._REPORT_HASH_FIELD],
        },
        "tables": copy.deepcopy(handoff["tables"]),
        "models": copy.deepcopy(handoff["models"]),
        "software": {**handoff["software"], "nemo_curator": _package_version("nemo-curator")},
        "counts": _counts(published_inputs, published_documents),
        "inputs": published_inputs,
        "documents": published_documents,
        "source_inventory": {
            "rehash_status": "validated",
            "rehashed_input_count": rehashed_input_count,
        },
        "publication": {
            "handoff_verified": True,
            "curator_output_reconciled": True,
            "table_revalidated_without_version_drift": True,
            "source_inventory_rehashed": True,
        },
    }
    completion = contract._seal_payload(completion_core, contract._COMPLETION_HASH_FIELD)
    _publish_marker_and_update_state(
        completion_path,
        completion,
        state_path=handoff_path.parent / contract.RUN_STATE_FILE,
        state_updates={
            "status": "published",
            "published_at": completion["published_at"],
            "consume_validated_at": report["validated_at"],
            "consume_report": str(report_path),
            "consume_report_sha256": report[contract._REPORT_HASH_FIELD],
            "completion_manifest": str(completion_path),
            "completion_sha256": completion[contract._COMPLETION_HASH_FIELD],
        },
    )
    return completion_path
