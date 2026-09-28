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

import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

TUTORIAL_DIR = Path(__file__).resolve().parents[4] / "tutorials" / "interleaved" / "nemotron_parse_pdf"
sys.path.insert(0, str(TUTORIAL_DIR))

import nrl_lance_contract as contract  # noqa: E402
import nrl_lance_runtime as runtime  # noqa: E402


def _document(
    path: Path, digest: str = "a" * 64, *, pages: int = 1, blanks: list[int] | None = None
) -> dict[str, Any]:
    blank_pages = blanks or []
    alias = {
        "path": str(path.resolve()),
        "url": f"https://example.test/{path.name}",
        "input_index": 0,
        "valid_blank_pages": blank_pages,
    }
    return {
        "path": str(path.resolve()),
        "url": alias["url"],
        "content_sha256": digest,
        "expected_page_count": pages,
        "document_valid_blank_pages": blank_pages,
        "aliases": [alias],
    }


def _marker(
    path: Path,
    page: int,
    *,
    outcome: str = "parsed",
    count: int = 1,
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "record_type": "page_outcome",
        "source_path": str(path.resolve()),
        "native_page_number": page,
        "page_outcome": outcome,
        "element_count": count,
        "issues_json": json.dumps(issues or [], sort_keys=True, separators=(",", ":")),
        "raw_output_sha256": "b" * 64 if page > 0 else None,
        "element_index": None,
        "element_class": None,
        "modality": None,
        "content_type": None,
        "text_content": None,
        "binary_content": None,
        "bbox_xyxy_norm_json": None,
        "bbox_coordinate_space": None,
    }


def _element(  # noqa: PLR0913
    path: Path,
    page: int,
    index: int,
    *,
    element_class: str = "Text",
    modality: str = "text",
    text: str = "hello",
    binary: bytes | None = None,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "record_type": "element",
        "source_path": str(path.resolve()),
        "native_page_number": page,
        "page_outcome": None,
        "element_count": None,
        "issues_json": "[]",
        "raw_output_sha256": None,
        "element_index": index,
        "element_class": element_class,
        "modality": modality,
        "content_type": "image/png" if modality == "image" else "text/markdown",
        "text_content": text,
        "binary_content": binary,
        "bbox_xyxy_norm_json": json.dumps(bbox or [0.1, 0.2, 0.4, 0.5]),
        "bbox_coordinate_space": contract.COORDINATE_SPACE,
    }


def _valid_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(output, format="PNG")
    return output.getvalue()


def _provenance(document: dict[str, Any], run_id: str, build: contract.DocumentBuild) -> dict[str, dict[str, Any]]:
    return {
        str(document["content_sha256"]): {
            "source_path": document["path"],
            "source_name": Path(document["path"]).name,
            "num_pages": document["expected_page_count"],
            "source_aliases": document["aliases"],
            "url": document["url"],
            "valid_blank_pages": document["document_valid_blank_pages"],
            "run_id": run_id,
            "extraction_status": build.status,
            "page_outcomes": build.page_outcomes,
            "issues": build.issues,
        }
    }


def test_build_document_rows_preserves_model_and_page_order(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    document = _document(source, pages=2)
    rows = [
        _marker(source, 2, count=1),
        _element(source, 2, 0, element_class="Table", modality="table", text="| A |"),
        _marker(source, 1, count=2),
        _element(
            source,
            1,
            1,
            element_class="Picture",
            modality="image",
            text="",
            binary=b"\x89PNG\r\n\x1a\ntruncated",
        ),
        _element(source, 1, 0, text="first"),
    ]

    build = contract.build_document_rows(document, rows, run_id="run")

    assert build.status == "success"
    assert [row["position"] for row in build.rows] == [-1, 0, 1, 2]
    assert [row["page_number"] for row in build.rows[1:]] == [0, 0, 1]
    assert [row["element_class"] for row in build.rows[1:]] == ["Text", "Picture", "Table"]
    assert build.rows[2]["binary_content"] == b"\x89PNG\r\n\x1a\ntruncated"
    assert build.rows[2]["source_ref"] is None


def test_declared_all_blank_document_publishes_metadata_only(tmp_path: Path) -> None:
    source = tmp_path / "blank.pdf"
    document = _document(source, pages=2, blanks=[0, 1])
    rows = [_marker(source, 1, outcome="empty", count=0), _marker(source, 2, outcome="empty", count=0)]

    build = contract.build_document_rows(document, rows, run_id="run")

    assert build.status == "valid_blank"
    assert build.page_count == 2
    assert build.blank_page_count == 2
    assert len(build.rows) == 1
    assert build.rows[0]["position"] == -1


@pytest.mark.parametrize(
    ("rows", "pages", "blanks", "expected_status", "issue"),
    [
        ([], 1, [], "failed", "missing_pages"),
        ([_marker(Path("source.pdf"), 1), _marker(Path("source.pdf"), 1)], 1, [], "failed", "duplicate_page_outcomes"),
        ([_marker(Path("source.pdf"), 1, outcome="empty", count=0)], 1, [], "failed", "unexpected_empty_output"),
        (
            [_marker(Path("source.pdf"), 1, outcome="failed", count=0, issues=[{"kind": "page_stage_error"}])],
            1,
            [],
            "failed",
            "page_failed",
        ),
        (
            [
                _marker(
                    Path("source.pdf"), 0, outcome="failed", count=0, issues=[{"kind": "document_or_split_failure"}]
                )
            ],
            1,
            [],
            "failed",
            "document_or_split_failure",
        ),
    ],
)
def test_document_gate_rejects_incomplete_envelopes(  # noqa: PLR0913
    tmp_path: Path,
    rows: list[dict[str, Any]],
    pages: int,
    blanks: list[int],
    expected_status: str,
    issue: str,
) -> None:
    source = tmp_path / "source.pdf"
    for row in rows:
        row["source_path"] = str(source.resolve())
    build = contract.build_document_rows(_document(source, pages=pages, blanks=blanks), rows, run_id="run")
    assert build.status == expected_status
    assert build.rows == []
    assert issue in {item["kind"] for item in build.issues}


def test_one_valid_page_plus_failure_is_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    rows = [
        _marker(source, 1),
        _element(source, 1, 0),
        _marker(source, 2, outcome="failed", count=0, issues=[{"kind": "non_stop_finish"}]),
    ]
    build = contract.build_document_rows(_document(source, pages=2), rows, run_id="run")
    assert build.status == "partial"
    assert [row["page_number"] for row in build.rows] == [None, 0]
    metadata = json.loads(build.rows[0]["text_content"])
    assert metadata["extraction_status"] == "partial"
    assert metadata["page_outcomes"] == build.page_outcomes
    assert [page["status"] for page in build.page_outcomes] == ["success", "failed"]
    assert {issue["kind"] for issue in build.page_outcomes[1]["issues"]} == {"non_stop_finish", "page_failed"}


@pytest.mark.parametrize("failure", ["missing", "duplicate", "empty", "failed"])
def test_partial_delivery_keeps_only_whole_valid_pages(tmp_path: Path, failure: str) -> None:
    source = tmp_path / "source.pdf"
    good = [_marker(source, 1), _element(source, 1, 0, text="keep first")]
    bad = [_marker(source, 2, count=2), _element(source, 2, 0, text="must not leak"), _element(source, 2, 1)]
    if failure == "missing":
        bad = []
    elif failure == "duplicate":
        bad.append(_marker(source, 2, count=2))
    elif failure == "empty":
        bad = [_marker(source, 2, outcome="empty", count=0)]
    elif failure == "failed":
        bad = [_marker(source, 2, outcome="failed", count=0, issues=[{"kind": "page_stage_error"}])]
    rows = [*good, *bad, _marker(source, 3), _element(source, 3, 0, text="keep last")]
    build = contract.build_document_rows(_document(source, pages=3), rows, run_id="run")
    assert build.status == "partial"
    assert build.page_count == 2
    assert [row["position"] for row in build.rows] == [-1, 0, 1]
    assert [row["page_number"] for row in build.rows[1:]] == [0, 2]
    assert [row["text_content"] for row in build.rows[1:]] == ["keep first", "keep last"]
    assert build.page_outcomes[1]["status"] == "failed"
    assert build.page_outcomes[1]["element_count"] == 0
    assert build.page_outcomes[1]["issues"]


def test_partial_blank_only_document_keeps_coverage_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    build = contract.build_document_rows(
        _document(source, pages=2, blanks=[0]), [_marker(source, 1, outcome="empty", count=0)], run_id="run"
    )
    assert build.status == "partial"
    assert build.page_count == build.blank_page_count == 1
    assert len(build.rows) == 1
    assert [page["status"] for page in build.page_outcomes] == ["valid_blank", "failed"]


def test_extra_pages_are_reported_but_never_delivered(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    build = contract.build_document_rows(
        _document(source),
        [_marker(source, 1), _element(source, 1, 0), _marker(source, 2), _element(source, 2, 0)],
        run_id="run",
    )
    assert build.status == "partial"
    assert [row["page_number"] for row in build.rows] == [None, 0]
    assert {issue["kind"] for issue in build.issues} == {"unexpected_pages"}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "status", "count", "issues", "invalid_status"])
def test_page_outcome_contract_rejects_false_completeness(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "source.pdf"
    build = contract.build_document_rows(
        _document(source, pages=2), [_marker(source, 1), _element(source, 1, 0)], run_id="run"
    )
    pages = json.loads(json.dumps(build.page_outcomes))
    status = build.status
    if mutation == "missing":
        pages.pop()
    elif mutation == "duplicate":
        pages[1]["page_number"] = 0
    elif mutation == "status":
        status = "success"
    elif mutation == "count":
        pages[1]["element_count"] = 1
    elif mutation == "issues":
        pages[1]["issues"] = []
    elif mutation == "invalid_status":
        pages[1]["status"] = []
    with pytest.raises(ValueError, match=r"coverage|outcomes|outcome|Extraction status"):
        contract.validate_page_outcomes(pages, expected_page_count=2, extraction_status=status, issues=build.issues)


def test_inventory_deduplicates_bytes_not_basenames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = tmp_path / "one" / "report.pdf"
    alias = tmp_path / "alias" / "copy.pdf"
    same_name = tmp_path / "two" / "report.pdf"
    for path in (first, alias, same_name):
        path.parent.mkdir()
    first.write_bytes(b"same")
    alias.write_bytes(b"same")
    same_name.write_bytes(b"different")
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 3)

    inputs, representatives = contract.inventory_sources(
        [
            {"path": str(first), "url": None, "valid_blank_pages": [0]},
            {"path": str(alias), "url": None, "valid_blank_pages": [2]},
            {"path": str(same_name), "url": None, "valid_blank_pages": []},
        ]
    )

    assert len(representatives) == 2
    assert inputs[1]["status"] == "duplicate"
    assert inputs[0]["content_sha256"] == inputs[1]["content_sha256"]
    assert inputs[2]["content_sha256"] != inputs[0]["content_sha256"]
    assert representatives[0]["document_valid_blank_pages"] == [0, 2]


def test_inventory_rejects_out_of_range_blank_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = tmp_path / "one.pdf"
    alias = tmp_path / "alias.pdf"
    first.write_bytes(b"same")
    alias.write_bytes(b"same")
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 1)
    inputs, representatives = contract.inventory_sources(
        [
            {"path": str(first), "url": None, "valid_blank_pages": []},
            {"path": str(alias), "url": None, "valid_blank_pages": [1]},
        ]
    )
    assert representatives[0]["status"] == "failed"
    assert representatives[0]["preflight_error"]["type"] == "InvalidBlankPageDeclaration"
    assert inputs[1]["representative_status"] == "failed"


def test_manifest_resolves_relative_paths_and_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "store" / "source.pdf"
    target.parent.mkdir()
    target.write_bytes(b"pdf")
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "linked.pdf").symlink_to(target)
    manifest = manifests / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": "linked.pdf", "valid_blank_pages": [0, 0]}) + "\n", encoding="utf-8")

    assert contract._source_records_from_manifest(manifest) == [
        {"path": str(target.resolve()), "url": None, "valid_blank_pages": [0]}
    ]


@pytest.mark.parametrize("run_id", ["", ".", "..", "nested/run", "run id"])
def test_run_id_must_name_exactly_one_new_child(run_id: str) -> None:
    with pytest.raises(ValueError, match="run-id"):
        contract.validate_run_id(run_id)


@pytest.mark.parametrize(
    ("row_index", "updates", "message"),
    [
        (0, {"element_count": 2}, "element count mismatch"),
        (1, {"element_index": 1}, "not contiguous"),
        (0, {"text_content": "payload"}, "page outcome field text_content must be null"),
        (0, {"issues_json": '[{"kind":"nested"}]'}, "only failed outcomes may carry issues"),
        (0, {"page_outcome": "empty", "element_count": 0, "issues_json": '[{"kind":"nested"}]'}, "only failed"),
        (0, {"raw_output_sha256": None}, "successful model outcomes require raw_output_sha256"),
        (1, {"issues_json": '[{"kind":"nested"}]'}, "element rows cannot carry issues"),
        (1, {"bbox_xyxy_norm_json": "[0.0,0.0,2.0,1.0]"}, "four finite ordered floats"),
        (1, {"bbox_xyxy_norm_json": "[0.5,0.5,0.1,0.1]"}, "four finite ordered floats"),
        (1, {"bbox_xyxy_norm_json": '["0.0",0.0,0.4,0.2]'}, "bbox coordinates must be JSON numbers"),
        (1, {"modality": "table"}, "modality/content_type"),
        (1, {"element_class": "Picture", "modality": "image", "content_type": "image/png"}, "inline PNG bytes"),
    ],
)
def test_envelope_validator_rejects_malformed_records(
    tmp_path: Path, row_index: int, updates: dict[str, Any], message: str
) -> None:
    source = tmp_path / "source.pdf"
    rows = [_marker(source, 1), _element(source, 1, 0)]
    rows[row_index].update(updates)

    with pytest.raises(ValueError, match=message):
        contract.validate_projection_envelope(rows)


def test_envelope_validator_rejects_duplicate_page_outcomes(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    with pytest.raises(ValueError, match="duplicate page outcome"):
        contract.validate_projection_envelope([_marker(source, 1), _element(source, 1, 0), _marker(source, 1)])


def test_envelope_validator_normalizes_pandas_values(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    picture = _element(source, 1, 0, element_class="Picture", modality="image", text="", binary=_valid_png())
    picture["binary_content"] = memoryview(picture["binary_content"])

    envelope = contract.validate_projection_envelope(pd.DataFrame([_marker(source, 1), picture]))

    marker, element = envelope.to_dict("records")
    assert marker["element_count"] == 1
    assert isinstance(marker["element_count"], int)
    assert marker["element_index"] is None
    assert element["element_index"] == 0
    assert isinstance(element["element_index"], int)
    assert element["element_count"] is None
    assert isinstance(element["binary_content"], bytes)


def test_element_schema_is_exact() -> None:
    schema = contract.element_schema()
    assert schema.names == [
        "sample_id",
        "position",
        "modality",
        "content_type",
        "text_content",
        "binary_content",
        "source_ref",
        "materialize_error",
        "url",
        "page_number",
        "pdf_name",
        "element_class",
        "source_path",
        "source_aliases",
        "content_sha256",
        "bbox_xyxy_norm",
        "bbox_coordinate_space",
        "run_id",
    ]
    assert not schema.field("sample_id").nullable
    assert not schema.field("position").nullable
    assert not schema.field("modality").nullable
    assert not schema.field("content_sha256").nullable
    assert not schema.field("run_id").nullable


def test_lance_writer_keeps_each_document_in_one_fragment(tmp_path: Path) -> None:
    pytest.importorskip("lancedb", reason="LanceDB writes the handoff in the NRL environment")
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"
    first = _document(first_path, "1" * 64)
    second = _document(second_path, "2" * 64)
    first_build = contract.build_document_rows(
        first, [_marker(first_path, 1), _element(first_path, 1, 0)], run_id="run"
    )
    second_build = contract.build_document_rows(
        second, [_marker(second_path, 1), _element(second_path, 1, 0)], run_id="run"
    )
    writer = contract.ElementTableWriter(tmp_path)
    writer.add_document(first_build.rows)
    writer.add_document(second_build.rows)

    result = runtime.validate_element_table(
        contract._table_path(tmp_path),
        {"1" * 64: 2, "2" * 64: 2},
        expected_provenance={**_provenance(first, "run", first_build), **_provenance(second, "run", second_build)},
    )
    assert result["row_count"] == 4
    assert result["document_count"] == 2
    assert result["fragment_count"] == 2


def test_marker_file_sync_failure_precedes_visibility(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    marker = tmp_path / contract.HANDOFF_MANIFEST_FILE

    def fail_sync(_descriptor: int) -> None:
        msg = "injected file sync failure"
        raise OSError(msg)

    monkeypatch.setattr(contract.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="file sync failure"):
        contract._write_json_exclusive_atomic(marker, {"status": "tables_validated"})
    assert not marker.exists()
    assert list(tmp_path.iterdir()) == []


def test_durability_confirmation_rejects_tampered_marker(tmp_path: Path) -> None:
    marker = tmp_path / contract.HANDOFF_MANIFEST_FILE
    payload = contract._seal_payload({"status": "tables_validated"}, contract._HANDOFF_HASH_FIELD)
    expected = payload[contract._HANDOFF_HASH_FIELD]
    payload["status"] = "tampered"
    contract._write_json_exclusive_atomic(marker, payload)
    with pytest.raises(ValueError, match="SHA-256 is"):
        contract._confirm_marker_durability(marker, contract._HANDOFF_HASH_FIELD, expected_sha256=expected)
