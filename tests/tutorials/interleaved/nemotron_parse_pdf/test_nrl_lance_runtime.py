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

import argparse
import copy
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import lance
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from .test_nrl_lance_contract import _document, _element, _marker, _provenance, _valid_png

if TYPE_CHECKING:
    from collections.abc import Callable

TUTORIAL_DIR = Path(__file__).resolve().parents[4] / "tutorials" / "interleaved" / "nemotron_parse_pdf"
sys.path.insert(0, str(TUTORIAL_DIR))

import nrl_lance_contract as contract  # noqa: E402
import nrl_lance_runtime as runtime  # noqa: E402


class _CuratorLanceWriter:
    """Write handoff fixtures with Curator's pylance; NRL writes the same Lance format with LanceDB."""

    def __init__(self, uri: Path) -> None:
        self.path = contract._table_path(uri)
        if self.path.exists():
            msg = f"Refusing to append to existing table at {self.path}"
            raise FileExistsError(msg)
        self.mode = "create"

    def add_document(self, rows: list[dict[str, Any]]) -> None:
        document = pa.Table.from_pylist(list(rows), schema=contract.element_schema())
        lance.write_dataset(document, str(self.path), mode=self.mode)
        self.mode = "append"


@pytest.fixture(autouse=True)
def _curator_lance_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "ElementTableWriter", _CuratorLanceWriter)


@pytest.mark.parametrize("option", ["parse_batch_size", "parse_cpus"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "4"])
def test_ingest_rejects_invalid_parse_scheduling_before_source_access(option: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{option} must be a positive integer"):
        runtime.run_ingest(argparse.Namespace(**{option: value}))


def test_ingest_rejects_silently_promoted_parse_batch_one() -> None:
    with pytest.raises(ValueError, match="NRL executor promotes batch size 1 to 64"):
        runtime.run_ingest(argparse.Namespace(parse_batch_size=1))


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "16"])
def test_ingest_rejects_invalid_projection_block_rows_before_source_access(value: object) -> None:
    with pytest.raises(ValueError, match="projection_block_rows must be a positive integer"):
        runtime.run_ingest(argparse.Namespace(projection_block_rows=value))


@pytest.fixture
def element_rows(tmp_path: Path) -> list[dict[str, Any]]:
    image = io.BytesIO()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image, format="PNG")
    source = tmp_path / "source.pdf"
    aliases = [{"path": str(source), "url": None, "input_index": 0, "valid_blank_pages": []}]
    provenance = {
        "content_sha256": "a" * 64,
        "pdf_name": source.name,
        "num_pages": 1,
        "source_path": str(source),
        "source_aliases": aliases,
        "url": None,
        "valid_blank_pages": [],
    }
    common = {
        **dict.fromkeys(contract.element_schema().names),
        "sample_id": "a" * 64,
        "content_sha256": "a" * 64,
        "pdf_name": source.name,
        "source_path": str(source),
        "source_aliases": json.dumps(aliases),
        "run_id": "test",
    }
    metadata = {
        **common,
        "position": -1,
        "modality": "metadata",
        "content_type": "application/json",
        "text_content": json.dumps(provenance),
    }
    content = {
        **common,
        "page_number": 0,
        "bbox_xyxy_norm": [0.125, 0.25, 0.5, 0.75],
        "bbox_coordinate_space": contract.COORDINATE_SPACE,
    }
    return [
        metadata,
        {
            **content,
            "position": 0,
            "modality": "text",
            "element_class": "Text",
            "content_type": "text/markdown",
            "text_content": "hello",
        },
        {
            **content,
            "position": 1,
            "modality": "table",
            "element_class": "Table",
            "content_type": "text/markdown",
            "text_content": (
                r"\begin{tabular}{lrr}"
                "\n"
                r"\multicolumn{3}{c}{Coverage <br> and <unknown>} \\"
                "\n"
                r"\multirow{2}{*}{North <sup>1</sup>} & 4 & 5 \\"
                "\n"
                r" & 6 & 7 \\"
                "\n"
                r"\end{tabular}"
            ),
        },
        {
            **content,
            "position": 2,
            "modality": "image",
            "element_class": "Picture",
            "content_type": "image/png",
            "text_content": "",
            "binary_content": image.getvalue(),
        },
    ]


@pytest.fixture
def published_table(tmp_path: Path, element_rows: list[dict[str, Any]]) -> tuple[Path, int]:
    source = tmp_path / "lance"
    source.mkdir()
    writer = contract.ElementTableWriter(source)
    writer.add_document(element_rows)
    path = contract._table_path(source)
    return path, int(lance.dataset(str(path)).version)


def _write_output(tmp_path: Path, rows: list[dict[str, Any]], schema: pa.Schema | None = None) -> Path:
    output = tmp_path / "output"
    output.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema or contract.element_schema()), output / "part.parquet")
    return output


@pytest.mark.parametrize("field_name", contract.element_schema().names)
def test_export_rejects_same_values_with_wrong_logical_type(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    field_name: str,
) -> None:
    schema = contract.element_schema()
    field = schema.field(field_name)
    replacements = {
        pa.string(): pa.large_string(),
        pa.int32(): pa.int64(),
        pa.large_binary(): pa.binary(),
        pa.list_(pa.float64()): pa.list_(pa.float32()),
    }
    changed = schema.set(schema.get_field_index(field_name), field.with_type(replacements[field.type]))
    output = _write_output(tmp_path, element_rows, changed)
    with pytest.raises(ValueError, match=f"Parquet .* field '{field_name}'"):
        runtime._validate_consumed_output(*published_table, output)


@pytest.mark.parametrize("field_name", [field.name for field in contract.element_schema() if not field.nullable])
def test_export_rejects_same_values_with_relaxed_nullability(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    field_name: str,
) -> None:
    schema = contract.element_schema()
    changed = schema.set(schema.get_field_index(field_name), schema.field(field_name).with_nullable(True))
    output = _write_output(tmp_path, element_rows, changed)
    with pytest.raises(ValueError, match=f"Parquet .* field '{field_name}'"):
        runtime._validate_consumed_output(*published_table, output)


@pytest.mark.parametrize("field_name", contract.element_schema().names)
def test_schema_validation_checks_every_fields_nullability(field_name: str) -> None:
    schema = contract.element_schema()
    field = schema.field(field_name)
    changed = schema.set(schema.get_field_index(field_name), field.with_nullable(not field.nullable))
    with pytest.raises(ValueError, match=f"field '{field_name}'"):
        runtime._validate_element_schema(changed, label="export")


@pytest.mark.parametrize(
    ("row_index", "field", "replacement"),
    [
        (1, "text_content", "changed"),
        (1, "text_content", None),
        (1, "bbox_xyxy_norm", [0.0, 0.0, 0.5, 0.5]),
        (1, "bbox_coordinate_space", "pixels"),
        (1, "source_path", "/different/source.pdf"),
        (1, "source_aliases", "[]"),
        (1, "source_ref", '{"path":"wrong"}'),
        (1, "materialize_error", "unreported error"),
        (1, "content_sha256", "b" * 64),
        (1, "page_number", 1),
        (1, "pdf_name", "different.pdf"),
        (1, "url", "https://example.test/different"),
        (1, "element_class", "Title"),
        (1, "modality", "table"),
        (1, "content_type", "text/plain"),
        (1, "run_id", "another-run"),
        (2, "text_content", "| changed |"),
        (3, "binary_content", b"different image bytes"),
        (3, "binary_content", None),
    ],
)
def test_export_rejects_changed_content_or_provenance(  # noqa: PLR0913
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    row_index: int,
    field: str,
    replacement: object,
) -> None:
    changed = copy.deepcopy(element_rows)
    changed[row_index][field] = replacement
    output = _write_output(tmp_path, changed)
    with pytest.raises(ValueError, match="differs from the pinned Lance source"):
        runtime._validate_consumed_output(*published_table, output)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unexpected", "reordered"])
def test_export_rejects_changed_keys_or_order(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    change: str,
) -> None:
    changed = copy.deepcopy(element_rows)
    if change == "missing":
        changed.pop()
    elif change == "duplicate":
        changed.append(changed[-1])
    elif change == "unexpected":
        changed[-1]["sample_id"] = "b" * 64
    else:
        changed[1], changed[2] = changed[2], changed[1]
    output = _write_output(tmp_path, changed)
    with pytest.raises(ValueError, match=r"row keys differ|duplicate row key|positions or order"):
        runtime._validate_consumed_output(*published_table, output)


def test_export_validates_every_parquet_file(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
) -> None:
    output = _write_output(tmp_path, element_rows[:2])
    schema = contract.element_schema()
    changed = schema.set(0, schema.field(0).with_nullable(True))
    pq.write_table(pa.Table.from_pylist(element_rows[2:], schema=changed), output / "second.parquet")
    with pytest.raises(ValueError, match=r"second\.parquet.*field 'sample_id'"):
        runtime._validate_consumed_output(*published_table, output)


@pytest.mark.parametrize("change", ["missing", "extra", "reordered"])
def test_export_rejects_changed_field_layout(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    change: str,
) -> None:
    schema = contract.element_schema()
    if change == "missing":
        schema = schema.remove(8)
    elif change == "extra":
        schema = schema.append(pa.field("unexpected", pa.string()))
    else:
        fields = list(schema)
        fields[0], fields[1] = fields[1], fields[0]
        schema = pa.schema(fields)
    output = _write_output(tmp_path, element_rows, schema)
    with pytest.raises(ValueError, match=r"fields are .*expected"):
        runtime._validate_consumed_output(*published_table, output)


@pytest.mark.parametrize("change", ["content", "schema", "missing"])
def test_native_reader_cannot_change_validated_stored_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    change: str,
) -> None:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.tasks import InterleavedBatch

    output = _write_output(tmp_path, element_rows)
    changed = copy.deepcopy(element_rows)
    schema = contract.element_schema()
    if change == "content":
        changed[1]["text_content"] = "reader changed content"
    elif change == "schema":
        schema = schema.set(17, schema.field(17).with_nullable(True))
    else:
        changed.pop()
    batch = InterleavedBatch(dataset_name="test", data=pa.Table.from_pylist(changed, schema=schema))
    monkeypatch.setattr(Pipeline, "run", lambda _self, _executor: [batch])
    with pytest.raises(ValueError, match="Curator native Parquet reader"):
        runtime._validate_consumed_output(*published_table, output)


def test_native_pipeline_preserves_exact_schema_and_reopens_export(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
) -> None:
    from nemo_curator.backends.ray_data import RayDataExecutor

    output = tmp_path / "output"
    pipeline = runtime._build_consume_pipeline(published_table[0], version=published_table[1], output_dir=output)
    pipeline.run(RayDataExecutor())
    report = runtime._validate_consumed_output(*published_table, output)

    assert report["output"]["row_count"] == len(element_rows)
    assert report["reconciliation"]["schema_and_nullability_validated"] is True
    assert report["reconciliation"]["native_parquet_reader_validated"] is True
    assert report["reconciliation"]["native_parquet_reader_seconds"] > 0
    exported_tables = []
    for path in output.rglob("*.parquet"):
        assert pq.read_schema(path).equals(contract.element_schema(), check_metadata=False)
        exported_tables.extend(row for row in pq.read_table(path).to_pylist() if row["modality"] == "table")
    assert exported_tables == [row for row in element_rows if row["modality"] == "table"]


def test_table_validation_rejects_multiple_documents_in_one_fragment(tmp_path: Path) -> None:
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
    lance.write_dataset(
        pa.Table.from_pylist(first_build.rows + second_build.rows, schema=contract.element_schema()),
        str(contract._table_path(tmp_path)),
    )

    with pytest.raises(ValueError, match="multiple documents"):
        runtime.validate_element_table(
            contract._table_path(tmp_path),
            {"1" * 64: 2, "2" * 64: 2},
            expected_provenance={**_provenance(first, "run", first_build), **_provenance(second, "run", second_build)},
        )


def _ingest_args(root: Path, manifest: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=None,
        manifest=str(manifest),
        output_root=str(root / "runs"),
        run_id="run",
        projection_workers=2,
    )


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("projection_block_rows", [None, 16])
@pytest.mark.parametrize(("parse_batch_size", "parse_cpus"), [(64, 1), (64, 4), (128, 1)])
def test_ingest_writes_handoff_only_after_validated_table(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    partial: bool,
    parse_batch_size: int,
    parse_cpus: int,
    projection_block_rows: int | None,
) -> None:
    source = tmp_path / "source.pdf"
    alias = tmp_path / "alias.pdf"
    source.write_bytes(b"same-pdf")
    alias.write_bytes(b"same-pdf")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"path": str(source)}) + "\n" + json.dumps({"path": str(alias)}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 2 if partial else 1)
    monkeypatch.setattr(runtime, "_model_revision", lambda _model: "model-revision")

    def fake_graph(paths: list[str], **kwargs) -> pd.DataFrame:
        assert paths == [str(source.resolve())]
        assert kwargs == {
            "projection_workers": 2,
            "projection_block_rows": projection_block_rows,
            "parse_batch_size": parse_batch_size,
            "parse_cpus": parse_cpus,
        }
        return pd.DataFrame([_marker(source, 1), _element(source, 1, 0)], dtype=object)

    args = _ingest_args(tmp_path, manifest)
    args.parse_batch_size = parse_batch_size
    args.parse_cpus = parse_cpus
    args.projection_block_rows = projection_block_rows
    handoff_path = runtime.run_ingest(args, graph_runner=fake_graph)
    handoff = runtime._load_handoff_manifest(handoff_path)
    assert handoff["status"] == "tables_validated"
    status = "partial" if partial else "success"
    assert handoff["counts"]["inputs"] == {"duplicate": 1, status: 1}
    assert handoff["inputs"][1]["representative_status"] == status
    assert handoff["counts"]["complete_document_count"] == int(not partial)
    assert handoff["counts"]["partial_document_count"] == int(partial)
    assert handoff["counts"]["failed_page_count"] == int(partial)
    assert handoff["counts"]["delivered_page_count"] == 1
    assert handoff["counts"]["delivered_content_element_count"] == 1
    assert handoff["tables"][contract.ELEMENT_TABLE]["row_count"] == 2
    assert handoff["configuration"]["parse_batch_size"] == parse_batch_size
    assert handoff["configuration"]["parse_cpus"] == parse_cpus
    assert handoff["configuration"]["projection_block_rows"] == projection_block_rows
    assert set(handoff["software"]) == {"nemo_retriever"}
    assert handoff["models"]["nemotron_parse"]["revision"] == "model-revision"
    assert not (handoff_path.parent / contract.COMPLETION_MANIFEST_FILE).exists()


def test_ingest_graph_failure_never_creates_publication_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": str(source)}) + "\n", encoding="utf-8")
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 1)

    def failed_graph(_paths: list[str], **_kwargs) -> pd.DataFrame:
        msg = "graph failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="graph failed"):
        runtime.run_ingest(_ingest_args(tmp_path, manifest), graph_runner=failed_graph)
    run_dir = tmp_path / "runs" / "run"
    assert not (run_dir / contract.HANDOFF_MANIFEST_FILE).exists()
    assert not (run_dir / contract.COMPLETION_MANIFEST_FILE).exists()
    assert contract._load_json(run_dir / contract.RUN_STATE_FILE)["status"] == "unpublished"


def test_structural_vertical_slice_accounts_for_duplicate_blank_and_corrupt_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    alias = tmp_path / "source-alias.pdf"
    blank = tmp_path / "blank.pdf"
    corrupt = tmp_path / "corrupt.pdf"
    source.write_bytes(b"multimodal")
    alias.write_bytes(b"multimodal")
    blank.write_bytes(b"blank")
    corrupt.write_bytes(b"corrupt")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {"path": str(source)},
                {"path": str(alias)},
                {"path": str(blank), "valid_blank_pages": [0]},
                {"path": str(corrupt)},
            )
        ),
        encoding="utf-8",
    )

    def page_count(path: Path) -> int:
        if path == corrupt:
            msg = "encrypted or corrupt"
            raise ValueError(msg)
        return 2 if path == source else 1

    monkeypatch.setattr(contract, "_pdf_page_count", page_count)
    monkeypatch.setattr(runtime, "_model_revision", lambda _model: "model-revision")

    def fake_graph(paths: list[str], **_kwargs) -> pd.DataFrame:
        assert paths == [str(source.resolve()), str(blank.resolve())]
        return pd.DataFrame(
            [
                _marker(source, 1, count=3),
                _element(source, 1, 0, text="first"),
                _element(source, 1, 1, element_class="Table", modality="table", text="| A |"),
                _element(
                    source,
                    1,
                    2,
                    element_class="Picture",
                    modality="image",
                    text="",
                    binary=_valid_png(),
                ),
                _marker(source, 2),
                _element(source, 2, 0, text="second"),
                _marker(blank, 1, outcome="empty", count=0),
            ],
            dtype=object,
        )

    handoff_path = runtime.run_ingest(_ingest_args(tmp_path, manifest), graph_runner=fake_graph)
    handoff = runtime._load_handoff_manifest(handoff_path)

    assert handoff["counts"]["inputs"] == {
        "duplicate": 1,
        "failed": 1,
        "success": 1,
        "valid_blank": 1,
    }
    assert handoff["counts"]["documents"] == {"failed": 1, "success": 1, "valid_blank": 1}
    assert handoff["counts"]["handed_off_document_count"] == 2
    assert handoff["counts"]["published_document_count"] == 0
    assert handoff["tables"][contract.ELEMENT_TABLE]["document_count"] == 2
    assert {document["status"] for document in handoff["documents"]} == {
        "failed",
        "success",
        "valid_blank",
    }
    assert not (handoff_path.parent / contract.COMPLETION_MANIFEST_FILE).exists()


def test_completion_marker_precedes_non_authoritative_state_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    completion_path = tmp_path / contract.COMPLETION_MANIFEST_FILE
    state_path = tmp_path / contract.RUN_STATE_FILE
    contract._write_json_atomic(state_path, {"status": "tables_validated"})
    original_write = contract._write_json_atomic

    def fail_state_update(path: Path, payload: dict[str, Any]) -> None:
        if path == state_path:
            msg = "diagnostic state unavailable"
            raise OSError(msg)
        original_write(path, payload)

    monkeypatch.setattr(contract, "_write_json_atomic", fail_state_update)
    runtime._publish_completion_and_update_state(
        completion_path,
        {"status": "published"},
        state_path=state_path,
        state_updates={"status": "published"},
    )

    assert contract._load_json(completion_path) == {"status": "published"}
    assert contract._load_json(state_path) == {"status": "tables_validated"}


def test_reconciliation_detects_dropped_invalid_image(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    document = _document(tmp_path / "source.pdf")
    rows = contract.build_document_rows(
        document,
        [_marker(Path(document["path"]), 1), _element(Path(document["path"]), 1, 0)],
        run_id="run",
    ).rows
    writer = contract.ElementTableWriter(source_dir)
    writer.add_document(rows)
    table = lance.dataset(str(contract._table_path(source_dir)))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    only_metadata = pa.Table.from_pylist(rows[:1], schema=contract.element_schema())
    pq.write_table(only_metadata, output_dir / "part.parquet")

    with pytest.raises(ValueError, match="row keys differ"):
        runtime._validate_consumed_output(contract._table_path(source_dir), int(table.version), output_dir)


def test_confirmed_completion_clears_prior_handoff_durability_diagnostic(tmp_path: Path) -> None:
    state_path = tmp_path / contract.RUN_STATE_FILE
    contract._write_json_atomic(
        state_path,
        {"status": "tables_validated", "marker_durability": {"path": "handoff", "status": "unconfirmed"}},
    )
    completion_path = tmp_path / contract.COMPLETION_MANIFEST_FILE
    completion = contract._seal_payload({"status": "published"}, contract._COMPLETION_HASH_FIELD)
    runtime._publish_completion_and_update_state(
        completion_path,
        completion,
        state_path=state_path,
        state_updates={"status": "published"},
    )
    assert contract._load_json(state_path) == {"status": "published"}
    assert (
        contract._load_sealed_json(completion_path, contract._COMPLETION_HASH_FIELD, label="completion") == completion
    )


def test_real_curator_pipeline_reads_validates_and_writes_pinned_lance(tmp_path: Path) -> None:
    from nemo_curator.backends.ray_data import RayDataExecutor

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    content_path = tmp_path / "content.pdf"
    blank_path = tmp_path / "blank.pdf"
    content = _document(content_path, "1" * 64)
    blank = _document(blank_path, "2" * 64, blanks=[0])
    content_rows = contract.build_document_rows(
        content,
        [
            _marker(content_path, 1, count=3),
            _element(content_path, 1, 0, text="text"),
            _element(content_path, 1, 1, element_class="Table", modality="table", text="| A |"),
            _element(
                content_path,
                1,
                2,
                element_class="Picture",
                modality="image",
                text="",
                binary=_valid_png(),
            ),
        ],
        run_id="run",
    ).rows
    blank_rows = contract.build_document_rows(
        blank,
        [_marker(blank_path, 1, outcome="empty", count=0)],
        run_id="run",
    ).rows
    writer = contract.ElementTableWriter(source_dir)
    writer.add_document(content_rows)
    writer.add_document(blank_rows)
    partial_rows = []
    for index, original in enumerate((content_rows, blank_rows), start=3):
        path = tmp_path / f"partial-{index}.pdf"
        document = _document(path, str(index) * 64, pages=2, blanks=[0] if len(original) == 1 else [])
        envelope = (
            [_marker(path, 1, outcome="empty", count=0)]
            if len(original) == 1
            else [_marker(path, 1), _element(path, 1, 0)]
        )
        build = contract.build_document_rows(document, envelope, run_id="run")
        assert build.status == "partial"
        partial_rows.extend(build.rows)
        writer.add_document(build.rows)
    table = lance.dataset(str(contract._table_path(source_dir)))
    output_dir = tmp_path / "output"
    pipeline = runtime._build_consume_pipeline(
        contract._table_path(source_dir), version=int(table.version), output_dir=output_dir
    )

    pipeline.run(RayDataExecutor())
    result = runtime._validate_consumed_output(
        contract._table_path(source_dir),
        int(table.version),
        output_dir,
    )

    assert result["status"] == "validated"
    assert result["source"]["row_count"] == len(content_rows) + len(blank_rows) + len(partial_rows)
    assert result["output"]["row_count"] == len(content_rows) + len(blank_rows) + len(partial_rows)
    assert result["output"]["document_count"] == 4


def test_truncated_png_passes_header_but_fails_pixel_decode_reconciliation(tmp_path: Path) -> None:
    from nemo_curator.backends.ray_data import RayDataExecutor

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = tmp_path / "source.pdf"
    document = _document(source_path)
    truncated = _valid_png()[:41]
    rows = contract.build_document_rows(
        document,
        [
            _marker(source_path, 1, count=2),
            _element(source_path, 1, 0, element_class="Table", modality="table", text="| A |"),
            _element(
                source_path,
                1,
                1,
                element_class="Picture",
                modality="image",
                text="",
                binary=truncated,
            ),
        ],
        run_id="run",
    ).rows
    assert rows[-1]["binary_content"].startswith(b"\x89PNG\r\n\x1a\n")
    writer = contract.ElementTableWriter(source_dir)
    writer.add_document(rows)
    table = lance.dataset(str(contract._table_path(source_dir)))
    output_dir = tmp_path / "output"
    pipeline = runtime._build_consume_pipeline(
        contract._table_path(source_dir), version=int(table.version), output_dir=output_dir
    )

    pipeline.run(RayDataExecutor())

    with pytest.raises(ValueError, match="row keys differ"):
        runtime._validate_consumed_output(
            contract._table_path(source_dir),
            int(table.version),
            output_dir,
        )


def _lifecycle_inputs(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> tuple[argparse.Namespace, Callable[..., pd.DataFrame]]:
    sources = [root / f"source-{index}.pdf" for index in range(3)]
    for index, path in enumerate(sources):
        path.write_bytes(f"disposable-pdf-{index}".encode())
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"path": str(path)}) + "\n" for path in sources))
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 1)
    monkeypatch.setattr(runtime, "_model_revision", lambda _model: "model-revision")

    def graph(paths: list[str], **_kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            [row for path in paths for row in (_marker(Path(path), 1), _element(Path(path), 1, 0))], dtype=object
        )

    return _ingest_args(root, manifest), graph


def _lifecycle_handoff(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    args, graph = _lifecycle_inputs(monkeypatch, root)
    return runtime.run_ingest(args, graph_runner=graph)


def _storage_consumer(monkeypatch: pytest.MonkeyPatch, after_write: Callable[[], None] = lambda: None) -> None:
    """Isolate lifecycle faults from Ray while retaining real Lance/Parquet reconciliation."""
    from nemo_curator.pipeline import Pipeline

    def build(table_path: Path, *, version: int, output_dir: Path) -> SimpleNamespace:
        def run(_executor: object) -> list[object]:
            output_dir.mkdir()
            pq.write_table(lance.dataset(str(table_path), version=version).to_table(), output_dir / "part.parquet")
            after_write()
            return []

        return SimpleNamespace(run=run)

    def reopen(pipeline: Pipeline, _executor: object) -> list[SimpleNamespace]:
        assert pipeline.name == "nrl_curator_parquet_reopen"
        return [
            SimpleNamespace(to_pyarrow=lambda path=path: pq.read_table(path)) for path in pipeline.stages[0].file_paths
        ]

    monkeypatch.setattr(runtime, "_build_consume_pipeline", build)
    monkeypatch.setattr(Pipeline, "run", reopen)


def _consume_args(handoff: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(handoff_manifest=str(handoff), output_dir=str(output))


def test_partial_handoff_consumption_preserves_status_and_missing_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, graph = _lifecycle_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 2)
    handoff_path = runtime.run_ingest(args, graph_runner=graph)
    handoff = runtime._load_handoff_manifest(handoff_path)
    assert handoff["counts"]["complete_document_count"] == 0
    assert handoff["counts"]["partial_document_count"] == 3
    assert handoff["counts"]["failed_page_count"] == 3
    _storage_consumer(monkeypatch)
    completion_path = runtime.run_consume(_consume_args(handoff_path, tmp_path / "export"))
    completion = contract._load_sealed_json(completion_path, contract._COMPLETION_HASH_FIELD, label="completion")
    assert completion["publication_policy"] == contract.PUBLICATION_POLICY
    assert set(completion["software"]) == {"nemo_retriever", "nemo_curator"}
    assert completion["counts"]["complete_document_count"] == 0
    assert completion["counts"]["partial_document_count"] == 3
    assert completion["counts"]["delivered_page_count"] == 3
    assert completion["counts"]["delivered_content_element_count"] == 3
    assert all(document["status"] == "partial" for document in completion["documents"])
    assert all(document["page_outcomes"][1]["status"] == "failed" for document in completion["documents"])


@pytest.mark.parametrize("mutation", ["status", "missing_page", "page_row", "issues"])
def test_partial_table_rejects_incorrect_coverage(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "source.pdf"
    document = _document(source, pages=2)
    build = contract.build_document_rows(document, [_marker(source, 1), _element(source, 1, 0)], run_id="run")
    metadata = json.loads(build.rows[0]["text_content"])
    provenance = json.loads(json.dumps(_provenance(document, "run", build)))
    if mutation == "status":
        metadata["extraction_status"] = "success"
    elif mutation == "missing_page":
        metadata["page_outcomes"].pop()
    elif mutation == "page_row":
        build.rows[1]["page_number"] = 1
    else:
        metadata["issues"] = [{"kind": "different_issue"}]
    build.rows[0]["text_content"] = json.dumps(metadata)
    contract.ElementTableWriter(tmp_path).add_document(build.rows)
    with pytest.raises(ValueError, match=r"coverage|outcomes|Extraction status"):
        runtime.validate_element_table(
            contract._table_path(tmp_path), {document["content_sha256"]: 2}, expected_provenance=provenance
        )


@pytest.mark.parametrize(
    "fault", ["first_write", "later_write", "table_validation", "source_changed", "state_write", "handoff_write"]
)
def test_ingest_storage_faults_never_publish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str) -> None:
    args, graph = _lifecycle_inputs(monkeypatch, tmp_path)
    original_add = contract.ElementTableWriter.add_document
    original_validate = runtime.validate_element_table
    original_state = contract._write_json_atomic
    calls = 0

    def add(writer: object, rows: list[dict[str, Any]]) -> None:
        nonlocal calls
        original_add(writer, rows)
        calls += 1
        if (fault == "first_write" and calls == 1) or (fault == "later_write" and calls == 2):
            msg = "injected document write failure"
            raise OSError(msg)

    def validate(*args, **kwargs) -> dict[str, Any]:
        if fault == "table_validation":
            msg = "injected validation failure"
            raise OSError(msg)
        if fault == "source_changed":
            (tmp_path / "source-0.pdf").write_bytes(b"changed after inventory")
        return original_validate(*args, **kwargs)

    def write_state(path: Path, payload: dict[str, Any]) -> None:
        if fault == "state_write" and payload.get("status") == "tables_validated":
            msg = "injected state failure"
            raise OSError(msg)
        original_state(path, payload)

    def marker_failure(_path: Path, _payload: object) -> None:
        msg = "injected marker failure"
        raise OSError(msg)

    monkeypatch.setattr(contract.ElementTableWriter, "add_document", add)
    monkeypatch.setattr(runtime, "validate_element_table", validate)
    monkeypatch.setattr(contract, "_write_json_atomic", write_state)
    if fault == "handoff_write":
        monkeypatch.setattr(contract, "_write_json_exclusive_atomic", marker_failure)
    with pytest.raises((OSError, RuntimeError), match=r"injected|changed after inventory"):
        runtime.run_ingest(args, graph_runner=graph)
    run_dir = tmp_path / "runs/run"
    assert not (run_dir / contract.HANDOFF_MANIFEST_FILE).exists()
    assert not (run_dir / contract.COMPLETION_MANIFEST_FILE).exists()
    assert contract._load_json(run_dir / contract.RUN_STATE_FILE)["status"] == "unpublished"


@pytest.mark.parametrize("phase", ["handoff", "completion"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_visible_marker_sync_failure_is_not_unpublished(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str, cleanup_failure: bool
) -> None:
    args, graph = _lifecycle_inputs(monkeypatch, tmp_path)
    handoff = None
    if phase == "completion":
        handoff = runtime.run_ingest(args, graph_runner=graph)
        _storage_consumer(monkeypatch)
    run_dir = tmp_path / "runs/run"
    marker = run_dir / (contract.HANDOFF_MANIFEST_FILE if phase == "handoff" else contract.COMPLETION_MANIFEST_FILE)
    hash_field = contract._HANDOFF_HASH_FIELD if phase == "handoff" else contract._COMPLETION_HASH_FIELD
    original_sync = contract._fsync_directory
    original_unlink = Path.unlink

    def fail_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.startswith(f".{marker.name}.") and marker.exists():
            msg = "injected marker temporary cleanup failure"
            raise OSError(msg)
        original_unlink(path, missing_ok=missing_ok)

    def fail_visible_sync(path: Path) -> None:
        if marker.exists():
            msg = "injected directory sync failure"
            raise OSError(msg)
        original_sync(path)

    def publish() -> None:
        if phase == "handoff":
            runtime.run_ingest(args, graph_runner=graph)
        else:
            runtime.run_consume(_consume_args(handoff, tmp_path / "export"))

    monkeypatch.setattr(contract, "_fsync_directory", fail_visible_sync)
    if cleanup_failure:
        monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(contract.MarkerDurabilityUnconfirmedError, match=r"visible.*durability is unconfirmed"):
        publish()
    payload = contract._load_sealed_json(marker, hash_field, label=phase)
    state = contract._load_json(run_dir / contract.RUN_STATE_FILE)
    assert state["status"] == ("tables_validated" if phase == "handoff" else "published")
    assert state["marker_durability"]["status"] == "unconfirmed"
    original_bytes = marker.read_bytes()
    with pytest.raises(FileExistsError):
        contract._write_json_exclusive_atomic(marker, {"replacement": True})
    monkeypatch.setattr(contract, "_fsync_directory", original_sync)
    with pytest.raises(ValueError, match="expected SHA-256"):
        contract._confirm_marker_durability(marker, hash_field, expected_sha256="0" * 64)
    confirmed = contract._confirm_marker_durability(marker, hash_field, expected_sha256=payload[hash_field])
    assert confirmed == payload
    assert marker.read_bytes() == original_bytes


def test_cleanup_failure_after_confirmed_visibility_does_not_fail_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_unlink = Path.unlink

    def fail_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.startswith(f".{contract.HANDOFF_MANIFEST_FILE}."):
            msg = "injected marker temporary cleanup failure"
            raise OSError(msg)
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    handoff = _lifecycle_handoff(monkeypatch, tmp_path)
    assert runtime._load_handoff_manifest(handoff)["status"] == "tables_validated"
    state = contract._load_json(handoff.parent / contract.RUN_STATE_FILE)
    assert state["status"] == "tables_validated"


@pytest.mark.parametrize(
    "fault", ["source_before", "source_during", "version_before", "version_during", "report_write", "completion_write"]
)
def test_consume_mutation_and_publication_faults_preserve_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    handoff = _lifecycle_handoff(monkeypatch, tmp_path)
    original_handoff = handoff.read_bytes()
    table_path = contract._table_path(handoff.parent)

    def mutate() -> None:
        if fault.startswith("source_"):
            (tmp_path / "source-0.pdf").write_bytes(b"changed during consumption")
        elif fault.startswith("version_"):
            lance.write_dataset(lance.dataset(str(table_path)).to_table(), str(table_path), mode="append")

    if fault.endswith("_before"):
        mutate()
    _storage_consumer(monkeypatch, mutate if fault.endswith("_during") else lambda: None)
    original_write = contract._write_json_exclusive_atomic

    def fail_marker(path: Path, payload: object) -> None:
        failing_name = contract.CONSUME_REPORT_FILE if fault == "report_write" else contract.COMPLETION_MANIFEST_FILE
        if fault.endswith("_write") and path.name == failing_name:
            msg = "injected publication failure"
            raise OSError(msg)
        original_write(path, payload)

    monkeypatch.setattr(contract, "_write_json_exclusive_atomic", fail_marker)
    with pytest.raises((OSError, RuntimeError), match=r"changed|Lance version|injected"):
        runtime.run_consume(_consume_args(handoff, tmp_path / "export"))
    assert handoff.read_bytes() == original_handoff
    assert not (handoff.parent / contract.COMPLETION_MANIFEST_FILE).exists()
    if fault == "report_write":
        assert not (tmp_path / "export" / contract.CONSUME_REPORT_FILE).exists()


def test_consume_requires_fresh_destination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    handoff = _lifecycle_handoff(monkeypatch, tmp_path)
    output = tmp_path / "export"
    output.mkdir()
    sentinel = output / "existing.txt"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="must be fresh"):
        runtime.run_consume(_consume_args(handoff, output))
    assert sentinel.read_bytes() == b"preserve"
    assert list(output.iterdir()) == [sentinel]


def test_completed_run_and_run_id_cannot_be_reused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    handoff = _lifecycle_handoff(monkeypatch, tmp_path)
    _storage_consumer(monkeypatch)
    completion = runtime.run_consume(_consume_args(handoff, tmp_path / "export"))
    timings = contract._load_json(tmp_path / "export" / contract.CONSUME_REPORT_FILE)["timings"]
    assert "total_seconds" not in timings
    assert timings["pipeline_and_reconciliation_seconds"] >= timings["pipeline_seconds"]
    snapshot = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="completion manifest already exists"):
        runtime.run_consume(_consume_args(handoff, tmp_path / "new-export"))
    with pytest.raises(FileExistsError):
        runtime.run_ingest(_ingest_args(tmp_path, tmp_path / "manifest.jsonl"))
    assert completion.exists()
    assert {path: path.read_bytes() for path in snapshot} == snapshot
    assert not (tmp_path / "new-export").exists()


@pytest.mark.parametrize("phase", ["ingest", "consume"])
def test_process_termination_leaves_no_false_completion_and_fresh_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str
) -> None:
    handoff = _lifecycle_handoff(monkeypatch, tmp_path) if phase == "consume" else None
    child_code = """
import pathlib, signal, sys, pytest
sys.path.insert(0, sys.argv[1])
from tests.tutorials.interleaved.nemotron_parse_pdf import test_nrl_lance_runtime as tests
root = pathlib.Path(sys.argv[2])
phase = sys.argv[3]
patch = pytest.MonkeyPatch()
patch.setattr(tests.contract, 'ElementTableWriter', tests._CuratorLanceWriter)
args, graph = tests._lifecycle_inputs(patch, root)
def pause():
    (root / 'ready').write_text('ready')
    signal.pause()
if phase == 'ingest':
    original = tests.contract.ElementTableWriter.add_document
    def write(writer, rows):
        original(writer, rows)
        pause()
    patch.setattr(tests.contract.ElementTableWriter, 'add_document', write)
    tests.runtime.run_ingest(args, graph_runner=graph)
else:
    tests._storage_consumer(patch, pause)
    tests.runtime.run_consume(tests._consume_args(root / 'runs/run/handoff_manifest.json', root / 'interrupted-export'))
"""
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", child_code, str(Path(__file__).resolve().parents[4]), str(tmp_path), phase],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 45
        while not (tmp_path / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "ready").exists(), process.communicate(timeout=5)
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == -signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)
    run_dir = tmp_path / "runs/run"
    assert not (run_dir / contract.COMPLETION_MANIFEST_FILE).exists()
    if phase == "ingest":
        assert not (run_dir / contract.HANDOFF_MANIFEST_FILE).exists()
    else:
        assert handoff.exists()
    args, graph = _lifecycle_inputs(monkeypatch, tmp_path)
    args.run_id = "retry"
    retry_handoff = runtime.run_ingest(args, graph_runner=graph)
    _storage_consumer(monkeypatch)
    completion = runtime.run_consume(_consume_args(retry_handoff, tmp_path / "retry-export"))
    assert (
        contract._load_sealed_json(completion, contract._COMPLETION_HASH_FIELD, label="completion")["status"]
        == "published"
    )
    assert not (run_dir / contract.COMPLETION_MANIFEST_FILE).exists()


def test_report_sync_failure_preserves_visible_report_without_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handoff = _lifecycle_handoff(monkeypatch, tmp_path)
    _storage_consumer(monkeypatch)
    report = tmp_path / "export" / contract.CONSUME_REPORT_FILE
    original_sync = contract._fsync_directory

    def fail_report_sync(path: Path) -> None:
        if report.exists() and path == report.parent:
            msg = "injected report sync failure"
            raise OSError(msg)
        original_sync(path)

    monkeypatch.setattr(contract, "_fsync_directory", fail_report_sync)
    with pytest.raises(contract.MarkerDurabilityUnconfirmedError):
        runtime.run_consume(_consume_args(handoff, report.parent))
    contract._load_sealed_json(report, contract._REPORT_HASH_FIELD, label="consume report")
    assert not (handoff.parent / contract.COMPLETION_MANIFEST_FILE).exists()
