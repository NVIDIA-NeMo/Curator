# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import (
    PDFPageBudgetValidationStage,
    validate_page_budget_outputs,
)
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_OUTPUT_VERSION, partition, project


def outputs(tmp_path: Path):
    manifest, tasks = partition(
        tmp_path, [{"file_name": "a.pdf", "num_pages": 4}, {"file_name": "b.pdf", "num_pages": 1}], 2
    )
    groups = {}
    for task in tasks:
        for chunk in project(task, failed={2}):
            groups.setdefault(chunk._metadata["pdf_document_id"], []).append(chunk)
    documents = [PDFDocumentMergeStage().process_batch(chunks)[0] for chunks in groups.values()]
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "output"), materialize_on_write=False, on_materialize_error="warn", use_output_file_id=True
    )
    files = [writer.process(document) for document in documents]
    return manifest, writer, documents, files


def test_writer_and_reader_roundtrip_preserve_lineage_and_failures(tmp_path: Path):
    manifest, _, _, files = outputs(tmp_path)
    result = validate_page_budget_outputs(
        str(manifest), [task.data[0] for task in files], expected_output_version=TEST_OUTPUT_VERSION
    )
    assert result["coverage_passed"]
    assert (result["documents"], result["pages"], result["successful_pages"], result["failed_pages"]) == (2, 5, 4, 1)
    assert not result["all_pages_succeeded"]
    assert not result["extraction_quality_validated"]


@pytest.mark.parametrize("empty", [False, True])
def test_completely_missing_documents_are_rejected(tmp_path: Path, empty: bool):
    manifest, _, _, files = outputs(tmp_path)
    with pytest.raises(ValueError, match="Missing whole PDFs"):
        validate_page_budget_outputs(
            str(manifest), [] if empty else [files[0].data[0]], expected_output_version=TEST_OUTPUT_VERSION
        )


def test_replayed_writer_uses_same_filename_after_fanin_task_id_changes(tmp_path: Path):
    _, writer, documents, files = outputs(tmp_path)
    documents[0].task_id = "random-fanin-task-id"
    replay = writer.process(documents[0])
    assert replay.data == files[0].data
    assert len(list((tmp_path / "output").glob("*.parquet"))) == 2


def test_swapped_source_version_is_not_accepted_as_same_document(tmp_path: Path):
    manifest, _, _, files = outputs(tmp_path)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[0]["source_version"] = "new-bytes"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="Unexpected"):
        validate_page_budget_outputs(
            str(manifest), [task.data[0] for task in files], expected_output_version=TEST_OUTPUT_VERSION
        )


def test_duplicate_delivered_documents_fail(tmp_path: Path):
    manifest, _, _, files = outputs(tmp_path)
    with pytest.raises(ValueError, match="Duplicate output paths"):
        validate_page_budget_outputs(
            str(manifest), [files[0].data[0]] * 2, expected_output_version=TEST_OUTPUT_VERSION
        )


def test_persisted_page_omission_is_rejected_independently(tmp_path: Path):
    manifest, writer, documents, files = outputs(tmp_path)
    rows = documents[0].to_pyarrow().to_pylist()
    rows = [row for row in rows if row.get("page_number") != 1]
    for i, row in enumerate(rows[1:]):
        row["position"] = i
    documents[0].data = pa.Table.from_pylist(rows, schema=documents[0].data.schema)
    writer.process(documents[0])
    with pytest.raises(ValueError, match="outcome"):
        validate_page_budget_outputs(
            str(manifest), [task.data[0] for task in files], expected_output_version=TEST_OUTPUT_VERSION
        )


def test_driver_validation_checks_even_an_empty_result_list(tmp_path: Path):
    manifest, _, _, _ = outputs(tmp_path)
    with pytest.raises(ValueError, match="Missing whole PDFs"):
        PDFPageBudgetValidationStage(
            manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION
        ).process_batch([])


def test_driver_validation_requires_writer_file_references(tmp_path: Path):
    manifest, _, documents, _ = outputs(tmp_path)
    with pytest.raises(TypeError, match="FileGroupTask outputs from a writer"):
        PDFPageBudgetValidationStage(
            manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION
        ).process_batch(documents)


@pytest.mark.parametrize("mixed", [False, True])
def test_wrong_or_mixed_output_version_is_rejected(tmp_path: Path, mixed: bool):
    manifest, writer, documents, files = outputs(tmp_path)
    for document in documents[:1] if mixed else documents:
        rows = document.data.to_pylist()
        info = json.loads(rows[0]["text_content"])
        info["pdf_provenance"]["output_version"] = "0" * 64
        rows[0]["text_content"] = json.dumps(info)
        document.data = pa.Table.from_pylist(rows, schema=document.data.schema)
        writer.process(document)
    with pytest.raises(ValueError, match="expected_output_version"):
        validate_page_budget_outputs(
            str(manifest), [task.data[0] for task in files], expected_output_version=TEST_OUTPUT_VERSION
        )
