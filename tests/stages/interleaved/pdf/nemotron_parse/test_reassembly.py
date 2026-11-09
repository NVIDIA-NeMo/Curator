# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.tasks import InterleavedBatch
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_EXTRACTION_CONFIG, partition, project


def chunks(tmp_path: Path, size: int = 64, **kwargs):
    _, tasks = partition(tmp_path, [{"file_name": "long.pdf", "num_pages": 130}], size)
    return [chunk for task in tasks for chunk in project(task, **kwargs)]


def test_unordered_chunks_reconstruct_content_images_boxes_and_order(tmp_path: Path):
    split = chunks(tmp_path, 64)
    split_result = PDFDocumentMergeStage().process_batch(list(reversed(split)))[0]
    whole = PDFDocumentMergeStage().process_batch(chunks(tmp_path, 200))[0]
    assert split_result.to_pyarrow().to_pylist() == whole.to_pyarrow().to_pylist()
    assert split_result._metadata["output_file_id"] == whole._metadata["output_file_id"]
    rows = split_result.to_pyarrow().to_pylist()
    assert [row["position"] for row in rows[1:]] == list(range(390))
    assert json.loads(rows[0]["text_content"])["num_pages"] == 130
    assert all(row["document_id"] for row in rows)


@pytest.mark.parametrize(
    "mutation", ["missing_chunk", "duplicate_chunk", "wrong_page", "missing_content", "wrong_version"]
)
def test_declared_chunk_counts_do_not_hide_corruption(tmp_path: Path, mutation: str):
    tasks = chunks(tmp_path)
    if mutation == "missing_chunk":
        tasks.pop()
    elif mutation == "duplicate_chunk":
        tasks.append(copy.deepcopy(tasks[-1]))
    elif mutation == "wrong_version":
        tasks[-1]._metadata["pdf_chunk"]["source_version"] = "wrong-version"
    else:
        rows = tasks[0].to_pyarrow().to_pylist()
        if mutation == "wrong_page":
            rows[1]["page_number"] = 500
        else:
            rows.pop(1)
            for position, row in enumerate(rows[1:]):
                row["position"] = position
        tasks[0].data = pa.Table.from_pylist(rows, schema=tasks[0].data.schema)
    with pytest.raises(ValueError, match=r"PDF|Chunk|outcome|page|document"):
        PDFDocumentMergeStage().process_batch(tasks)


def test_noncontiguous_physical_pages_preserve_selection_order(tmp_path: Path):
    _, tasks = partition(tmp_path, [{"file_name": "a.pdf", "selected_page_indices": [19, 2, 7, 0]}], 2)
    projected = [chunk for task in tasks for chunk in project(task)]
    result = PDFDocumentMergeStage().process_batch(list(reversed(projected)))[0]
    rows = result.to_pyarrow().to_pylist()
    assert [row["page_number"] for row in rows[1:]][::3] == [19, 2, 7, 0]
    assert [x["page_number"] for x in json.loads(rows[0]["text_content"])["page_outcomes"]] == [19, 2, 7, 0]


def test_failures_and_zero_extracted_elements_remain_accounted(tmp_path: Path):
    result = PDFDocumentMergeStage().process_batch(chunks(tmp_path, failed={64}, no_elements={65}))[0]
    rows = result.to_pyarrow().to_pylist()
    outcomes = json.loads(rows[0]["text_content"])["page_outcomes"]
    assert outcomes[64] == {"page_number": 64, "status": "failed", "element_count": 1}
    assert outcomes[65] == {"page_number": 65, "status": "success", "element_count": 0}
    assert any(row.get("materialize_error") for row in rows if row.get("page_number") == 64)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong", "wrong_document"])
def test_postprocessing_verifies_actual_physical_identity(tmp_path: Path, mutation: str):
    _, tasks = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 3}])
    source = tasks[0]
    chunk = source._metadata["pdf_page_chunks"]["a.pdf"]
    rows = [
        {"sample_id": "a", "pdf_name": "a.pdf", "document_id": chunk["document_id"], "page_number": p}
        for p in range(3)
    ]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "wrong":
        rows[-1]["page_number"] = 99
    else:
        rows[-1]["document_id"] = "another-document"
    batch = InterleavedBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(rows),
        _metadata={**source._metadata, "parse_mode": "single", "pdf_extraction_config": TEST_EXTRACTION_CONFIG},
    )
    with pytest.raises(ValueError, match=r"Missing|physical"):
        NemotronParsePostprocessStage().process(batch)


def test_merge_requires_grouped_ray_execution():
    with pytest.raises(RuntimeError, match="RayDataExecutor"):
        PDFDocumentMergeStage().process(None)


@pytest.mark.parametrize("length", [0, 1, 3])
def test_merge_accepts_ray_numpy_groups(tmp_path: Path, length: int):
    _, tasks = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": max(length, 1)}], 1)
    projected = [chunk for task in tasks for chunk in project(task)] if length else []
    group = np.empty(length, dtype=object)
    group[:] = projected
    result = PDFDocumentMergeStage().process_batch(group)
    if length:
        assert json.loads(result[0].data["text_content"][0].as_py())["num_pages"] == length
    else:
        assert result == []


@pytest.mark.parametrize("first_range", ["text_only", "failed", "no_elements"])
def test_heterogeneous_ranges_match_whole_document(tmp_path: Path, first_range: str):
    kwargs = {first_range: {0}}
    _, split = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 2}], 1)
    split_output = PDFDocumentMergeStage().process_batch(
        [chunk for task in reversed(split) for chunk in project(task, **kwargs)]
    )[0]
    _, whole = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 2}], 2)
    whole_output = PDFDocumentMergeStage().process_batch(project(whole[0], **kwargs))[0]
    assert split_output.data.equals(whole_output.data)
    assert split_output._metadata["output_file_id"] == whole_output._metadata["output_file_id"]


def test_contiguous_positions_do_not_hide_reversed_physical_pages(tmp_path: Path):
    _, tasks = partition(tmp_path, [{"file_name": "a.pdf", "selected_page_indices": [19, 2, 7, 0]}], 4)
    chunk = project(tasks[0])[0]
    rows = chunk.data.to_pylist()
    content = list(reversed(rows[1:]))
    for position, row in enumerate(content):
        row["position"] = position
    chunk.data = pa.Table.from_pylist([rows[0], *content], schema=chunk.data.schema)
    with pytest.raises(ValueError, match="selection order"):
        PDFDocumentMergeStage().process_batch([chunk])


def test_fanout_does_not_repeat_other_documents_metadata(tmp_path: Path):
    _, tasks = partition(tmp_path, [{"file_name": f"{i}.pdf", "num_pages": 1} for i in range(4)])
    output = project(tasks[0])
    for task in output:
        assert "pdf_page_chunks" not in task._metadata
        assert task._metadata["source_files"] == [task._metadata["pdf_chunk"]["source_locator"]]
