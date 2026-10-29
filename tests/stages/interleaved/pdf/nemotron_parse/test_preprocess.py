# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Retain rendering failures only for the opted-in served pipeline."""

import io
import json
from pathlib import Path

import pyarrow as pa
import pytest
from PIL import Image

from nemo_curator.stages.interleaved.pdf.nemotron_parse import preprocess, utils
from nemo_curator.tasks import FileGroupTask


def pdf_bytes():
    pdfium = pytest.importorskip("pypdfium2")
    doc = pdfium.PdfDocument.new()
    for _ in range(3):
        doc.new_page(220, 170).close()
    stream = io.BytesIO()
    doc.save(stream)
    doc.close()
    return stream.getvalue()


@pytest.mark.parametrize("retain", [False, True])
def test_unreadable_document_failure_is_opt_in(tmp_path: Path, retain: bool):
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=retain)
    task = FileGroupTask(dataset_name="test", data=[json.dumps({"file_name": "missing.pdf"})])
    result = stage.process(task)
    if not retain:
        assert result is None
    else:
        rows = result.to_pyarrow().to_pylist()
        assert len(rows) == 1
        assert rows[0]["page_number"] is None
        assert json.loads(rows[0]["materialize_error"])["stage"] == "read_pdf"


def test_unopened_document_keeps_integer_page_numbers(tmp_path: Path):
    (tmp_path / "three.pdf").write_bytes(pdf_bytes())
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=True)
    entries = [json.dumps({"file_name": name}) for name in ("missing.pdf", "three.pdf")]
    table = stage.process(FileGroupTask(dataset_name="test", data=entries)).to_pyarrow()
    assert table.schema.field("page_number").type == pa.int64()
    assert table.column("page_number").to_pylist() == [None, 0, 1, 2]


def test_partial_render_preserves_physical_indexes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def render(_document: object, index: int, scale: float, max_size: tuple[int, int]) -> Image.Image | None:
        assert scale == 300 / 72.0
        assert max_size == (1664, 2048)
        return None if index == 1 else Image.new("RGB", (220, 170), "red")

    monkeypatch.setattr(utils, "_render_page", render)
    (tmp_path / "three.pdf").write_bytes(pdf_bytes())
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=True)
    task = FileGroupTask(dataset_name="test", data=[json.dumps({"file_name": "three.pdf"})])
    rows = stage.process(task).to_pyarrow().to_pylist()
    assert [(row["position"], row["page_number"]) for row in rows] == [(0, 0), (1, 1), (2, 2)]
    assert rows[0]["binary_content"]
    assert rows[2]["binary_content"]
    assert rows[1]["binary_content"] is None
    assert json.loads(rows[1]["materialize_error"])["stage"] == "render_pdf"


def test_manifest_selected_pages_only_preserve_physical_indexes(tmp_path: Path):
    (tmp_path / "three.pdf").write_bytes(pdf_bytes())
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=True)
    task = FileGroupTask(
        dataset_name="test",
        data=[json.dumps({"file_name": "three.pdf", "selected_page_indices": [2, 0]})],
    )
    rows = stage.process(task).to_pyarrow().to_pylist()
    assert [(row["position"], row["page_number"]) for row in rows] == [(2, 2), (0, 0)]


@pytest.mark.parametrize("retain", [False, True])
def test_selected_pages_keep_successful_indexes_after_render_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, retain: bool
):
    def render(_document: object, index: int, _scale: float, _max_size: tuple[int, int]) -> Image.Image | None:
        return None if index == 1 else Image.new("RGB", (10, 10), "red")

    monkeypatch.setattr(utils, "_render_page", render)
    (tmp_path / "three.pdf").write_bytes(pdf_bytes())
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=retain)
    task = FileGroupTask(
        dataset_name="test",
        data=[json.dumps({"file_name": "three.pdf", "selected_page_indices": [2, 1, 0]})],
    )
    rows = stage.process(task).to_pyarrow().to_pylist()
    assert [row["position"] for row in rows] == ([2, 1, 0] if retain else [2, 0])
    if retain:
        assert json.loads(rows[1]["materialize_error"])["stage"] == "render_pdf"


def test_timed_out_render_window_is_retried_in_smaller_groups(monkeypatch: pytest.MonkeyPatch):
    stage = preprocess.PDFPreprocessStage(retain_failures=True)
    calls = []

    def render_once(_pdf_bytes: bytes, indexes: list[int] | None) -> tuple[list[Image.Image | None], bool, int | None]:
        calls.append(indexes)
        if indexes is None:
            return [], True, 4
        return [Image.new("RGB", (1, 1), "red") for _ in indexes], False, None

    monkeypatch.setattr(stage, "_render_once", render_once)
    pages = stage._render_with_timeout(b"pdf", "slow.pdf")

    assert calls == [None, [0, 1], [2, 3]]
    assert len(pages) == 4


def test_single_page_timeout_does_not_drop_successful_siblings(monkeypatch: pytest.MonkeyPatch):
    stage = preprocess.PDFPreprocessStage(retain_failures=True)

    def render_once(_pdf_bytes: bytes, indexes: list[int] | None) -> tuple[list[Image.Image | None], bool, int | None]:
        assert indexes is not None
        if len(indexes) > 1:
            return [], True, None
        if indexes == [1]:
            return [], True, None
        return [Image.new("RGB", (1, 1), "red")], False, None

    monkeypatch.setattr(stage, "_render_once", render_once)
    pages = stage._render_with_timeout(b"pdf", "slow.pdf", [0, 1, 2])

    assert [page is not None for page in pages] == [True, False, True]


@pytest.mark.parametrize("indexes", [[], [-1], [1, 1], [True]])
def test_invalid_manifest_selected_pages_are_retained_as_failures(tmp_path: Path, indexes: list[int]):
    (tmp_path / "three.pdf").write_bytes(pdf_bytes())
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=True)
    task = FileGroupTask(
        dataset_name="test",
        data=[json.dumps({"file_name": "three.pdf", "selected_page_indices": indexes})],
    )
    rows = stage.process(task).to_pyarrow().to_pylist()
    assert len(rows) == 1
    assert json.loads(rows[0]["materialize_error"])["stage"] == "manifest"
