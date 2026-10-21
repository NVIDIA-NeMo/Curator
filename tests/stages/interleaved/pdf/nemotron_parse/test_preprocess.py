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

import pytest
from PIL import Image

from nemo_curator.stages.interleaved.pdf.nemotron_parse import preprocess
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


def test_partial_render_preserves_physical_indexes(monkeypatch: pytest.MonkeyPatch):
    def render(_document: object, index: int, scale: float, max_size: tuple[int, int]) -> Image.Image | None:
        assert scale == 300 / 72.0
        assert max_size == (1664, 2048)
        return None if index == 1 else Image.new("RGB", (220, 170), "red")

    monkeypatch.setattr(preprocess, "_render_page", render)
    stage = preprocess.PDFPreprocessStage(retain_failures=True)
    records = stage._render_with_timeout(pdf_bytes(), "three.pdf")
    assert [record["page_number"] for record in records] == [0, 1, 2]
    assert records[0]["binary_content"]
    assert records[1]["materialize_error"]
    assert records[2]["binary_content"]
