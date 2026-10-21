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

"""Model-owned content and landscape crop geometry."""

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import build_interleaved_rows, image_to_bytes
from nemo_curator.tasks import InterleavedBatch


def test_landscape_picture_crops_original_content_not_center_padding():
    package = pytest.importorskip("nemotron_parse.inference")
    image = Image.new("RGB", (2200, 1700), "red")
    image.paste("blue", (220, 170, 1980, 1530))
    raw = "<x_0.1><y_0.1><x_0.9><y_0.9><class_Picture>"
    parser = package.DocumentParser(backend=lambda *_a, **_k: {"raw_out": raw, "finish_reason": "stop"})
    document = parser.parse(image)
    assert document["blocks"][0]["bbox_pixels"] == [220, 170, 1980, 1530]
    batch = InterleavedBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(
            [
                {
                    "sample_id": "wide",
                    "url": "source",
                    "pdf_name": "wide.pdf",
                    "page_number": 4,
                    "binary_content": image_to_bytes(image),
                    "parse_document": json.dumps(document),
                    "materialize_error": None,
                }
            ]
        ),
        _metadata={"parse_mode": "single"},
    )
    row = NemotronParsePostprocessStage().process(batch).to_pyarrow().to_pylist()[0]
    with Image.open(io.BytesIO(row["binary_content"])) as crop:
        assert crop.size == (1760, 1360)
        assert crop.getextrema() == ((0, 0), (0, 0), (255, 255))
    assert row["page_number"] == 4


def test_legacy_projection_remains_unchanged():
    image = Image.new("RGB", (220, 170), "red")
    raw = "<x_0.1><y_0.1>Hello<x_0.9><y_0.9><class_Text>"
    batch = InterleavedBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(
            [
                {
                    "sample_id": "legacy",
                    "url": "source",
                    "pdf_name": "legacy.pdf",
                    "position": 0,
                    "binary_content": image_to_bytes(image),
                    "text_content": raw,
                }
            ]
        ),
        _metadata={"model_path": "nvidia/NVIDIA-Nemotron-Parse-v1.2"},
    )
    actual = NemotronParsePostprocessStage().process(batch).to_pyarrow().to_pylist()
    expected = build_interleaved_rows("legacy", "source", "legacy.pdf", [image], [raw], reorder_floaters=False)
    expected = [{**row, "materialize_error": None} for row in expected]
    assert actual == expected


def test_error_row_roundtrip_uses_writer_warn_policy(tmp_path: Path):
    from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage

    error = json.dumps({"stage": "fullpage", "type": "GenerationError", "message": "cut off"})
    batch = InterleavedBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(
            [
                {
                    "sample_id": "failed",
                    "pdf_name": "failed.pdf",
                    "url": "source",
                    "position": 5,
                    "page_number": 5,
                    "modality": "page_image",
                    "binary_content": None,
                    "parse_document": None,
                    "materialize_error": error,
                }
            ]
        ),
        _metadata={"parse_mode": "single"},
    )
    projected = NemotronParsePostprocessStage().process(batch)
    strict = InterleavedParquetWriterStage(path=str(tmp_path / "strict"), materialize_on_write=False)
    with pytest.raises(RuntimeError, match="Materialization failed"):
        strict.process(projected)
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "warn"), materialize_on_write=False, on_materialize_error="warn"
    )
    writer.process(projected)
    files = list((tmp_path / "warn").glob("*.parquet"))
    assert len(files) == 1
    rows = pq.read_table(files[0]).to_pylist()
    assert len(rows) == 1
    assert rows[0]["page_number"] == 5
    assert rows[0]["materialize_error"] == error
