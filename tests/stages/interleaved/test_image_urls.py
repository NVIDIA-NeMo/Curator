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

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved.image_urls import InterleavedRemoteImageUrlStage
from nemo_curator.stages.interleaved.io.readers.parquet import InterleavedParquetReaderStage
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA


def _image(sample_id: str, position: int, path: str, member: str | None = None) -> dict:
    ref = json.dumps({"path": path, "member": member})
    return {"sample_id": sample_id, "position": position, "modality": "image", "source_ref": ref}


def _write(tmp_path: Path) -> str:
    rows = [
        {"sample_id": "a", "position": -1, "modality": "metadata", "url": "http://p/a", "score": 0.5},
        _image("a", 0, "http://i/1"),
        {"sample_id": "a", "position": 1, "modality": "text", "text_content": "words"},
        _image("a", 2, "http://i/1"),  # repeated on the page: listed once
        _image("a", 3, "s3://store/batch.tar", member="x.jpg"),  # stored: not remote
        {"sample_id": "b", "position": -1, "modality": "metadata", "url": "http://p/b", "score": None},
        _image("b", 0, "https://i/2"),
        {"sample_id": "b", "position": 1, "modality": "image", "source_ref": None},
        # the URL is kept in a user column
        {"sample_id": "b", "position": 2, "modality": "image", "source_ref": None, "linked": "http://i/3"},
    ]
    user = [("url", pa.string()), ("score", pa.float32()), ("linked", pa.string()), ("unused", pa.string())]
    schema = pa.schema([*INTERLEAVED_SCHEMA, *user])
    path = str(tmp_path / "pages.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema), path)
    return path


def test_reader_reserved_columns_leaves_the_rest_unread(tmp_path: Path) -> None:
    task = FileGroupTask(dataset_name="d", data=[_write(tmp_path)])
    reader = InterleavedParquetReaderStage(fields=("url", "score"), reserved_columns=("source_ref",))
    columns = reader._columns_to_read(pq.read_schema(task.data[0]))
    assert set(columns[:4]) == {"sample_id", "position", "modality", "source_ref"}
    assert columns[4:] == ["url", "score"]
    table = reader.process(task).to_pyarrow()
    assert set(INTERLEAVED_SCHEMA.names) <= set(table.schema.names)  # unread reserved columns are null-filled
    assert table["text_content"].null_count == table.num_rows
    assert "unused" not in table.schema.names
    with pytest.raises(ValueError, match="source_refs"):
        InterleavedParquetReaderStage(reserved_columns=("source_refs",))  # a typo must not read as "no data"
    everything = InterleavedParquetReaderStage(reserved_columns=("source_ref",))  # no fields: every user column
    assert everything._columns_to_read(pq.read_schema(task.data[0]))[4:] == ["url", "score", "linked", "unused"]


@pytest.mark.parametrize("dedup_rows", [4_000_000, 1])  # 1: every sample is deduplicated in its own slice
def test_remote_image_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dedup_rows: int) -> None:
    monkeypatch.setattr("nemo_curator.stages.interleaved.image_urls._DEDUP_ROWS", dedup_rows)
    task = FileGroupTask(dataset_name="d", data=[_write(tmp_path)])
    batch = InterleavedParquetReaderStage(fields=("url", "score"), reserved_columns=("source_ref",)).process(task)
    out = InterleavedRemoteImageUrlStage(sample_fields=("url", "score")).process(batch).to_pyarrow()
    rows = sorted(out.to_pylist(), key=lambda r: r["image_url"])
    assert rows == [
        {"sample_id": "a", "image_url": "http://i/1", "url": "http://p/a", "score": 0.5},
        {"sample_id": "b", "image_url": "https://i/2", "url": "http://p/b", "score": None},
    ]
    absent = InterleavedParquetReaderStage(fields=("url", "absent"), reserved_columns=("source_ref",)).process(task)
    filled = InterleavedRemoteImageUrlStage(sample_fields=("url", "absent")).process(absent).to_pyarrow()
    assert filled["absent"].null_count == filled.num_rows == 2  # a field the file lacks stays null
    bare = InterleavedRemoteImageUrlStage().process(batch).to_pyarrow()
    assert bare.column_names == ["sample_id", "image_url"]
    assert bare.num_rows == 2
    with_field = InterleavedParquetReaderStage(fields=("linked",), reserved_columns=("source_ref",)).process(task)
    urls = InterleavedRemoteImageUrlStage(url_field="linked").process(with_field).to_pyarrow()["image_url"].to_pylist()
    assert sorted(urls) == ["http://i/1", "http://i/3", "https://i/2"]
