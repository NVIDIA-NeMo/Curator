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
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved.io.readers.webdataset import WebdatasetReaderStage
from nemo_curator.stages.interleaved.io.writers.tabular import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.stages import BaseInterleavedFilterStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA, RESERVED_COLUMNS


def _read_batch(input_task: FileGroupTask) -> InterleavedBatch:
    batch = WebdatasetReaderStage(source_id_field="pdf_name").process(input_task)
    assert isinstance(batch, InterleavedBatch)
    return batch


def _source_ref(content_path: str, content_key: str | None) -> str:
    return json.dumps(
        {
            "path": content_path,
            "member": content_key,
            "byte_offset": None,
            "byte_size": None,
        }
    )


def test_writer_marks_materialize_error_on_bad_source_path(tmp_path: Path, input_task: FileGroupTask) -> None:
    batch = _read_batch(input_task)
    df = batch.to_pandas().copy()
    image_mask = df["modality"] == "image"
    assert image_mask.any()
    first_image_idx = df[image_mask].index[0]
    df.loc[first_image_idx, "source_ref"] = _source_ref("/definitely/missing/path.tar", "abc123.tiff")
    bad_batch = InterleavedBatch(
        task_id=batch.task_id,
        dataset_name=batch.dataset_name,
        data=df,
        _metadata=batch._metadata,
        _stage_perf=batch._stage_perf,
    )

    writer = InterleavedParquetWriterStage(path=str(tmp_path / "out_bad"), materialize_on_write=True, mode="overwrite")
    write_task = writer.process(bad_batch)
    written = pd.read_parquet(write_task.data[0])

    target = written.loc[first_image_idx]
    assert pd.isna(target["binary_content"])
    assert isinstance(target["materialize_error"], str)


def test_writer_materializes_direct_content_path_without_key(tmp_path: Path) -> None:
    image_bytes = b"raw-image-bytes"
    raw_path = tmp_path / "raw_image.jpg"
    raw_path.write_bytes(image_bytes)

    table = pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": None,
                "source_ref": _source_ref(str(raw_path), None),
                "materialize_error": None,
            }
        ],
        schema=INTERLEAVED_SCHEMA,
    )
    task = InterleavedBatch(
        task_id="direct_content_path",
        dataset_name="mint_test",
        data=table,
        _metadata={"source_files": [str(raw_path)]},
    )

    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "out_direct"), materialize_on_write=True, mode="overwrite"
    )
    write_task = writer.process(task)
    written = pd.read_parquet(write_task.data[0])
    assert written.loc[0, "binary_content"] == image_bytes
    assert pd.isna(written.loc[0, "materialize_error"])


def test_writer_does_not_persist_dataframe_index(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            }
        ]
    )
    df.index = pd.Index([99])
    task = InterleavedBatch(task_id="idx_task", dataset_name="mint_test", data=df)
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "out_idx"), materialize_on_write=False, mode="overwrite"
    )
    write_task = writer.process(task)
    written = pd.read_parquet(write_task.data[0])
    assert "__index_level_0__" not in written.columns


def test_interleaved_ordering_preserved_through_filter_and_write(tmp_path: Path) -> None:
    """End-to-end: interleaved text+image rows survive filtering and parquet roundtrip."""

    class _DropSecondImage(BaseInterleavedFilterStage):
        name: str = "drop_second_image"

        def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
            keep = pd.Series(True, index=df.index, dtype=bool)
            image_indices = df.index[df["modality"] == "image"].tolist()
            if len(image_indices) > 1:
                keep.loc[image_indices[1]] = False
            return keep

    def _row(sample_id: str, position: int, modality: str, text: str | None = None) -> dict:
        return {
            "sample_id": sample_id,
            "position": position,
            "modality": modality,
            "content_type": "text/plain" if modality == "text" else "image/png",
            "text_content": text,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        }

    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        _row("s1", 0, "text", "intro"),
        _row("s1", 1, "image"),
        _row("s1", 2, "text", "middle"),
        _row("s1", 3, "image"),
        _row("s1", 4, "text", "end"),
    ]
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    task = InterleavedBatch(task_id="e2e_order", dataset_name="d", data=table)

    filter_stage = _DropSecondImage(drop_invalid_rows=False)
    filtered_task = filter_stage.process(task)

    out_dir = str(tmp_path / "e2e_out")
    writer = InterleavedParquetWriterStage(path=out_dir, materialize_on_write=False, mode="overwrite")
    write_task = writer.process(filtered_task)
    written = pd.read_parquet(write_task.data[0])

    meta = written[written["modality"] == "metadata"]
    content = written[written["modality"] != "metadata"].sort_values("position")

    assert meta["position"].tolist() == [-1]
    assert content["position"].tolist() == [0, 1, 2, 3]
    assert content["modality"].tolist() == ["text", "image", "text", "text"]
    assert content["text_content"].tolist()[0] == "intro"
    assert content["text_content"].tolist()[2] == "middle"
    assert content["text_content"].tolist()[3] == "end"


def test_writer_write_kwargs_cannot_override_index_false(tmp_path: Path) -> None:
    """User-supplied write_kwargs must not be able to override index=False."""
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            }
        ]
    )
    df.index = pd.Index([42])
    task = InterleavedBatch(task_id="kwargs_override", dataset_name="test", data=df)
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "override_out"),
        materialize_on_write=False,
        mode="overwrite",
        write_kwargs={"index": True},
    )
    write_task = writer.process(task)
    schema = pq.read_schema(write_task.data[0])
    assert "__index_level_0__" not in schema.names, "index=True in write_kwargs must not leak index into parquet"


def _build_tar(tar_path: Path, sample_id: str, payload: dict, image_bytes: bytes = b"fake-img") -> str:
    with tarfile.open(tar_path, "w") as tf:
        json_blob = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name=f"{sample_id}.json")
        json_info.size = len(json_blob)
        tf.addfile(json_info, BytesIO(json_blob))

        img_info = tarfile.TarInfo(name=f"{sample_id}.tiff")
        img_info.size = len(image_bytes)
        tf.addfile(img_info, BytesIO(image_bytes))
    return str(tar_path)


def test_heterogeneous_passthrough_fields_combine_as_nullable(tmp_path: Path) -> None:
    """Two shards with different extra fields produce parquet files that combine
    into a unified schema where missing passthrough columns are null."""
    shard_a = _build_tar(
        tmp_path / "shard_a.tar",
        sample_id="doc_a",
        payload={
            "pdf_name": "a.pdf",
            "url": "https://example.com/a",
            "texts": ["hello"],
            "images": [None],
            "score": 0.95,
        },
    )
    shard_b = _build_tar(
        tmp_path / "shard_b.tar",
        sample_id="doc_b",
        payload={
            "pdf_name": "b.pdf",
            "url": "https://example.com/b",
            "texts": ["world"],
            "images": [None],
            "language": "en",
        },
    )

    reader = WebdatasetReaderStage(source_id_field="pdf_name")
    batch_a = reader.process(FileGroupTask(task_id="a", dataset_name="d", data=[shard_a]))
    batch_b = reader.process(FileGroupTask(task_id="b", dataset_name="d", data=[shard_b]))
    assert isinstance(batch_a, InterleavedBatch)
    assert isinstance(batch_b, InterleavedBatch)

    out_dir = tmp_path / "combined_out"
    writer = InterleavedParquetWriterStage(
        path=str(out_dir),
        materialize_on_write=False,
        mode="overwrite",
    )
    writer.process(batch_a)
    writer.process(batch_b)

    parquet_files = sorted(out_dir.glob("*.parquet"))
    assert len(parquet_files) == 2
    tables = [pq.read_table(f) for f in parquet_files]
    combined = pa.concat_tables(tables, promote_options="default").to_pandas()

    all_columns = set(combined.columns)
    assert "url" in all_columns
    assert "score" in all_columns
    assert "language" in all_columns
    assert all_columns >= set(RESERVED_COLUMNS) - {"binary_content"}

    rows_a = combined[combined["sample_id"] == "doc_a"]
    rows_b = combined[combined["sample_id"] == "doc_b"]
    assert not rows_a.empty
    assert not rows_b.empty

    meta_a = rows_a[rows_a["position"] == -1].iloc[0]
    meta_b = rows_b[rows_b["position"] == -1].iloc[0]

    assert meta_a["url"] == "https://example.com/a"
    assert meta_a["score"] == 0.95
    assert pd.isna(meta_a["language"])

    assert meta_b["url"] == "https://example.com/b"
    assert meta_b["language"] == "en"
    assert pd.isna(meta_b["score"])


# --- writer edge cases ---


def test_writer_uses_uuid_when_no_source_files(tmp_path: Path) -> None:
    """Writer falls back to UUID filename when task has no source_files metadata."""
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            }
        ]
    )
    task = InterleavedBatch(task_id="no_source", dataset_name="test", data=df, _metadata={})
    out_dir = tmp_path / "uuid_out"
    writer = InterleavedParquetWriterStage(
        path=str(out_dir),
        materialize_on_write=False,
        mode="overwrite",
    )
    write_task = writer.process(task)
    assert len(write_task.data) == 1
    assert Path(write_task.data[0]).exists()


def test_writer_no_materialize_preserves_null_binary(tmp_path: Path) -> None:
    """materialize_on_write=False leaves binary_content null even for image rows."""
    table = pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": None,
                "source_ref": InterleavedBatch.build_source_ref(path="/fake/img.jpg", member=None),
                "materialize_error": None,
            }
        ],
        schema=INTERLEAVED_SCHEMA,
    )
    task = InterleavedBatch(
        task_id="no_mat",
        dataset_name="test",
        data=table,
        _metadata={"source_files": ["/fake/img.jpg"]},
    )
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "no_mat_out"),
        materialize_on_write=False,
        mode="overwrite",
    )
    write_task = writer.process(task)
    written = pd.read_parquet(write_task.data[0])
    assert pd.isna(written.loc[0, "binary_content"])


@pytest.mark.parametrize(
    "compression",
    [pytest.param("gzip", id="gzip"), pytest.param("snappy", id="snappy")],
)
def test_writer_custom_compression(tmp_path: Path, compression: str) -> None:
    """Custom compression in write_kwargs is used in the written parquet."""
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            }
        ]
    )
    task = InterleavedBatch(
        task_id="comp",
        dataset_name="test",
        data=df,
        _metadata={"source_files": ["test.tar"]},
    )
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / f"{compression}_out"),
        materialize_on_write=False,
        mode="overwrite",
        write_kwargs={"compression": compression},
    )
    write_task = writer.process(task)
    meta = pq.read_metadata(write_task.data[0])
    actual = meta.row_group(0).column(0).compression.lower()
    assert actual == compression
