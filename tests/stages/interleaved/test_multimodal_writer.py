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

from nemo_curator.stages.interleaved.io.readers.webdataset import InterleavedWebdatasetReaderStage
from nemo_curator.stages.interleaved.io.writers.tabular import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.io.writers.webdataset import (
    InterleavedWebdatasetWriterStage,
    _escape_key,
    _ext_from_content_type,
    _is_null,
)
from nemo_curator.stages.interleaved.stages import BaseInterleavedScoreFilterStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA, RESERVED_COLUMNS

from .conftest import make_row


def _read_batch(input_task: FileGroupTask) -> InterleavedBatch:
    batch = InterleavedWebdatasetReaderStage().process(input_task)
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

    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "out_bad"),
        materialize_on_write=True,
        mode="overwrite",
        on_materialize_error="warn",
    )
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
    """End-to-end: filter stage annotates all rows; parquet roundtrip preserves order."""

    class _DropSecondImage(BaseInterleavedScoreFilterStage):
        name: str = "drop_second_image"

        def annotation_columns(self, task: InterleavedBatch, df: pd.DataFrame) -> dict[str, pd.Series]:
            s = pd.Series(0.0, index=df.index, dtype=float)
            image_indices = df.index[df["modality"] == "image"].tolist()
            if len(image_indices) > 1:
                s.loc[image_indices[1]] = 1.0
            return {f"{self.name}_second_image_flag": s}

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

    filter_stage = _DropSecondImage()
    filtered_task = filter_stage.process(task)

    out_dir = str(tmp_path / "e2e_out")
    writer = InterleavedParquetWriterStage(path=out_dir, materialize_on_write=False, mode="overwrite")
    write_task = writer.process(filtered_task)
    written = pd.read_parquet(write_task.data[0])

    meta = written[written["modality"] == "metadata"]
    content = written[written["modality"] != "metadata"].sort_values("position")

    assert meta["position"].tolist() == [-1]
    assert content["position"].tolist() == [0, 1, 2, 3, 4]
    assert content["modality"].tolist() == ["text", "image", "text", "image", "text"]
    assert content["text_content"].tolist()[0] == "intro"
    assert content["text_content"].tolist()[2] == "middle"
    assert content["text_content"].tolist()[4] == "end"


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

    reader = InterleavedWebdatasetReaderStage()
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


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_wds_batch(
    sample_id: str = "s1",
    text: str = "hello",
    image_bytes: bytes | None = b"fake-img",
    extra_cols: dict | None = None,
    source_files: list[str] | None = None,
) -> InterleavedBatch:
    """Build a minimal metadata+text+image InterleavedBatch for WDS writer tests."""
    rows: list[dict] = [
        {
            "sample_id": sample_id,
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            **(extra_cols or {}),
        },
        {
            "sample_id": sample_id,
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": text,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            **(extra_cols or {}),
        },
    ]
    if image_bytes is not None:
        rows.append(
            {
                "sample_id": sample_id,
                "position": 1,
                "modality": "image",
                "content_type": "image/png",
                "text_content": None,
                "binary_content": image_bytes,
                "source_ref": None,
                "materialize_error": None,
                **(extra_cols or {}),
            }
        )
    return InterleavedBatch(
        task_id=f"wds_{sample_id}",
        dataset_name="test",
        data=pd.DataFrame(rows),
        _metadata={"source_files": source_files or ["test.tar"]},
    )


def _read_wds_tar(tar_path: str) -> tuple[dict, dict[str, bytes]]:
    """Read a WDS tar: returns (json_payload, {member_name: bytes})."""
    payload: dict = {}
    members: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            members[m.name] = data
            if m.name.endswith(".json"):
                payload = json.loads(data)
    return payload, members


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


def test_escape_key_encodes_special_chars() -> None:
    assert "/" not in _escape_key("a/b")
    assert ":" not in _escape_key("a:b")
    assert _escape_key("simple") == "simple"
    assert _escape_key("a/b:c") == "a%2Fb%3Ac"


def test_ext_from_content_type_known() -> None:
    assert _ext_from_content_type("image/jpeg") == "jpg"
    assert _ext_from_content_type("image/png") == "png"
    assert _ext_from_content_type("image/tiff") == "tiff"


def test_ext_from_content_type_fallback() -> None:
    assert _ext_from_content_type(None) == "bin"
    assert _ext_from_content_type("application/octet-stream") == "bin"


# ---------------------------------------------------------------------------
# InterleavedWebdatasetWriterStage tests
# ---------------------------------------------------------------------------


def test_wds_writer_roundtrip(tmp_path: Path) -> None:
    """Write a WDS batch, read it back; text and image content are preserved."""
    batch = _make_wds_batch(sample_id="doc1", text="roundtrip text", image_bytes=b"img-data")
    writer = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "wds_out"),
        materialize_on_write=False,
        mode="overwrite",
    )
    write_task = writer.process(batch)
    tar_path = write_task.data[0]
    assert tar_path.endswith(".tar")

    read_task = FileGroupTask(task_id="rt", dataset_name="test", data=[tar_path])
    result = InterleavedWebdatasetReaderStage(sample_id_field="sample_id").process(read_task)
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()

    assert "doc1" in df["sample_id"].tolist()
    text_rows = df[df["modality"] == "text"]
    assert text_rows["text_content"].tolist() == ["roundtrip text"]
    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 1


def test_wds_writer_text_only_sample(tmp_path: Path) -> None:
    """No image rows → tar has JSON with all-None images list."""
    batch = _make_wds_batch(sample_id="text_only", image_bytes=None)
    writer = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "text_only_out"),
        materialize_on_write=False,
        mode="overwrite",
    )
    write_task = writer.process(batch)
    payload, members = _read_wds_tar(write_task.data[0])

    assert all(v is None for v in payload["images"])
    assert not any(k.endswith((".png", ".jpg")) for k in members)


def test_wds_writer_unsupported_modality_raises(tmp_path: Path) -> None:
    """A modality='video' row raises ValueError naming the bad modality."""
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "video",
                "content_type": "video/mp4",
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            }
        ]
    )
    task = InterleavedBatch(task_id="vid", dataset_name="t", data=df, _metadata={"source_files": ["x.tar"]})
    writer = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "vid_out"), materialize_on_write=False, mode="overwrite"
    )
    with pytest.raises(ValueError, match="video"):
        writer.process(task)


def test_wds_writer_key_escaping(tmp_path: Path) -> None:
    """sample_id with special chars → tar members have safe names; roundtrip recovers original id."""
    sample_id = "a/b:c.d"
    batch = _make_wds_batch(sample_id=sample_id, image_bytes=None)
    writer = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "escape_out"), materialize_on_write=False, mode="overwrite"
    )
    write_task = writer.process(batch)
    _, members = _read_wds_tar(write_task.data[0])

    for name in members:
        assert "/" not in name.split(".json")[0], f"unescaped slash in member name: {name}"
        assert ":" not in name.split(".json")[0], f"unescaped colon in member name: {name}"

    read_task = FileGroupTask(task_id="esc_rt", dataset_name="test", data=[write_task.data[0]])
    result = InterleavedWebdatasetReaderStage(sample_id_field="sample_id").process(read_task)
    assert isinstance(result, InterleavedBatch)
    assert sample_id in result.to_pandas()["sample_id"].tolist()


def test_wds_writer_passthrough_columns_in_json(tmp_path: Path) -> None:
    """Extra 'url' col in metadata row appears in the written JSON payload."""
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
            "url": "https://example.com",
        },
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "hi",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            "url": None,
        },
    ]
    task = InterleavedBatch(
        task_id="pt", dataset_name="t", data=pd.DataFrame(rows), _metadata={"source_files": ["x.tar"]}
    )
    writer = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "pt_out"), materialize_on_write=False, mode="overwrite"
    )
    write_task = writer.process(task)
    payload, _ = _read_wds_tar(write_task.data[0])
    assert payload.get("url") == "https://example.com"


def test_wds_writer_null_binary_skips_member(tmp_path: Path) -> None:
    """null binary_content with materialize_on_write=False → no image member written."""
    table = pa.Table.from_pylist(
        [
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
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "image",
                "content_type": "image/png",
                "text_content": None,
                "binary_content": None,
                "source_ref": InterleavedBatch.build_source_ref(path="/fake/img.png", member=None),
                "materialize_error": None,
            },
        ],
        schema=INTERLEAVED_SCHEMA,
    )
    task = InterleavedBatch(
        task_id="null_bin", dataset_name="t", data=table, _metadata={"source_files": ["/fake/img.png"]}
    )
    writer = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "null_bin_out"), materialize_on_write=False, mode="overwrite"
    )
    write_task = writer.process(task)
    payload, members = _read_wds_tar(write_task.data[0])

    assert any(k.endswith(".json") for k in members)
    assert not any(k.endswith(".png") for k in members)
    assert any(v is not None for v in payload.get("images", []))


def test_wds_writer_deterministic_filename(tmp_path: Path) -> None:
    """Same source_files + task_id → same output filename across two writer instances."""
    source = ["shard-00000.tar"]
    task_id = "fixed_task"
    batch = InterleavedBatch(
        task_id=task_id,
        dataset_name="test",
        data=_make_wds_batch(sample_id="s1", source_files=source).to_pandas(),
        _metadata={"source_files": source},
    )

    out1 = tmp_path / "det_out1"
    out2 = tmp_path / "det_out2"
    t1 = InterleavedWebdatasetWriterStage(path=str(out1), materialize_on_write=False, mode="overwrite").process(batch)
    t2 = InterleavedWebdatasetWriterStage(path=str(out2), materialize_on_write=False, mode="overwrite").process(batch)
    assert Path(t1.data[0]).name == Path(t2.data[0]).name


def test_wds_writer_per_image_fields_roundtrip(tmp_path: Path) -> None:
    """per_image_fields are written as a list in JSON and survive a write→read round-trip."""
    fake_png = b"\x89PNG\r\n"
    rows = [
        # image_alt_text is null on the metadata row — it's a per-image field, not sample-level
        make_row("s1", -1, "metadata", url="http://example.com", image_alt_text=None),
        make_row("s1", 0, "text", text_content="intro text", url=None, image_alt_text=None),
        make_row(
            "s1",
            1,
            "image",
            content_type="image/png",
            binary_content=fake_png,
            url=None,
            image_alt_text="a cat on a chair",
        ),
        make_row("s1", 2, "image", content_type="image/png", binary_content=fake_png, url=None, image_alt_text=None),
        make_row(
            "s1",
            3,
            "image",
            content_type="image/png",
            binary_content=fake_png,
            url=None,
            image_alt_text="a dog in a park",
        ),
    ]
    task = InterleavedBatch(
        task_id="per_img", dataset_name="t", data=pd.DataFrame(rows), _metadata={"source_files": ["x.tar"]}
    )
    writer = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "per_img_out"), materialize_on_write=False, mode="overwrite"
    )
    write_task = writer.process(task)

    payload, _ = _read_wds_tar(write_task.data[0])
    assert "image_alt_text" in payload, "image_alt_text missing from JSON payload"
    assert payload["image_alt_text"] == ["a cat on a chair", None, "a dog in a park"]
    # sample-level passthrough still present
    assert payload["url"] == "http://example.com"

    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="sample_id",
        per_image_fields=("image_alt_text",),
    )
    result = reader.process(FileGroupTask(task_id="rt", dataset_name="t", data=write_task.data))
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()

    image_rows = df[df["modality"] == "image"].sort_values("position")
    alt_texts = [None if pd.isna(v) else v for v in image_rows["image_alt_text"]]
    assert alt_texts == ["a cat on a chair", None, "a dog in a park"]


def test_wds_writer_per_text_fields_roundtrip(tmp_path: Path) -> None:
    """per_text_fields are written as a list in JSON and survive a write→read round-trip."""
    rows = [
        make_row("s1", -1, "metadata", text_confidence=None),
        make_row("s1", 0, "text", text_content="first paragraph", text_confidence=0.95),
        make_row("s1", 1, "text", text_content="second paragraph", text_confidence=None),  # gap
        make_row("s1", 2, "text", text_content="third paragraph", text_confidence=0.72),
    ]
    task = InterleavedBatch(
        task_id="per_txt", dataset_name="t", data=pd.DataFrame(rows), _metadata={"source_files": ["x.tar"]}
    )
    writer = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "per_txt_out"), materialize_on_write=False, mode="overwrite"
    )
    write_task = writer.process(task)

    payload, _ = _read_wds_tar(write_task.data[0])
    assert "text_confidence" in payload, "text_confidence missing from JSON payload"
    assert payload["text_confidence"] == [0.95, None, 0.72]

    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="sample_id",
        per_text_fields=("text_confidence",),
    )
    result = reader.process(FileGroupTask(task_id="rt2", dataset_name="t", data=write_task.data))
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()

    text_rows = df[df["modality"] == "text"].sort_values("position")
    confidences = [None if pd.isna(v) else v for v in text_rows["text_confidence"]]
    assert confidences == [0.95, None, 0.72]


def test_wds_writer_non_ascii_per_image_field_roundtrip(tmp_path: Path) -> None:
    """Non-ASCII per_image_fields are written as raw UTF-8, not \\uXXXX escapes."""
    alt_texts = ["北极熊", None, "الدب القطبي"]  # Chinese, gap, Arabic
    fake_png = b"\x89PNG"
    rows = [
        make_row("s1", -1, "metadata", image_alt_text=None),
        make_row("s1", 0, "image", content_type="image/png", binary_content=fake_png, image_alt_text=alt_texts[0]),
        make_row("s1", 1, "image", content_type="image/png", binary_content=fake_png, image_alt_text=alt_texts[1]),
        make_row("s1", 2, "image", content_type="image/png", binary_content=fake_png, image_alt_text=alt_texts[2]),
    ]
    task = InterleavedBatch(
        task_id="non_ascii", dataset_name="t", data=pd.DataFrame(rows), _metadata={"source_files": ["x.tar"]}
    )
    writer = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "non_ascii_out"), materialize_on_write=False, mode="overwrite"
    )
    write_task = writer.process(task)

    _, members = _read_wds_tar(write_task.data[0])
    raw_bytes = next(v for k, v in members.items() if k.endswith(".json"))
    assert "\\u" not in raw_bytes.decode("utf-8"), "Non-ASCII chars should not be \\u-escaped"
    assert "北极熊".encode() in raw_bytes
    assert "الدب القطبي".encode() in raw_bytes

    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="sample_id",
        per_image_fields=("image_alt_text",),
    )
    result = reader.process(FileGroupTask(task_id="rt3", dataset_name="t", data=write_task.data))
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()
    image_rows = df[df["modality"] == "image"].sort_values("position")
    recovered = [None if pd.isna(v) else v for v in image_rows["image_alt_text"]]
    assert recovered == alt_texts


def test_wds_writer_mixed_modality_field_written_as_position_aligned_list(tmp_path: Path) -> None:
    """A column with non-null values in BOTH image and text rows is written as a
    position-aligned list (one entry per content row).  When read back without
    per_image_fields / per_text_fields, the entire list lands on the metadata row
    as a passthrough JSON string."""
    fake_png = b"\x89PNG"
    rows = [
        make_row("s1", -1, "metadata", confidence=None),
        make_row("s1", 0, "text", text_content="intro", confidence="text_conf_A"),
        make_row("s1", 1, "image", content_type="image/png", binary_content=fake_png, confidence="img_conf_B"),
        make_row("s1", 2, "text", text_content="outro", confidence=None),
    ]
    task = InterleavedBatch(
        task_id="mixed", dataset_name="t", data=pd.DataFrame(rows), _metadata={"source_files": ["x.tar"]}
    )
    write_task = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "mixed_out"), materialize_on_write=False, mode="overwrite"
    ).process(task)

    payload, _ = _read_wds_tar(write_task.data[0])
    assert payload["confidence"] == ["text_conf_A", "img_conf_B", None]

    # Without per_image/per_text declaration, reader treats it as a sample-level passthrough
    reader = InterleavedWebdatasetReaderStage(sample_id_field="sample_id")
    result = reader.process(FileGroupTask(task_id="rt_mixed", dataset_name="t", data=write_task.data))
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()
    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert json.loads(meta_row["confidence"]) == ["text_conf_A", "img_conf_B", None]


def test_is_null_null_variants() -> None:
    assert _is_null(None) is True
    assert _is_null(float("nan")) is True
    assert _is_null(pd.NA) is True


def test_is_null_non_null_and_uncomparable() -> None:
    class _Uncomparable:
        __hash__ = None  # type: ignore[assignment]

        def __eq__(self, other: object) -> bool:
            msg = "cannot compare"
            raise TypeError(msg)

    for v in ("hello", 42, 0, b"bytes", _Uncomparable()):
        assert _is_null(v) is False
