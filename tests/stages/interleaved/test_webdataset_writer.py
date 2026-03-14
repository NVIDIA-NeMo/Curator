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
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.io.writers.webdataset import InterleavedWebdatasetWriterStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

from .conftest import make_interleaved_batch

_EXTRA_SCHEMA = pa.schema(
    [
        *INTERLEAVED_SCHEMA,
        pa.field("image_metadata", pa.string(), nullable=True),
        pa.field("text_score", pa.string(), nullable=True),
        pa.field("url", pa.string(), nullable=True),
    ]
)


def _make_batch_with_extras(num_samples: int = 1) -> InterleavedBatch:
    """Build a batch with per-image, per-text, and sample-level extra columns."""
    rows: list[dict[str, Any]] = []
    for i in range(num_samples):
        sid = f"sample_{i}"
        rows.append(
            {
                "sample_id": sid,
                "position": -1,
                "modality": "metadata",
                "content_type": "application/json",
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
                "image_metadata": None,
                "text_score": None,
                "url": f"https://example.com/{i}",
            }
        )
        rows.append(
            {
                "sample_id": sid,
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": f"Hello {i}",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
                "image_metadata": None,
                "text_score": json.dumps({"quality": 0.9}),
                "url": None,
            }
        )
        rows.append(
            {
                "sample_id": sid,
                "position": 1,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": b"fake-jpeg-bytes",
                "source_ref": None,
                "materialize_error": None,
                "image_metadata": json.dumps({"height": 100, "width": 200}),
                "text_score": None,
                "url": None,
            }
        )
        rows.append(
            {
                "sample_id": sid,
                "position": 2,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": f"World {i}",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
                "image_metadata": None,
                "text_score": json.dumps({"quality": 0.7}),
                "url": None,
            }
        )
        rows.append(
            {
                "sample_id": sid,
                "position": 3,
                "modality": "image",
                "content_type": "image/png",
                "text_content": None,
                "binary_content": b"fake-png-bytes",
                "source_ref": None,
                "materialize_error": None,
                "image_metadata": json.dumps({"height": 300, "width": 400}),
                "text_score": None,
                "url": None,
            }
        )
    table = pa.Table.from_pylist(rows, schema=_EXTRA_SCHEMA)
    return InterleavedBatch(
        task_id="test_extras",
        dataset_name="test",
        data=table,
        _metadata={"source_files": ["test.tar"]},
    )


def test_write_creates_tar(tmp_path: Path) -> None:
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir()
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    result = writer.process(make_interleaved_batch())
    assert isinstance(result, FileGroupTask)
    assert len(result.data) == 1
    assert result.data[0].endswith(".tar")
    assert Path(result.data[0]).exists()


def test_tar_contains_json_and_image_members(tmp_path: Path) -> None:
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir()
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    result = writer.process(make_interleaved_batch(num_samples=2))
    with tarfile.open(result.data[0], "r") as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
    json_members = [n for n in names if n.endswith(".json")]
    image_members = [n for n in names if not n.endswith(".json")]
    assert len(json_members) == 2
    assert len(image_members) == 2


def test_tar_json_has_texts_and_images(tmp_path: Path) -> None:
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir()
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    result = writer.process(make_interleaved_batch(num_samples=1))
    with tarfile.open(result.data[0], "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".json"):
                payload = json.load(tf.extractfile(member))
                assert "texts" in payload
                assert "images" in payload
                assert payload["texts"] == ["Hello 0", None]
                assert isinstance(payload["images"], list)


def test_write_no_binary_still_records_image_positions(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "s0",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s0",
            "position": 0,
            "modality": "image",
            "content_type": "image/png",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    batch = InterleavedBatch(task_id="t", dataset_name="test", data=table, _metadata={"source_files": ["x.tar"]})
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir()
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    result = writer.process(batch)
    with tarfile.open(result.data[0], "r") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        json_members = [m for m in members if m.name.endswith(".json")]
        image_members = [m for m in members if not m.name.endswith(".json")]
        assert len(json_members) == 1
        assert len(image_members) == 0
        payload = json.load(tf.extractfile(json_members[0]))
        key = json_members[0].name.removesuffix(".json")
        assert payload["images"] == [f"{key}.0.png"]


def _read_first_json(tar_path: str) -> dict[str, Any]:
    """Extract the first JSON member from a tar."""
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".json"):
                return json.load(tf.extractfile(member))
    msg = "No JSON member found in tar"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        pytest.param(
            "image_metadata",
            [{"height": 100, "width": 200}, {"height": 300, "width": 400}],
            id="per_image",
        ),
        pytest.param(
            "text_score",
            [{"quality": 0.9}, {"quality": 0.7}],
            id="per_text",
        ),
        pytest.param(
            "url",
            "https://example.com/0",
            id="sample_level",
        ),
    ],
)
def test_extra_field_round_trip(tmp_path: Path, field: str, expected: object) -> None:
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir()
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    payload = _read_first_json(writer.process(_make_batch_with_extras(num_samples=1)).data[0])
    assert field in payload
    assert payload[field] == expected


def test_json_encoded_sample_field_parsed_back(tmp_path: Path) -> None:
    """A sample-level field stored as a JSON string should be deserialized back."""
    rows: list[dict[str, Any]] = [
        {
            "sample_id": "s0",
            "position": -1,
            "modality": "metadata",
            "content_type": None,
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            "lang": json.dumps({"en": 0.9}),
        },
        {
            "sample_id": "s0",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "Hi",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
            "lang": None,
        },
    ]
    schema = pa.schema([*INTERLEAVED_SCHEMA, pa.field("lang", pa.string(), nullable=True)])
    table = pa.Table.from_pylist(rows, schema=schema)
    batch = InterleavedBatch(task_id="t", dataset_name="test", data=table, _metadata={"source_files": ["x.tar"]})
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir()
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    result = writer.process(batch)
    payload = _read_first_json(result.data[0])

    assert payload["lang"] == {"en": 0.9}


# ---------------------------------------------------------------------------
# Round-trip tests (write -> read)
# ---------------------------------------------------------------------------


def _write_and_get_tar(tmp_path: Path, batch: InterleavedBatch) -> str:
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir(exist_ok=True)
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    result = writer.process(batch)
    return result.data[0]


def test_images_list_entries_are_tar_members(tmp_path: Path) -> None:
    """Every non-null entry in the JSON ``images`` list must be an actual tar member."""
    tar_path = _write_and_get_tar(tmp_path, make_interleaved_batch(num_samples=2))
    with tarfile.open(tar_path, "r") as tf:
        member_names = {m.name for m in tf.getmembers()}
        for member in tf.getmembers():
            if not member.name.endswith(".json"):
                continue
            payload = json.load(tf.extractfile(member))
            for img_ref in payload.get("images", []):
                if img_ref is not None:
                    assert img_ref in member_names, f"{img_ref!r} not found in tar members"


def test_key_preserves_sample_id(tmp_path: Path) -> None:
    """Tar key (JSON member stem) must equal the original sample_id."""
    tar_path = _write_and_get_tar(tmp_path, make_interleaved_batch(num_samples=3))
    with tarfile.open(tar_path, "r") as tf:
        stems = sorted(m.name.removesuffix(".json") for m in tf.getmembers() if m.name.endswith(".json"))
    assert stems == ["sample_0", "sample_1", "sample_2"]


def test_key_escapes_unsafe_chars(tmp_path: Path) -> None:
    """Sample_ids with dots, slashes, colons get percent-encoded in the tar key."""
    rows = [
        {
            "sample_id": "dir/doc.v2",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "dir/doc.v2",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "hello",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    batch = InterleavedBatch(task_id="t", dataset_name="test", data=table, _metadata={"source_files": ["x.tar"]})
    tar_path = _write_and_get_tar(tmp_path, batch)
    with tarfile.open(tar_path, "r") as tf:
        json_names = [m.name for m in tf.getmembers() if m.name.endswith(".json")]
    assert json_names == ["dir%2Fdoc%2Ev2.json"]


def test_key_escape_no_collisions(tmp_path: Path) -> None:
    """sample.001 and sample_001 must produce distinct tar keys."""
    rows = []
    for sid in ("sample.001", "sample_001"):
        rows.append(
            {
                "sample_id": sid,
                "position": -1,
                "modality": "metadata",
                "content_type": "application/json",
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            }
        )
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    batch = InterleavedBatch(task_id="t", dataset_name="test", data=table, _metadata={"source_files": ["x.tar"]})
    tar_path = _write_and_get_tar(tmp_path, batch)
    with tarfile.open(tar_path, "r") as tf:
        json_names = sorted(m.name for m in tf.getmembers() if m.name.endswith(".json"))
    assert len(json_names) == 2
    assert json_names[0] != json_names[1]


def test_write_read_round_trip_structure(tmp_path: Path) -> None:
    """Write -> read produces same row count, doc count, modalities, and sample_ids."""
    from nemo_curator.stages.interleaved.io.readers.webdataset import WebdatasetReaderStage

    batch = make_interleaved_batch(num_samples=2)
    orig_df = batch.to_pandas()
    tar_path = _write_and_get_tar(tmp_path, batch)

    reader = WebdatasetReaderStage()
    task = FileGroupTask(task_id="rt", dataset_name="test", data=[tar_path], _metadata={})
    result = reader.process(task)
    rt_df = result.to_pandas() if not isinstance(result, list) else result[0].to_pandas()

    assert len(rt_df) == len(orig_df)
    assert rt_df["sample_id"].nunique() == orig_df["sample_id"].nunique()
    assert sorted(rt_df["sample_id"].unique()) == sorted(orig_df["sample_id"].unique())
    assert set(rt_df["modality"].unique()) == set(orig_df["modality"].unique())


def test_write_read_round_trip_materialization(tmp_path: Path) -> None:
    """Write with image bytes, read with materialize_on_read=True -> bytes round-trip."""
    from nemo_curator.stages.interleaved.io.readers.webdataset import WebdatasetReaderStage

    batch = make_interleaved_batch(num_samples=1)
    tar_path = _write_and_get_tar(tmp_path, batch)

    reader = WebdatasetReaderStage(materialize_on_read=True)
    task = FileGroupTask(task_id="rt", dataset_name="test", data=[tar_path], _metadata={})
    result = reader.process(task)
    rt_df = result.to_pandas() if not isinstance(result, list) else result[0].to_pandas()

    img_rows = rt_df[rt_df["modality"] == "image"]
    assert len(img_rows) == 1
    assert img_rows["binary_content"].notna().all()
    assert bytes(img_rows["binary_content"].iloc[0]) == b"fake-jpeg-bytes"


def test_write_read_round_trip_extra_columns(tmp_path: Path) -> None:
    """Extra columns (url, scores) survive the write -> read round-trip on metadata rows."""
    from nemo_curator.stages.interleaved.io.readers.webdataset import WebdatasetReaderStage

    batch = _make_batch_with_extras(num_samples=1)
    tar_path = _write_and_get_tar(tmp_path, batch)

    reader = WebdatasetReaderStage()
    task = FileGroupTask(task_id="rt", dataset_name="test", data=[tar_path], _metadata={})
    result = reader.process(task)
    rt_df = result.to_pandas() if not isinstance(result, list) else result[0].to_pandas()

    meta = rt_df[rt_df["modality"] == "metadata"].iloc[0]
    assert meta["url"] == "https://example.com/0"
    assert "image_metadata" in rt_df.columns
    assert "text_score" in rt_df.columns
