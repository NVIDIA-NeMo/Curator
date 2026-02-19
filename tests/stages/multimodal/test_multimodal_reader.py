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
import pytest

from nemo_curator.stages.multimodal.io.readers.webdataset import WebdatasetReaderStage
from nemo_curator.tasks import FileGroupTask, MultiBatchTask


def _as_df(task_or_tasks: MultiBatchTask | list[MultiBatchTask]) -> pd.DataFrame:
    task = task_or_tasks[0] if isinstance(task_or_tasks, list) else task_or_tasks
    return task.to_pandas()


def _write_tar_sample(
    tar_path: Path,
    payload: dict[str, object],
    *,
    json_name: str = "sample.json",
    image_name: str = "image.jpg",
    image_bytes: bytes = b"abc",
) -> None:
    with tarfile.open(tar_path, "w") as tf:
        json_blob = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name=json_name)
        json_info.size = len(json_blob)
        tf.addfile(json_info, BytesIO(json_blob))
        img_info = tarfile.TarInfo(name=image_name)
        img_info.size = len(image_bytes)
        tf.addfile(img_info, BytesIO(image_bytes))


def _task_for_tar(tar_path: Path, task_id: str) -> FileGroupTask:
    return FileGroupTask(
        task_id=task_id,
        dataset_name="custom_dataset",
        data=[str(tar_path)],
        _metadata={"source_files": [str(tar_path)]},
    )


def test_reader_supports_custom_field_mapping(tmp_path: Path) -> None:
    tar_path = tmp_path / "alt-shard-00000.tar"
    payload = {
        "doc_id": "doc-custom",
        "source_doc": "custom.pdf",
        "captions": ["a", "b"],
        "frames": ["custom-image.jpg"],
        "primary_image": "custom-image.jpg",
        "p_hash": "abc123",
    }
    image_bytes = b"custom-image-bytes"
    _write_tar_sample(
        tar_path,
        payload,
        json_name="sample-xyz.meta.json",
        image_name="custom-image.jpg",
        image_bytes=image_bytes,
    )
    task = _task_for_tar(tar_path, "file_group_custom")
    reader = WebdatasetReaderStage(
        sample_id_field="doc_id",
        source_id_field="source_doc",
        texts_field="captions",
        images_field="frames",
        image_member_field="primary_image",
        json_extensions=(".meta.json",),
        materialize_on_read=True,
        fields=("p_hash",),
    )
    df = _as_df(reader.process(task))
    assert ((df["sample_id"] == "doc-custom") & (df["modality"] == "metadata")).any()
    text_rows = df[df["modality"] == "text"]
    assert text_rows["text_content"].tolist() == ["a", "b"]
    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 1
    assert image_rows.iloc[0]["binary_content"] == image_bytes
    assert "p_hash" in df.columns
    assert image_rows.iloc[0]["p_hash"] == "abc123"


def test_reader_reads_all_fields_by_default(tmp_path: Path) -> None:
    tar_path = tmp_path / "all-fields.tar"
    payload = {
        "doc_id": "doc-all",
        "source_doc": "all.pdf",
        "captions": ["hello"],
        "frames": ["image.jpg"],
        "primary_image": "image.jpg",
        "p_hash": "phash-1",
        "score": 0.91,
        "aux": {"page": 3},
    }
    _write_tar_sample(tar_path, payload, json_name="sample.meta.json")
    task = _task_for_tar(tar_path, "all_fields")
    reader = WebdatasetReaderStage(
        sample_id_field="doc_id",
        source_id_field="source_doc",
        texts_field="captions",
        images_field="frames",
        image_member_field="primary_image",
        json_extensions=(".meta.json",),
    )
    df = _as_df(reader.process(task))
    image_row = df[df["modality"] == "image"].iloc[0]
    assert image_row["p_hash"] == "phash-1"
    assert image_row["score"] == 0.91
    assert image_row["aux"] == json.dumps({"page": 3}, ensure_ascii=True)
    assert "captions" not in df.columns
    assert "frames" not in df.columns


def test_reader_uses_resolved_content_key_for_content_type(tmp_path: Path) -> None:
    tar_path = tmp_path / "content-type-resolve.tar"
    payload = {
        "doc_id": "doc-ct",
        "source_doc": "ct.pdf",
        "captions": ["hello"],
        "frames": ["token.png"],
        "primary_image": "fallback.jpg",
    }
    with tarfile.open(tar_path, "w") as tf:
        json_blob = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name="sample.meta.json")
        json_info.size = len(json_blob)
        tf.addfile(json_info, BytesIO(json_blob))
        png_info = tarfile.TarInfo(name="token.png")
        png_info.size = 3
        tf.addfile(png_info, BytesIO(b"png"))
        jpg_info = tarfile.TarInfo(name="fallback.jpg")
        jpg_info.size = 3
        tf.addfile(jpg_info, BytesIO(b"jpg"))

    task = _task_for_tar(tar_path, "content_type_resolve")
    reader = WebdatasetReaderStage(
        sample_id_field="doc_id",
        source_id_field="source_doc",
        texts_field="captions",
        images_field="frames",
        image_member_field="primary_image",
        json_extensions=(".meta.json",),
    )
    df = _as_df(reader.process(task))
    image_row = df[df["modality"] == "image"].iloc[0]
    assert image_row["content_type"] == "image/png"


def test_reader_empty_output_schema_includes_requested_passthrough_fields(tmp_path: Path) -> None:
    tar_path = tmp_path / "empty-no-json.tar"
    with tarfile.open(tar_path, "w") as tf:
        img_info = tarfile.TarInfo(name="image.jpg")
        img_info.size = 3
        tf.addfile(img_info, BytesIO(b"abc"))

    task = _task_for_tar(tar_path, "empty_schema")
    reader = WebdatasetReaderStage(source_id_field="pdf_name", fields=("p_hash",))
    df = _as_df(reader.process(task))
    assert "p_hash" in df.columns


@pytest.mark.parametrize(
    ("task_id", "fields", "error_pattern"),
    [
        ("missing_key", ("p_hash",), "fields not found in source sample"),
        ("reserved_key", ("sample_id",), "fields contains reserved keys"),
    ],
)
def test_reader_fields_validation_errors(
    tmp_path: Path, task_id: str, fields: tuple[str, ...], error_pattern: str
) -> None:
    tar_path = tmp_path / f"{task_id}.tar"
    payload = {"pdf_name": "doc.pdf", "texts": ["t"], "images": []}
    _write_tar_sample(tar_path, payload)
    task = _task_for_tar(tar_path, task_id)
    reader = WebdatasetReaderStage(source_id_field="pdf_name", fields=fields)
    with pytest.raises(ValueError, match=error_pattern):
        _ = reader.process(task)
