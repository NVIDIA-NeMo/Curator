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


def test_reader_emits_metadata_text_image_rows(
    input_task: FileGroupTask, mint_like_tar: tuple[str, str, bytes]
) -> None:
    _, sample_id, _ = mint_like_tar
    reader = WebdatasetReaderStage(source_id_field="pdf_name")
    df = _as_df(reader.process(input_task))
    assert set(df["modality"].unique()) == {"metadata", "text", "image"}
    assert ((df["sample_id"] == sample_id) & (df["modality"] == "metadata") & (df["position"] == -1)).any()


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
    with tarfile.open(tar_path, "w") as tf:
        json_blob = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name="sample-xyz.meta.json")
        json_info.size = len(json_blob)
        tf.addfile(json_info, BytesIO(json_blob))

        img_info = tarfile.TarInfo(name="custom-image.jpg")
        img_info.size = len(image_bytes)
        tf.addfile(img_info, BytesIO(image_bytes))

    task = FileGroupTask(
        task_id="file_group_custom",
        dataset_name="custom_dataset",
        data=[str(tar_path)],
        _metadata={"source_files": [str(tar_path)]},
    )
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


def test_reader_propagates_source_storage_options(input_task: FileGroupTask) -> None:
    reader = WebdatasetReaderStage(source_id_field="pdf_name", read_kwargs={"storage_options": {"anon": False}})
    output = reader.process(input_task)
    assert isinstance(output, MultiBatchTask)  # metadata lives on task, not dataframe
    assert output._metadata.get("source_storage_options") == {"anon": False}


def test_reader_materialize_on_read_flag(input_task: FileGroupTask) -> None:
    reader = WebdatasetReaderStage(source_id_field="pdf_name", materialize_on_read=True)
    df = _as_df(reader.process(input_task))
    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) > 0
    assert image_rows["binary_content"].notna().any()


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
    with tarfile.open(tar_path, "w") as tf:
        blob = json.dumps(payload).encode("utf-8")
        info = tarfile.TarInfo(name="sample.meta.json")
        info.size = len(blob)
        tf.addfile(info, BytesIO(blob))
        img = tarfile.TarInfo(name="image.jpg")
        img.size = 3
        tf.addfile(img, BytesIO(b"abc"))

    task = FileGroupTask(
        task_id="all_fields",
        dataset_name="custom_dataset",
        data=[str(tar_path)],
        _metadata={"source_files": [str(tar_path)]},
    )
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


def test_reader_fields_raises_for_missing_key(tmp_path: Path) -> None:
    tar_path = tmp_path / "missing-key.tar"
    payload = {"pdf_name": "doc.pdf", "texts": ["t"], "images": []}
    with tarfile.open(tar_path, "w") as tf:
        blob = json.dumps(payload).encode("utf-8")
        info = tarfile.TarInfo(name="sample.json")
        info.size = len(blob)
        tf.addfile(info, BytesIO(blob))

    task = FileGroupTask(
        task_id="missing_key",
        dataset_name="custom_dataset",
        data=[str(tar_path)],
        _metadata={"source_files": [str(tar_path)]},
    )
    reader = WebdatasetReaderStage(source_id_field="pdf_name", fields=("p_hash",))
    with pytest.raises(ValueError, match="fields not found in source sample"):
        _ = reader.process(task)


def test_reader_fields_raises_for_reserved_key(tmp_path: Path) -> None:
    tar_path = tmp_path / "reserved-key.tar"
    payload = {"pdf_name": "doc.pdf", "texts": ["t"], "images": []}
    with tarfile.open(tar_path, "w") as tf:
        blob = json.dumps(payload).encode("utf-8")
        info = tarfile.TarInfo(name="sample.json")
        info.size = len(blob)
        tf.addfile(info, BytesIO(blob))

    task = FileGroupTask(
        task_id="reserved_key",
        dataset_name="custom_dataset",
        data=[str(tar_path)],
        _metadata={"source_files": [str(tar_path)]},
    )
    reader = WebdatasetReaderStage(source_id_field="pdf_name", fields=("sample_id",))
    with pytest.raises(ValueError, match="fields contains reserved keys"):
        _ = reader.process(task)
