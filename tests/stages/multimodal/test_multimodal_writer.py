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

import pandas as pd
import pytest

from nemo_curator.stages.multimodal.io.readers.webdataset import WebdatasetReaderStage
from nemo_curator.stages.multimodal.io.writers.multimodal import MultimodalParquetWriterStage
from nemo_curator.tasks import FileGroupTask, MultiBatchTask


@pytest.fixture
def mint_like_tar(tmp_path):
    tar_path = tmp_path / "shard-00000.tar"
    sample_id = "abc123"
    payload = {
        "pdf_name": "doc.pdf",
        "url": "https://example.com/doc.pdf",
        "texts": ["hello", None, "world"],
        "images": ["page_0_image_1", None, "page_2_image_9"],
        "image_metadata": [{"page": 0}, {"page": 2}],
    }
    image_bytes = b"fake-image-bytes"
    with tarfile.open(tar_path, "w") as tf:
        json_blob = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name=f"{sample_id}.json")
        json_info.size = len(json_blob)
        tf.addfile(json_info, BytesIO(json_blob))

        img_info = tarfile.TarInfo(name=f"{sample_id}.tiff")
        img_info.size = len(image_bytes)
        tf.addfile(img_info, BytesIO(image_bytes))
    return str(tar_path), image_bytes


@pytest.fixture
def input_task(mint_like_tar):
    tar_path, _ = mint_like_tar
    return FileGroupTask(
        task_id="file_group_0",
        dataset_name="mint_test",
        data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )


def test_writer_materializes_and_marks_errors(tmp_path, input_task: FileGroupTask, mint_like_tar) -> None:
    _, image_bytes = mint_like_tar
    reader = WebdatasetReaderStage()
    batch = reader.process(input_task)
    assert isinstance(batch, MultiBatchTask)

    writer = MultimodalParquetWriterStage(path=str(tmp_path / "out"), materialize_on_write=True, mode="overwrite")
    write_task = writer.process(batch)
    out_file = write_task.data[0]

    written = pd.read_parquet(out_file)
    image_rows = written[written["modality"] == "image"]
    assert len(image_rows) > 0
    assert (image_rows["binary_content"].apply(lambda x: x == image_bytes).any())
    assert (image_rows["materialize_error"].isna().any())


def test_writer_marks_materialize_error_on_bad_source_path(tmp_path, input_task: FileGroupTask) -> None:
    reader = WebdatasetReaderStage()
    batch = reader.process(input_task)
    assert isinstance(batch, MultiBatchTask)

    df = batch.to_pandas().copy()
    image_mask = df["modality"] == "image"
    assert image_mask.any()
    first_image_idx = df[image_mask].index[0]
    df.at[first_image_idx, "metadata_source"] = json.dumps(
        {
            "source_id": "doc.pdf",
            "source_shard": "shard-00000.tar",
            "content_path": "/definitely/missing/path.tar",
            "content_key": "abc123.tiff",
        }
    )
    bad_batch = MultiBatchTask(
        task_id=batch.task_id,
        dataset_name=batch.dataset_name,
        data=df,
        _metadata=batch._metadata,
        _stage_perf=batch._stage_perf,
    )

    writer = MultimodalParquetWriterStage(path=str(tmp_path / "out_bad"), materialize_on_write=True, mode="overwrite")
    write_task = writer.process(bad_batch)
    written = pd.read_parquet(write_task.data[0])

    target = written.loc[first_image_idx]
    assert pd.isna(target["binary_content"])
    assert isinstance(target["materialize_error"], str)
