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

import pytest

from nemo_curator.stages.multimodal.io.readers.webdataset import WebdatasetReaderStage
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
    return str(tar_path), sample_id, image_bytes


@pytest.fixture
def input_task(mint_like_tar):
    tar_path, _, _ = mint_like_tar
    return FileGroupTask(
        task_id="file_group_0",
        dataset_name="mint_test",
        data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )


def test_reader_emits_metadata_text_image_rows(input_task: FileGroupTask, mint_like_tar) -> None:
    _, sample_id, _ = mint_like_tar
    reader = WebdatasetReaderStage()
    output = reader.process(input_task)
    assert isinstance(output, MultiBatchTask)

    df = output.to_pandas()
    assert set(df["modality"].unique()) == {"metadata", "text", "image"}
    assert ((df["sample_id"] == sample_id) & (df["modality"] == "metadata") & (df["position"] == -1)).any()
