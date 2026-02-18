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
import pytest

from nemo_curator.stages.multimodal.stages import MultimodalJpegAspectRatioFilterStage
from nemo_curator.stages.multimodal.utils import load_bytes_from_content_reference
from nemo_curator.tasks import MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA


@pytest.fixture
def single_row_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "metadata_source": json.dumps(
                    {
                        "source_id": "doc.pdf",
                        "source_shard": "shard-00000.tar",
                        "content_path": "/dataset/shard-00000.tar",
                        "content_key": "s1.json",
                    }
                ),
                "metadata_json": None,
                "materialize_error": None,
            }
        ],
        schema=MULTIMODAL_SCHEMA,
    )


@pytest.fixture
def single_row_task(single_row_table: pa.Table) -> MultiBatchTask:
    return MultiBatchTask(task_id="t1", dataset_name="d1", data=single_row_table)


def test_with_parsed_source_columns(single_row_task: MultiBatchTask) -> None:
    df = single_row_task.with_parsed_source_columns()
    assert df.loc[0, "_src_source_id"] == "doc.pdf"
    assert df.loc[0, "_src_source_shard"] == "shard-00000.tar"
    assert df.loc[0, "_src_content_path"] == "/dataset/shard-00000.tar"
    assert df.loc[0, "_src_content_key"] == "s1.json"


def test_load_bytes_from_content_reference_direct_and_keyed(tmp_path: Path) -> None:
    direct_path = tmp_path / "direct.bin"
    direct_payload = b"direct-bytes"
    direct_path.write_bytes(direct_payload)
    tar_path = tmp_path / "blob.tar"
    tar_payload = b"tar-bytes"
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="x.bin")
        info.size = len(tar_payload)
        tf.addfile(info, BytesIO(tar_payload))

    cache: dict[tuple[str, str], bytes | None] = {}
    assert load_bytes_from_content_reference(str(direct_path), None, {}, cache) == direct_payload
    assert load_bytes_from_content_reference(str(tar_path), "x.bin", {}, cache) == tar_payload


def test_jpeg_filter_handles_non_default_dataframe_index() -> None:
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "ok",
                "binary_content": None,
                "metadata_source": None,
                "metadata_json": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s1",
                "position": 1,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": b"not-a-valid-jpeg",
                "metadata_source": None,
                "metadata_json": None,
                "materialize_error": None,
            },
        ]
    )
    df.index = pd.Index([10, 42])
    task = MultiBatchTask(task_id="non_default_index", dataset_name="d1", data=df)
    stage = MultimodalJpegAspectRatioFilterStage(drop_invalid_rows=False)
    out = stage.process(task).to_pandas()
    assert len(out) == 1
    assert out.iloc[0]["modality"] == "text"
