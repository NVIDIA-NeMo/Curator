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

from pathlib import Path

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.io.writers.webdataset import InterleavedWebdatasetWriterStage
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA


@pytest.mark.parametrize("modality", ["image", "text"])
@pytest.mark.parametrize("position", [0, 2])
@pytest.mark.parametrize("binary", [None, b"original image bytes"])
def test_single_content_matches_empty_metadata_sample(
    tmp_path: Path, modality: str, position: int, binary: bytes | None
) -> None:
    schema = pa.schema(
        [
            *INTERLEAVED_SCHEMA,
            pa.field("caption", pa.string()),
            pa.field("sparse", pa.string()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("score", pa.float64()),
        ]
    )
    content = {
        "sample_id": "sample/one",
        "position": position,
        "modality": modality,
        "content_type": "image/tiff" if modality == "image" else "text/plain",
        "binary_content": binary,
        "text_content": "original text",
        "caption": "caption",
        "sparse": None,
        "tags": ["one", None],
        "score": float("nan"),
    }
    metadata = {"sample_id": "sample/one", "position": -1, "modality": "metadata"}
    writer = InterleavedWebdatasetWriterStage(path=str(tmp_path), materialize_on_write=False)
    outputs = []
    for name, rows in [("single", [content]), ("with_metadata", [metadata, content])]:
        task = InterleavedBatch(dataset_name="test", data=pa.Table.from_pylist(rows, schema=schema))
        path = tmp_path / f"{name}.tar"
        writer.write_data(task, str(path))
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]
