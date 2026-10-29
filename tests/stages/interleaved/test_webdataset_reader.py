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

import pytest

from nemo_curator.stages.interleaved.io.readers.webdataset import InterleavedWebdatasetReaderStage
from nemo_curator.stages.interleaved.utils import materialize_task_binary_content
from nemo_curator.tasks import FileGroupTask, InterleavedBatch


@pytest.mark.parametrize("mode", ["w:", "w:gz", "w:bz2", "w:xz"])
def test_source_refs_materialize_stored_tar_bytes(tmp_path: Path, mode: str) -> None:
    # The suffix deliberately does not identify compression; tarfile detects the stream.
    path = tmp_path / "images.tar"
    payload = b"original-image-payload" * 100
    member_name = "long-name-" + "x" * 110 + ".jpg"
    metadata = json.dumps({"texts": [], "images": [member_name]}).encode()
    with tarfile.open(path, mode) as archive:
        for name, content in [(member_name, payload), ("sample.json", metadata)]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, BytesIO(content))
    batch = InterleavedWebdatasetReaderStage().process(FileGroupTask(dataset_name="test", data=[str(path)]))
    assert isinstance(batch, InterleavedBatch)
    for row in batch.to_pyarrow().to_pylist():
        ref = InterleavedBatch.parse_source_ref(row["source_ref"])
        if mode == "w:":
            with path.open("rb") as stream:
                stream.seek(ref["byte_offset"])
                expected = payload if row["modality"] == "image" else metadata
                assert stream.read(ref["byte_size"]) == expected
        else:
            assert ref["byte_offset"] is None
            assert ref["byte_size"] is None
    materialized = materialize_task_binary_content(batch).to_pyarrow().to_pylist()
    images = [row for row in materialized if row["modality"] == "image"]
    assert len(images) == 1
    assert images[0]["materialize_error"] is None
    assert images[0]["binary_content"] == payload
