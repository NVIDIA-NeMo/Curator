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

import lance
import pyarrow as pa

from nemo_curator.stages.interleaved.io.writers.lance import (
    InterleavedLanceFragmentWriterStage,
    commit_lance_fragments,
)
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA


def _make_batch(num_samples: int = 2, task_id: str = "test_batch") -> InterleavedBatch:
    rows = []
    for i in range(num_samples):
        sid = f"sample_{i}"
        rows.append({"sample_id": sid, "position": -1, "modality": "metadata", "content_type": "application/json",
                      "text_content": None, "binary_content": None, "source_ref": None, "materialize_error": None})
        rows.append({"sample_id": sid, "position": 0, "modality": "text", "content_type": "text/plain",
                      "text_content": f"Hello {i}", "binary_content": None,
                      "source_ref": None, "materialize_error": None})
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    return InterleavedBatch(task_id=task_id, dataset_name="test", data=table,
                            _metadata={"source_files": ["test.tar"]})


def test_fragment_writer_produces_metadata(tmp_path: Path) -> None:
    lance_path = str(tmp_path / "dataset.lance")
    writer = InterleavedLanceFragmentWriterStage(path=lance_path, materialize_on_write=False, mode="overwrite")
    result = writer.process(_make_batch(num_samples=2))
    assert isinstance(result, FileGroupTask)
    assert "lance_fragments" in result._metadata
    assert len(result._metadata["lance_fragments"]) > 0


def test_commit_creates_dataset(tmp_path: Path) -> None:
    lance_path = str(tmp_path / "dataset.lance")
    writer = InterleavedLanceFragmentWriterStage(path=lance_path, materialize_on_write=False, mode="overwrite")
    tasks = []
    for i in range(3):
        tasks.append(writer.process(_make_batch(num_samples=2, task_id=f"batch_{i}")))
    commit_lance_fragments(lance_path, tasks, mode="overwrite")
    ds = lance.dataset(lance_path)
    assert ds.count_rows() == 12


def test_commit_with_no_fragments_warns(tmp_path: Path) -> None:
    lance_path = str(tmp_path / "empty.lance")
    empty_task = FileGroupTask(task_id="t", dataset_name="test", data=[], _metadata={})
    commit_lance_fragments(lance_path, [empty_task], mode="overwrite")
