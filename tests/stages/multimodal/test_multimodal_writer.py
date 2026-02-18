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
from pathlib import Path

import pandas as pd
import pyarrow as pa

from nemo_curator.stages.multimodal.io.readers.webdataset import WebdatasetReaderStage
from nemo_curator.stages.multimodal.io.writers.tabular import MultimodalParquetWriterStage
from nemo_curator.tasks import FileGroupTask, MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA


def _read_batch(input_task: FileGroupTask) -> MultiBatchTask:
    batch = WebdatasetReaderStage(source_id_field="pdf_name").process(input_task)
    assert isinstance(batch, MultiBatchTask)
    return batch


def _metadata_source(content_path: str, content_key: str | None, source_shard: str = "shard-00000.tar") -> str:
    return json.dumps(
        {
            "source_id": "doc.pdf",
            "source_shard": source_shard,
            "content_path": content_path,
            "content_key": content_key,
        }
    )


def test_writer_marks_materialize_error_on_bad_source_path(tmp_path: Path, input_task: FileGroupTask) -> None:
    batch = _read_batch(input_task)
    df = batch.to_pandas().copy()
    image_mask = df["modality"] == "image"
    assert image_mask.any()
    first_image_idx = df[image_mask].index[0]
    df.loc[first_image_idx, "metadata_source"] = _metadata_source("/definitely/missing/path.tar", "abc123.tiff")
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
                "metadata_source": _metadata_source(str(raw_path), None, source_shard="raw_image.jpg"),
                "metadata_json": None,
                "materialize_error": None,
            }
        ],
        schema=MULTIMODAL_SCHEMA,
    )
    task = MultiBatchTask(
        task_id="direct_content_path",
        dataset_name="mint_test",
        data=table,
        _metadata={"source_files": [str(raw_path)]},
    )

    writer = MultimodalParquetWriterStage(
        path=str(tmp_path / "out_direct"), materialize_on_write=True, mode="overwrite"
    )
    write_task = writer.process(task)
    written = pd.read_parquet(write_task.data[0])
    assert written.loc[0, "binary_content"] == image_bytes
    assert pd.isna(written.loc[0, "materialize_error"])
