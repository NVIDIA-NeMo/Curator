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

from __future__ import annotations

from dataclasses import dataclass, field

import fsspec
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, ImageBatch, ImageObject


@dataclass
class _SplitByTarStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Split image locator rows into small offset-sorted groups that share one TAR."""

    tar_column: str
    offset_column: str
    size_column: str
    images_per_batch: int
    max_request_bytes: int
    name: str = "split_by_tar"

    def process(self, task: DocumentBatch) -> list[DocumentBatch]:
        df = task.to_pandas().sort_values([self.tar_column, self.offset_column], kind="stable")
        tars, starts = df[self.tar_column].tolist(), df[self.offset_column].tolist()
        ends = (df[self.offset_column] + df[self.size_column]).tolist()
        bounds, first = [], 0
        for i in range(1, len(df) + 1):
            if (
                i == len(df)
                or tars[i] != tars[first]
                or i - first == self.images_per_batch
                or ends[i] - starts[first] > self.max_request_bytes
            ):
                bounds.append((first, i))
                first = i
        return [
            DocumentBatch(
                dataset_name=task.dataset_name,
                data=df.iloc[a:b],
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
            )
            for a, b in bounds
        ]


@dataclass
class _FetchTarRangeStage(ProcessingStage[DocumentBatch, ImageBatch]):
    """Read every payload of a single-TAR group with one ranged GET."""

    tar_column: str
    offset_column: str
    size_column: str
    id_column: str
    storage_options: dict
    name: str = "fetch_tar_range"
    # Network-bound: a fractional CPU lets many fetches overlap with CPU decoding.
    resources = Resources(cpus=0.25)

    def process(self, task: DocumentBatch) -> ImageBatch:
        rows = task.to_pandas().to_dict("records")
        uri = rows[0][self.tar_column]
        start = rows[0][self.offset_column]
        end = max(row[self.offset_column] + row[self.size_column] for row in rows)
        try:
            fs, path = fsspec.core.url_to_fs(uri, **self.storage_options)
            blob = fs.cat_file(path, start=start, end=end)
        except FileNotFoundError:
            logger.warning(f"TAR not found, keeping {len(rows)} rows without bytes: {uri}")
            blob = None
        if blob is not None and len(blob) != end - start:
            msg = f"Short read from {uri}: expected {end - start} bytes, got {len(blob)}"
            raise OSError(msg)
        images = []
        for row in rows:
            offset = row[self.offset_column] - start
            payload = None if blob is None else blob[offset : offset + row[self.size_column]]
            images.append(ImageObject(image_id=str(row[self.id_column]), metadata=row, image_bytes=payload))
        return ImageBatch(
            dataset_name=task.dataset_name,
            data=images,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class ImageTarRangeReader(CompositeStage[DocumentBatch, ImageBatch]):
    """Fetch encoded images located by ``(tar, byte offset, byte size)`` rows.

    Use this when a table (for example a Parquet catalog) records where each image
    lives inside WebDataset-style TARs on S3 or any other fsspec filesystem. Rows are
    sorted by TAR and offset and cut into groups of at most ``images_per_batch``
    images spanning at most ``max_request_bytes``; each group is read with a single
    ranged GET instead of one request per image. Encoded bytes go to
    ``ImageObject.image_bytes`` and every input column to ``ImageObject.metadata``.
    Object stores often cap per-connection throughput, so set ``fetch_workers`` to run
    many concurrent fetches (each reserves a quarter CPU); ``None`` lets Ray decide.

    Rows whose TAR does not exist are kept with ``image_bytes=None``. Short reads and
    other I/O errors are raised so the executor can retry the task.
    """

    tar_column: str = "tar_path"
    offset_column: str = "byte_offset"
    size_column: str = "byte_size"
    id_column: str = "image_id"
    images_per_batch: int = 128
    max_request_bytes: int = 256 * 1024**2
    storage_options: dict = field(default_factory=dict)
    fetch_workers: int | None = None
    name: str = "image_tar_range_reader"

    def __post_init__(self) -> None:
        super().__init__()
        if self.images_per_batch <= 0 or self.max_request_bytes <= 0:
            msg = "images_per_batch and max_request_bytes must be positive"
            raise ValueError(msg)

    def decompose(self) -> list[ProcessingStage]:
        columns = {"tar_column": self.tar_column, "offset_column": self.offset_column, "size_column": self.size_column}
        return [
            _SplitByTarStage(
                **columns, images_per_batch=self.images_per_batch, max_request_bytes=self.max_request_bytes
            ),
            _FetchTarRangeStage(**columns, id_column=self.id_column, storage_options=self.storage_options).with_(
                num_workers=self.fetch_workers
            ),
        ]
