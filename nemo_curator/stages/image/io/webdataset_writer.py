# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import hashlib
import io
import json
import os
import tarfile
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks.file_group import FileGroupTask
from nemo_curator.tasks.webdataset import WebDatasetBatch

# Well-known image extensions that should be encoded via PIL
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({"jpg", "jpeg", "png", "bmp", "tiff", "tif"})


@dataclass
class WebDatasetWriterStage(ProcessingStage[WebDatasetBatch, FileGroupTask]):
    """Write WebDataset samples to tar shards with sidecar Parquet metadata.

    Each tar shard contains up to ``samples_per_shard`` samples.  Every
    component of a :class:`WebDatasetSample` is written as a separate tar
    member named ``{key}.{ext}``.

    Image components (``jpg``, ``jpeg``, ``png``, …) are encoded via PIL.
    ``json`` components are serialized with :func:`json.dumps`.
    ``txt`` components are stored as UTF-8 text.
    All other components are stored as raw bytes.

    Metadata from :attr:`WebDatasetSample.metadata` is written to a
    sidecar Parquet file sharing the same base name as the tar.

    Args:
        output_dir: Directory to write tar/parquet files into.
        samples_per_shard: Maximum number of samples per tar shard.
        deterministic_name: Use a deterministic hash for shard naming.
        name: Stage name for the pipeline registry.
    """

    output_dir: str = ""
    samples_per_shard: int = 1000
    deterministic_name: bool = True
    verbose: bool = False
    name: str = "webdataset_writer"

    def __post_init__(self) -> None:
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    # -- naming helpers -----------------------------------------------------------

    def _construct_base_name(self, task: WebDatasetBatch) -> str:
        """Produce a shard base name (without extension)."""
        if self.deterministic_name:
            keys = [s.key for s in task.data]
            combined = "|".join(sorted(keys)) + "|" + task.task_id
            hash_hex = hashlib.sha256(combined.encode()).hexdigest()[:12]
            return f"shard-{hash_hex}"
        return f"shard-{uuid.uuid4().hex[:16]}"

    # -- component encoding -------------------------------------------------------

    @staticmethod
    def _encode_component(ext: str, data: Any) -> bytes:
        """Encode a single component value to bytes for tar storage."""
        if ext in _IMAGE_EXTENSIONS:
            return WebDatasetWriterStage._encode_image(data)
        if ext == "json":
            return json.dumps(data, ensure_ascii=False).encode("utf-8")
        if ext == "txt":
            return data.encode("utf-8") if isinstance(data, str) else bytes(data)
        if isinstance(data, bytes):
            return data
        if isinstance(data, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, data)
            return buf.getvalue()
        return str(data).encode("utf-8")

    @staticmethod
    def _encode_image(image: np.ndarray) -> bytes:
        """Encode a numpy image array to JPEG bytes."""
        from PIL import Image  # type: ignore[import-not-found]

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        ndim_grayscale = 2
        channels_rgb = 3
        channels_rgba = 4

        if img.ndim == ndim_grayscale:
            mode = "L"
        elif img.shape[2] == channels_rgb:
            mode = "RGB"
        elif img.shape[2] == channels_rgba:
            mode = "RGBA"
        else:
            mode = "RGB"
            img = img[..., :channels_rgb]

        with io.BytesIO() as buf:
            Image.fromarray(img, mode=mode).save(buf, format="JPEG", quality=92)
            return buf.getvalue()

    # -- tar / parquet I/O --------------------------------------------------------

    def _write_tar(self, base_name: str, members: list[tuple[str, bytes]]) -> str:
        """Write a tar file with given ``(member_name, bytes)`` entries.

        Returns the tar path.
        """
        tar_filename = f"{base_name}.tar"
        tar_path = os.path.join(self.output_dir, tar_filename)

        if os.path.exists(tar_path):
            logger.warning(f"File {tar_path} already exists. Overwriting it.")

        with open(tar_path, "wb") as fobj, tarfile.open(fileobj=fobj, mode="w") as tf:
            for member_name, payload in members:
                info = tarfile.TarInfo(name=member_name)
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))

        logger.debug(f"Wrote tar: {tar_path} with {len(members)} members")
        return tar_path

    def _write_parquet(self, base_name: str, rows: list[dict[str, Any]]) -> str:
        """Write metadata rows to a Parquet file and return its path."""
        parquet_path = os.path.join(self.output_dir, f"{base_name}.parquet")

        if os.path.exists(parquet_path):
            logger.warning(f"File {parquet_path} already exists. Overwriting it.")

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, parquet_path)

        logger.debug(f"Wrote parquet: {parquet_path} with {len(rows)} rows")
        return parquet_path

    # -- ProcessingStage interface ------------------------------------------------

    def process(self, task: WebDatasetBatch) -> FileGroupTask:
        if task is None or not isinstance(task.data, list) or len(task.data) == 0:
            logger.warning("Empty WebDatasetBatch provided to WebDatasetWriterStage; writing empty metadata only")

        tar_paths: list[str] = []
        parquet_paths: list[str] = []
        samples = task.data

        for start in range(0, len(samples), self.samples_per_shard):
            chunk = samples[start : start + self.samples_per_shard]
            members: list[tuple[str, bytes]] = []

            for sample in chunk:
                for ext, data in sample.components.items():
                    member_name = f"{sample.key}.{ext}"
                    payload = self._encode_component(ext, data)
                    members.append((member_name, payload))

            if members:
                chunk_index = start // self.samples_per_shard
                base_prefix = self._construct_base_name(task)
                base_name = f"{base_prefix}-{chunk_index:06d}"
                tar_path = self._write_tar(base_name, members)
                tar_paths.append(tar_path)

                metadata_rows: list[dict[str, Any]] = []
                for sample in chunk:
                    row: dict[str, Any] = {
                        "key": sample.key,
                        "tar_file": tar_path,
                        "extensions": ",".join(sample.extensions),
                        "shard_path": sample.shard_path,
                    }
                    row.update(sample.metadata)
                    metadata_rows.append(row)

                parquet_path = self._write_parquet(base_name, metadata_rows)
                parquet_paths.append(parquet_path)

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[*tar_paths, *parquet_paths],
            _metadata={
                **task._metadata,
                "output_dir": self.output_dir,
                "samples_per_shard": self.samples_per_shard,
                "num_samples": len(task.data),
            },
            _stage_perf=task._stage_perf,
        )
