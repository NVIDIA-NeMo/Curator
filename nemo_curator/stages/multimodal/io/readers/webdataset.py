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

import json
import mimetypes
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import pyarrow as pa

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.tasks import FileGroupTask, MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA

from .base import BaseMultimodalReader

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".gif")


@dataclass
class WebdatasetReaderStage(BaseMultimodalReader):
    """Read MINT1T-style WebDataset shards into a row-wise multimodal task."""

    load_binary: bool = False
    max_batch_bytes: int | None = None
    name: str = "webdataset_reader"

    def _rows_from_sample(
        self,
        sample_id: str,
        sample: dict[str, Any],
        source_shard: str,
        tar_path: str,
        json_member_name: str,
        image_member_name: str | None,
    ) -> list[dict[str, Any]]:
        source_id = sample.get("pdf_name")
        rows: list[dict[str, Any]] = []

        rows.append(
            {
                "sample_id": sample_id,
                "position": -1,
                "modality": "metadata",
                "content_type": "application/json",
                "text_content": None,
                "binary_content": None,
                "metadata_source": MultiBatchTask.build_metadata_source(
                    source_id=source_id,
                    source_shard=source_shard,
                    content_path=tar_path,
                    content_key=json_member_name,
                ),
                "metadata_json": json.dumps(sample, ensure_ascii=True),
                "materialize_error": None,
            }
        )

        texts = sample.get("texts")
        if isinstance(texts, list):
            for idx, text_value in enumerate(texts):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "position": idx,
                        "modality": "text",
                        "content_type": "text/plain",
                        "text_content": text_value if isinstance(text_value, str) else None,
                        "binary_content": None,
                        "metadata_source": MultiBatchTask.build_metadata_source(
                            source_id=source_id,
                            source_shard=source_shard,
                            content_path=tar_path,
                            content_key=json_member_name,
                        ),
                        "metadata_json": None,
                        "materialize_error": None,
                    }
                )

        images = sample.get("images")
        if isinstance(images, list):
            for idx, image_token in enumerate(images):
                content_key = image_member_name if image_token is not None else None
                content_type, _ = mimetypes.guess_type(image_member_name or "")
                rows.append(
                    {
                        "sample_id": sample_id,
                        "position": idx,
                        "modality": "image",
                        "content_type": content_type or ("application/octet-stream" if image_member_name else None),
                        "text_content": None,
                        "binary_content": None,
                        "metadata_source": MultiBatchTask.build_metadata_source(
                            source_id=source_id,
                            source_shard=source_shard,
                            content_path=tar_path,
                            content_key=content_key,
                        ),
                        "metadata_json": None,
                        "materialize_error": None,
                    }
                )

        return rows

    def process(self, task: FileGroupTask) -> MultiBatchTask | list[MultiBatchTask]:
        rows: list[dict[str, Any]] = []
        storage_options = (self.read_kwargs or {}).get("storage_options", {})

        for tar_path in task.data:
            source_shard = Path(tar_path).name
            with fsspec.open(tar_path, mode="rb", **storage_options) as fobj:
                with tarfile.open(fileobj=fobj, mode="r:*") as tf:
                    members = [m for m in tf.getmembers() if m.isfile()]
                    member_names = {m.name for m in members}
                    for member in members:
                        if not member.name.endswith(".json"):
                            continue
                        extracted = tf.extractfile(member)
                        if extracted is None:
                            continue
                        payload = json.load(extracted)
                        sample_id = Path(member.name).stem
                        image_member_name = next(
                            (f"{sample_id}{ext}" for ext in _IMAGE_EXTENSIONS if f"{sample_id}{ext}" in member_names),
                            None,
                        )
                        sample_rows = self._rows_from_sample(
                            sample_id=sample_id,
                            sample=payload,
                            source_shard=source_shard,
                            tar_path=tar_path,
                            json_member_name=member.name,
                            image_member_name=image_member_name,
                        )
                        if self.load_binary and image_member_name is not None:
                            img = tf.extractfile(image_member_name)
                            if img is not None:
                                image_bytes = img.read()
                                for row in sample_rows:
                                    if row["modality"] == "image" and row["position"] >= 0:
                                        row["binary_content"] = image_bytes
                        rows.extend(sample_rows)

        table = pa.Table.from_pylist(rows, schema=MULTIMODAL_SCHEMA)
        splits = split_table_by_group_max_bytes(table, "sample_id", self.max_batch_bytes)
        batches: list[MultiBatchTask] = []
        for idx, split in enumerate(splits):
            task_id = f"{task.task_id}_processed" if len(splits) == 1 else f"{task.task_id}_processed_{idx:05d}"
            batches.append(
                MultiBatchTask(
                    task_id=task_id,
                    dataset_name=task.dataset_name,
                    data=split,
                    _metadata=task._metadata,
                    _stage_perf=task._stage_perf,
                )
            )
        return batches if len(batches) > 1 else batches[0]
