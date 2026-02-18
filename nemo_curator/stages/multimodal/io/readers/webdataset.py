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

import json
import mimetypes
import tarfile
from dataclasses import dataclass, field
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
    json_extensions: tuple[str, ...] = (".json",)
    image_extensions: tuple[str, ...] = field(default_factory=lambda: _IMAGE_EXTENSIONS)
    source_id_field: str | None = None
    sample_id_field: str | None = None
    texts_field: str = "texts"
    images_field: str = "images"
    image_member_field: str | None = None
    name: str = "webdataset_reader"

    def __post_init__(self) -> None:
        if not self.source_id_field:
            msg = "source_id_field must be provided explicitly (e.g., 'pdf_name')"
            raise ValueError(msg)

    def _rows_from_sample(
        self,
        sample_id: str,
        sample: dict[str, Any],
        source: dict[str, str],
        member_names: set[str],
    ) -> list[dict[str, Any]]:
        source_id = sample.get(self.source_id_field) if self.source_id_field else None
        rows: list[dict[str, Any]] = []
        images = sample.get(self.images_field)
        image_member_name = self._resolve_default_image_member_name(sample_id, sample, images, member_names)

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
                    source_shard=source["source_shard"],
                    content_path=source["tar_path"],
                    content_key=source["json_member_name"],
                ),
                "metadata_json": json.dumps(sample, ensure_ascii=True),
                "materialize_error": None,
            }
        )

        texts = sample.get(self.texts_field)
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
                            source_shard=source["source_shard"],
                            content_path=source["tar_path"],
                            content_key=source["json_member_name"],
                        ),
                        "metadata_json": None,
                        "materialize_error": None,
                    }
                )

        if isinstance(images, list):
            for idx, image_token in enumerate(images):
                content_key = self._resolve_image_content_key(image_token, image_member_name, member_names)
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
                            source_shard=source["source_shard"],
                            content_path=source["tar_path"],
                            content_key=content_key,
                        ),
                        "metadata_json": None,
                        "materialize_error": None,
                    }
                )

        return rows

    def _resolve_default_image_member_name(
        self,
        sample_id: str,
        sample: dict[str, Any],
        images: list[object] | None,
        member_names: set[str],
    ) -> str | None:
        if self.image_member_field:
            image_member_name = sample.get(self.image_member_field)
            if isinstance(image_member_name, str) and image_member_name in member_names:
                return image_member_name
        if isinstance(images, list):
            for image_token in images:
                if isinstance(image_token, str) and image_token in member_names:
                    return image_token
        return next(
            (f"{sample_id}{ext}" for ext in self.image_extensions if f"{sample_id}{ext}" in member_names), None
        )

    @staticmethod
    def _resolve_image_content_key(
        image_token: object,
        default_image_member_name: str | None,
        member_names: set[str],
    ) -> str | None:
        if image_token is None:
            return None
        if isinstance(image_token, str) and image_token in member_names:
            return image_token
        return default_image_member_name

    def _rows_from_member(
        self,
        tf: tarfile.TarFile,
        member: tarfile.TarInfo,
        member_names: set[str],
        source_info: dict[str, str],
        binary_cache: dict[str, bytes | None],
    ) -> list[dict[str, Any]]:
        extracted = tf.extractfile(member)
        if extracted is None:
            return []
        payload = json.load(extracted)
        sample_id = (
            str(payload.get(self.sample_id_field))
            if self.sample_id_field and payload.get(self.sample_id_field) is not None
            else Path(member.name).stem
        )
        source = {
            "source_shard": source_info["source_shard"],
            "tar_path": source_info["tar_path"],
            "json_member_name": member.name,
        }
        sample_rows = self._rows_from_sample(
            sample_id=sample_id,
            sample=payload,
            source=source,
            member_names=member_names,
        )
        if self.load_binary:
            for row in sample_rows:
                if row["modality"] != "image" or row["position"] < 0:
                    continue
                source_meta = MultiBatchTask.parse_metadata_source(row["metadata_source"])
                content_key = source_meta.get("content_key")
                if not content_key:
                    continue
                if content_key not in binary_cache:
                    img = tf.extractfile(content_key)
                    binary_cache[content_key] = img.read() if img is not None else None
                row["binary_content"] = binary_cache[content_key]
        return sample_rows

    def process(self, task: FileGroupTask) -> MultiBatchTask | list[MultiBatchTask]:
        rows: list[dict[str, Any]] = []
        storage_options = (self.read_kwargs or {}).get("storage_options", {})

        for tar_path in task.data:
            source_shard = Path(tar_path).name
            with (
                fsspec.open(tar_path, mode="rb", **storage_options) as fobj,
                tarfile.open(fileobj=fobj, mode="r:*") as tf,
            ):
                members = [m for m in tf.getmembers() if m.isfile()]
                member_names = {m.name for m in members}
                binary_cache: dict[str, bytes | None] = {}
                source = {"source_shard": source_shard, "tar_path": tar_path}
                for member in members:
                    if not member.name.endswith(self.json_extensions):
                        continue
                    rows.extend(
                        self._rows_from_member(
                            tf=tf,
                            member=member,
                            member_names=member_names,
                            source_info=source,
                            binary_cache=binary_cache,
                        )
                    )

        table = pa.Table.from_pylist(rows, schema=MULTIMODAL_SCHEMA)
        splits = split_table_by_group_max_bytes(table, "sample_id", self.max_batch_bytes)
        batches: list[MultiBatchTask] = []
        for idx, split in enumerate(splits):
            task_id = f"{task.task_id}_processed" if len(splits) == 1 else f"{task.task_id}_processed_{idx:05d}"
            metadata = dict(task._metadata)
            if storage_options:
                metadata["source_storage_options"] = storage_options
            batches.append(
                MultiBatchTask(
                    task_id=task_id,
                    dataset_name=task.dataset_name,
                    data=split,
                    _metadata=metadata,
                    _stage_perf=task._stage_perf,
                )
            )
        return batches if len(batches) > 1 else batches[0]
