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
from nemo_curator.stages.multimodal.utils import (
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_JSON_EXTENSIONS,
    load_bytes_from_metadata_source,
    require_source_id_field,
    resolve_storage_options,
)
from nemo_curator.tasks import FileGroupTask, MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA

from .base import BaseMultimodalReader


@dataclass
class _ReadContext:
    source_shard: str
    tar_path: str
    member_names: set[str]
    storage_options: dict[str, object]
    byte_cache: dict[tuple[str, str], bytes | None]


@dataclass
class WebdatasetReaderStage(BaseMultimodalReader):
    """Read MINT1T-style WebDataset shards into a row-wise multimodal task."""

    materialize_on_read: bool = False
    max_batch_bytes: int | None = None
    json_extensions: tuple[str, ...] = DEFAULT_JSON_EXTENSIONS
    image_extensions: tuple[str, ...] = field(default_factory=lambda: DEFAULT_IMAGE_EXTENSIONS)
    source_id_field: str = ""
    sample_id_field: str | None = None
    texts_field: str = "texts"
    images_field: str = "images"
    image_member_field: str | None = None
    fields: tuple[str, ...] | None = None
    name: str = "webdataset_reader"

    def __post_init__(self) -> None:
        self.source_id_field = require_source_id_field(self.source_id_field)

    def _rows_from_sample(
        self,
        sample_id: str,
        sample: dict[str, Any],
        source: dict[str, str],
        member_names: set[str],
    ) -> list[dict[str, Any]]:
        source_id = sample.get(self.source_id_field)
        rows: list[dict[str, Any]] = []
        images = sample.get(self.images_field)
        image_member_name = self._resolve_default_image_member_name(sample_id, sample, images, member_names)
        source_shard = source["source_shard"]
        tar_path = source["tar_path"]
        json_member_name = source["json_member_name"]
        passthrough_row = self._build_passthrough_row(sample)

        def build_metadata_source(content_key: str | None) -> str:
            return MultiBatchTask.build_metadata_source(
                source_id=source_id,
                source_shard=source_shard,
                content_path=tar_path,
                content_key=content_key,
            )

        def append_row(row: dict[str, Any]) -> None:
            rows.append(
                {
                    "sample_id": sample_id,
                    "position": row["position"],
                    "modality": row["modality"],
                    "content_type": row.get("content_type"),
                    "text_content": row.get("text_content"),
                    "binary_content": row.get("binary_content"),
                    "metadata_source": row.get("metadata_source"),
                    "metadata_json": row.get("metadata_json"),
                    "materialize_error": None,
                    **passthrough_row,
                }
            )

        append_row(
            {
                "position": -1,
                "modality": "metadata",
                "content_type": "application/json",
                "metadata_source": build_metadata_source(json_member_name),
                "metadata_json": json.dumps(sample, ensure_ascii=True),
            }
        )

        texts = sample.get(self.texts_field)
        if isinstance(texts, list):
            for idx, text_value in enumerate(texts):
                append_row(
                    {
                        "position": idx,
                        "modality": "text",
                        "content_type": "text/plain",
                        "text_content": text_value if isinstance(text_value, str) else None,
                        "metadata_source": build_metadata_source(json_member_name),
                    }
                )

        if isinstance(images, list):
            for idx, image_token in enumerate(images):
                content_key = self._resolve_image_content_key(image_token, image_member_name, member_names)
                content_type, _ = mimetypes.guess_type(image_member_name or "")
                append_row(
                    {
                        "position": idx,
                        "modality": "image",
                        "content_type": content_type or ("application/octet-stream" if image_member_name else None),
                        "metadata_source": build_metadata_source(content_key),
                    }
                )

        return rows

    def _build_passthrough_row(self, sample: dict[str, Any]) -> dict[str, Any]:
        excluded = {
            self.source_id_field,
            self.sample_id_field,
            self.texts_field,
            self.images_field,
            self.image_member_field,
            "sample_id",
            "position",
            "modality",
            "content_type",
            "text_content",
            "binary_content",
            "metadata_source",
            "metadata_json",
            "materialize_error",
        }
        if self.fields is None:
            fields = [key for key in sample if key not in excluded]
        else:
            fields = list(self.fields)
            reserved = sorted(field for field in fields if field in excluded)
            if reserved:
                msg = f"fields contains reserved keys: {reserved}"
                raise ValueError(msg)
            missing = sorted(field for field in fields if field not in sample)
            if missing:
                msg = f"fields not found in source sample: {missing}"
                raise ValueError(msg)
        return {
            field: (
                json.dumps(sample.get(field), ensure_ascii=True)
                if isinstance(sample.get(field), (dict, list))
                else sample.get(field)
            )
            for field in fields
        }

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
        context: _ReadContext,
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
            "source_shard": context.source_shard,
            "tar_path": context.tar_path,
            "json_member_name": member.name,
        }
        sample_rows = self._rows_from_sample(
            sample_id=sample_id,
            sample=payload,
            source=source,
            member_names=context.member_names,
        )
        if self.materialize_on_read:
            for row in sample_rows:
                if row["modality"] != "image" or row["position"] < 0:
                    continue
                row["binary_content"] = load_bytes_from_metadata_source(
                    source_value=row["metadata_source"],
                    storage_options=context.storage_options,
                    byte_cache=context.byte_cache,
                )
        return sample_rows

    def process(self, task: FileGroupTask) -> MultiBatchTask | list[MultiBatchTask]:
        rows: list[dict[str, Any]] = []
        storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)

        for tar_path in task.data:
            source_shard = Path(tar_path).name
            with (
                fsspec.open(tar_path, mode="rb", **storage_options) as fobj,
                tarfile.open(fileobj=fobj, mode="r:*") as tf,
            ):
                members = [m for m in tf.getmembers() if m.isfile()]
                member_names = {m.name for m in members}
                context = _ReadContext(
                    source_shard=source_shard,
                    tar_path=tar_path,
                    member_names=member_names,
                    storage_options=storage_options,
                    byte_cache={},
                )
                for member in members:
                    if not member.name.endswith(self.json_extensions):
                        continue
                    rows.extend(
                        self._rows_from_member(
                            tf=tf,
                            member=member,
                            context=context,
                        )
                    )

        table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA)
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
