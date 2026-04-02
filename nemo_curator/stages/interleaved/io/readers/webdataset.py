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
from loguru import logger

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.stages.interleaved.utils import (
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_JSON_EXTENSIONS,
    resolve_storage_options,
    validate_and_project_source_fields,
)
from nemo_curator.stages.interleaved.utils.materialization import _extract_tiff_frame
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA, RESERVED_COLUMNS

from .base import BaseInterleavedReader


@dataclass
class _ReadContext:
    """Per-tar state shared across all members in a single tar archive."""

    tar_path: str
    member_names: set[str]
    member_info: dict[str, tarfile.TarInfo]
    storage_options: dict[str, object]
    byte_cache: dict[str, bytes | None]


@dataclass
class _SampleContext:
    """Per-sample state passed to row builder methods."""

    sample_id: str
    sample: dict[str, Any]
    tar_path: str
    json_member_name: str
    member_names: set[str]
    member_info: dict[str, tarfile.TarInfo] | None
    passthrough: dict[str, Any]
    per_image_passthrough: dict[str, list[Any]]
    per_text_passthrough: dict[str, list[Any]]


@dataclass
class InterleavedWebdatasetReaderStage(BaseInterleavedReader):
    """Read MINT1T-style WebDataset shards into a row-wise multimodal task."""

    materialize_on_read: bool = False
    max_batch_bytes: int | None = None
    json_extensions: tuple[str, ...] = DEFAULT_JSON_EXTENSIONS
    image_extensions: tuple[str, ...] = field(default_factory=lambda: DEFAULT_IMAGE_EXTENSIONS)
    sample_id_field: str | None = None
    texts_field: str = "texts"
    images_field: str = "images"
    image_member_field: str | None = None
    fields: tuple[str, ...] | None = None
    per_image_fields: tuple[str, ...] = ()
    per_text_fields: tuple[str, ...] = ()
    name: str = "webdataset_reader"

    def __post_init__(self) -> None:
        super().__post_init__()

    # -- source_ref construction --

    def _build_source_ref(
        self,
        ctx: _SampleContext,
        content_key: str | None,
        *,
        frame_index: int | None = None,
    ) -> str:
        if content_key is None:
            return InterleavedBatch.build_source_ref(path=None, member=None)
        byte_offset = None
        byte_size = None
        if ctx.member_info and content_key in ctx.member_info:
            info = ctx.member_info[content_key]
            byte_offset = info.offset_data
            byte_size = info.size
        return InterleavedBatch.build_source_ref(
            path=ctx.tar_path,
            member=content_key,
            byte_offset=byte_offset,
            byte_size=byte_size,
            frame_index=frame_index,
        )

    # -- row builders (override in subclasses for custom formats) --

    @staticmethod
    def _build_row(ctx: _SampleContext, row_fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": ctx.sample_id,
            "position": row_fields.get("position"),
            "modality": row_fields.get("modality"),
            "content_type": row_fields.get("content_type"),
            "text_content": row_fields.get("text_content"),
            "binary_content": row_fields.get("binary_content"),
            "source_ref": row_fields.get("source_ref"),
            "materialize_error": None,
        }

    def _metadata_row(self, ctx: _SampleContext) -> dict[str, Any]:
        return {
            **self._build_row(
                ctx,
                {
                    "position": -1,
                    "modality": "metadata",
                    "content_type": "application/json",
                    "source_ref": self._build_source_ref(ctx, ctx.json_member_name),
                },
            ),
            **ctx.passthrough,
        }

    @staticmethod
    def _apply_per_modality_fields(
        row: dict[str, Any],
        passthrough: dict[str, list[Any]],
        index: int,
    ) -> None:
        for field_name, values in passthrough.items():
            if index < len(values):
                val = values[index]
                row[field_name] = json.dumps(val, ensure_ascii=True) if isinstance(val, (dict, list)) else val

    @staticmethod
    def _warn_per_modality_length_mismatch(
        sample_id: str,
        passthrough: dict[str, list[Any]],
        actual_count: int,
        modality: str,
    ) -> None:
        for field_name, values in passthrough.items():
            if actual_count != len(values):
                logger.warning(
                    "sample_id={}: per_{}_field '{}' has {} values but {} non-None {}s",
                    sample_id,
                    modality,
                    field_name,
                    len(values),
                    actual_count,
                    modality,
                )

    def _text_rows(self, ctx: _SampleContext) -> list[dict[str, Any]]:
        texts = ctx.sample.get(self.texts_field)
        if not isinstance(texts, list):
            return []
        source_ref = self._build_source_ref(ctx, ctx.json_member_name)
        rows: list[dict[str, Any]] = []
        non_none_counter = 0
        for idx, text_value in enumerate(texts):
            if text_value is None:
                continue
            row = self._build_row(
                ctx,
                {
                    "position": idx,
                    "modality": "text",
                    "content_type": "text/plain",
                    "text_content": str(text_value),
                    "source_ref": source_ref,
                },
            )
            self._apply_per_modality_fields(row, ctx.per_text_passthrough, non_none_counter)
            non_none_counter += 1
            rows.append(row)
        self._warn_per_modality_length_mismatch(ctx.sample_id, ctx.per_text_passthrough, non_none_counter, "text")
        return rows

    def _image_rows(self, ctx: _SampleContext) -> list[dict[str, Any]]:
        images = ctx.sample.get(self.images_field)
        if not isinstance(images, list):
            return []
        image_member_name = self._resolve_default_image_member_name(
            ctx.sample_id,
            ctx.sample,
            images,
            ctx.member_names,
        )
        rows: list[dict[str, Any]] = []
        frame_counters: dict[str, int] = {}
        non_none_counter = 0
        for idx, image_token in enumerate(images):
            if image_token is None:
                continue
            content_key = self._resolve_image_content_key(image_token, image_member_name, ctx.member_names)
            content_type, _ = mimetypes.guess_type(content_key or image_member_name or "")
            frame_index = None
            is_multiframe_candidate = content_type == "image/tiff"
            if content_key is not None and is_multiframe_candidate:
                frame_index = frame_counters.get(content_key, 0)
                frame_counters[content_key] = frame_index + 1
            row = self._build_row(
                ctx,
                {
                    "position": idx,
                    "modality": "image",
                    "content_type": content_type or ("application/octet-stream" if image_member_name else None),
                    "source_ref": self._build_source_ref(ctx, content_key, frame_index=frame_index),
                },
            )
            self._apply_per_modality_fields(row, ctx.per_image_passthrough, non_none_counter)
            non_none_counter += 1
            rows.append(row)
        self._warn_per_modality_length_mismatch(ctx.sample_id, ctx.per_image_passthrough, non_none_counter, "image")
        return rows

    # -- sample-level orchestration --

    def _rows_from_sample(self, ctx: _SampleContext) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        rows.append(self._metadata_row(ctx))
        content_rows = self._text_rows(ctx) + self._image_rows(ctx)
        content_rows.sort(key=lambda r: r["position"])
        rows.extend(content_rows)
        per_modality_keys = set(ctx.per_image_passthrough) | set(ctx.per_text_passthrough)
        if per_modality_keys:
            for row in rows:
                for key in per_modality_keys:
                    row.setdefault(key, None)
        return rows

    # -- passthrough / schema helpers --

    def _build_passthrough_row(self, sample: dict[str, Any]) -> dict[str, Any]:
        excluded = RESERVED_COLUMNS | {
            *([self.sample_id_field] if self.sample_id_field else []),
            self.texts_field,
            self.images_field,
            *([self.image_member_field] if self.image_member_field else []),
            *self.per_image_fields,
            *self.per_text_fields,
        }
        return validate_and_project_source_fields(sample=sample, fields=self.fields, excluded_fields=excluded)

    @staticmethod
    def _extract_per_modality_fields(
        sample: dict[str, Any],
        field_names: tuple[str, ...],
    ) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = {}
        for field_name in field_names:
            if field_name not in sample:
                logger.warning("per-modality field '{}' not found in source sample", field_name)
                continue
            value = sample[field_name]
            if isinstance(value, list):
                result[field_name] = value
            else:
                msg = f"per-modality field '{field_name}' must be a list, got {type(value).__name__}"
                raise TypeError(msg)
        return result

    def _empty_output_schema(self) -> pa.Schema:
        # Use explicit schema if set; otherwise fall back to INTERLEAVED_SCHEMA as base
        base = self.schema if self.schema is not None else INTERLEAVED_SCHEMA
        seen = set(self.fields or ())
        all_extra = list(self.fields or ())
        for f in (*self.per_image_fields, *self.per_text_fields):
            if f not in seen:
                all_extra.append(f)
                seen.add(f)
        if not all_extra:
            return base
        existing = set(base.names)
        extra_fields = []
        for name in all_extra:
            if name not in existing:
                extra_fields.append(pa.field(name, pa.null()))
        return pa.schema([*base, *extra_fields]) if extra_fields else base

    # _align_output is inherited from BaseInterleavedReader

    # -- image member resolution --

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

    # -- tar member extraction --

    @staticmethod
    def _extract_tar_member(tf: tarfile.TarFile, member_name: str, cache: dict[str, bytes | None]) -> bytes | None:
        if member_name in cache:
            return cache[member_name]
        try:
            extracted = tf.extractfile(member_name)
        except KeyError:
            extracted = None
        payload = extracted.read() if extracted is not None else None
        cache[member_name] = payload
        return payload

    # -- per-member processing --

    def _rows_from_member(
        self,
        tf: tarfile.TarFile,
        member: tarfile.TarInfo,
        read_ctx: _ReadContext,
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
        ctx = _SampleContext(
            sample_id=sample_id,
            sample=payload,
            tar_path=read_ctx.tar_path,
            json_member_name=member.name,
            member_names=read_ctx.member_names,
            member_info=read_ctx.member_info,
            passthrough=self._build_passthrough_row(payload),
            per_image_passthrough=self._extract_per_modality_fields(payload, self.per_image_fields),
            per_text_passthrough=self._extract_per_modality_fields(payload, self.per_text_fields),
        )
        sample_rows = self._rows_from_sample(ctx)
        if self.materialize_on_read:
            for row in sample_rows:
                if row["modality"] != "image" or row["position"] < 0:
                    continue
                parsed_ref = InterleavedBatch.parse_source_ref(row["source_ref"])
                content_key = parsed_ref.get("member")
                if not content_key:
                    continue
                raw_bytes = self._extract_tar_member(tf, content_key, read_ctx.byte_cache)
                if raw_bytes is None:
                    row["materialize_error"] = f"missing member '{content_key}'"
                else:
                    frame_index = parsed_ref.get("frame_index")
                    if frame_index is not None:
                        tiff_frame = _extract_tiff_frame(raw_bytes, frame_index)
                        if tiff_frame is None:
                            row["materialize_error"] = f"failed to extract frame {frame_index} from '{content_key}'"
                        else:
                            raw_bytes = tiff_frame
                row["binary_content"] = raw_bytes
            read_ctx.byte_cache.clear()
        return sample_rows

    # -- source file helpers --

    @staticmethod
    def _source_files_for_split(
        split: pa.Table,
        idx: int,
        sample_id_to_tar: dict[str, str],
        all_tars: list[str],
    ) -> list[str]:
        """Return source_files for one split, listing only the contributing tars."""
        seen: set[str] = set()
        for sid in split["sample_id"].unique().to_pylist():
            tar = sample_id_to_tar.get(sid)
            if tar is not None:
                seen.add(tar)
        # Preserve original task.data order; fall back to all tars if none mapped.
        split_tars = [p for p in all_tars if p in seen] or all_tars
        return [f"{p}::split_{idx:05d}" for p in split_tars]

    # -- main entry point --

    def process(self, task: FileGroupTask) -> InterleavedBatch | list[InterleavedBatch]:
        rows: list[dict[str, Any]] = []
        sample_id_to_tar: dict[str, str] = {}
        storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)

        for tar_path in task.data:
            with (
                fsspec.open(tar_path, mode="rb", **storage_options) as fobj,
                tarfile.open(fileobj=fobj, mode="r:*") as tf,
            ):
                members = [m for m in tf.getmembers() if m.isfile()]
                member_names = {m.name for m in members}
                read_ctx = _ReadContext(
                    tar_path=tar_path,
                    member_names=member_names,
                    member_info={m.name: m for m in members},
                    storage_options=storage_options,
                    byte_cache={},
                )
                for member in members:
                    if not member.name.endswith(self.json_extensions):
                        continue
                    member_rows = self._rows_from_member(tf=tf, member=member, read_ctx=read_ctx)
                    if member_rows:
                        sample_id_to_tar.setdefault(member_rows[0]["sample_id"], tar_path)
                    rows.extend(member_rows)

        if rows:
            table = pa.Table.from_pylist(rows)
            table = self._align_output(table)
        else:
            # Empty tables use _empty_output_schema(); passthrough columns get
            # pa.null() type which is intentional (no data to infer from).
            table = pa.Table.from_pylist([], schema=self._empty_output_schema())
        splits = split_table_by_group_max_bytes(table, "sample_id", self.max_batch_bytes)
        batches: list[InterleavedBatch] = []
        for idx, split in enumerate(splits):
            task_id = f"{task.task_id}_processed" if len(splits) == 1 else f"{task.task_id}_processed_{idx:05d}"
            metadata = dict(task._metadata)
            if len(splits) == 1:
                metadata["source_files"] = list(task.data)
            else:
                metadata["source_files"] = self._source_files_for_split(split, idx, sample_id_to_tar, task.data)
            if storage_options:
                metadata["source_storage_options"] = storage_options
            batches.append(
                InterleavedBatch(
                    task_id=task_id,
                    dataset_name=task.dataset_name,
                    data=split,
                    _metadata=metadata,
                    _stage_perf=task._stage_perf,
                )
            )
        return batches if len(batches) > 1 else batches[0]
