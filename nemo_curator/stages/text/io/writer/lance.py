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

import base64
import pickle
from dataclasses import dataclass, field
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.lance import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
    lance_checkpoint_record_id,
    read_lance_checkpoint,
    write_lance_checkpoint_marker,
    write_lance_checkpoint_record,
)

_RESERVED_LANCE_PREFIX = "__lance_"


def _drop_reserved_lance_columns(table: pa.Table) -> pa.Table:
    columns = [name for name in table.column_names if not name.startswith(_RESERVED_LANCE_PREFIX)]
    return table.select(columns)


def _metadata_lance_schema(task: DocumentBatch) -> pa.Schema | None:
    schema = (task._metadata.get("lance") or {}).get("schema")
    if not isinstance(schema, dict):
        return None
    from lance.schema import json_to_schema

    return json_to_schema(schema)


def _schema_for_table(lance_schema: pa.Schema, table: pa.Table) -> pa.Schema:
    fields = []
    for table_field in table.schema:
        if table_field.name in lance_schema.names:
            fields.append(lance_schema.field(table_field.name))
        else:
            fields.append(table_field)
    return pa.schema(fields)


@dataclass
class LanceWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    """Write ``DocumentBatch`` tables to Lance fragments and checkpoint the commit."""

    path: str
    commit_path: str
    schema: pa.Schema | None = None
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    fields: list[str] | None = None
    enable_stable_row_ids: bool = False
    name: str = "lance_writer"
    mode: Literal["create", "append", "overwrite"] = "create"

    def __post_init__(self) -> None:
        self.write_kwargs = dict(self.write_kwargs or {})
        if "enable_stable_row_ids" in self.write_kwargs:
            msg = "Set enable_stable_row_ids on LanceWriter, not in write_kwargs"
            raise ValueError(msg)
        if self.enable_stable_row_ids and self.mode != "create":
            msg = "enable_stable_row_ids requires mode='create' because Lance only enables it for new datasets"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _output_table_and_schema(self, task: DocumentBatch) -> tuple[pa.Table, pa.Schema | None]:
        table = task.to_pyarrow()
        schema = self.schema or _metadata_lance_schema(task)
        if self.schema is not None:
            table = table.select(self.schema.names)
            return table, self.schema
        table = table.select(self.fields) if self.fields is not None else _drop_reserved_lance_columns(table)
        return table, _schema_for_table(schema, table) if schema is not None else None

    def process(self, task: DocumentBatch) -> FileGroupTask:
        from lance.schema import schema_to_json
        from lance_ray.fragment import write_fragment

        write_kwargs = dict(self.write_kwargs)
        checkpoint_storage_options = write_kwargs.pop("checkpoint_storage_options", None)
        write_kwargs.setdefault("max_rows_per_file", 500_000)
        table, schema = self._output_table_and_schema(task)
        results = write_fragment(
            [table],
            self.path,
            schema=schema,
            **write_kwargs,
        )

        record_paths = []
        for index, (fragment, schema) in enumerate(results):
            record = {
                "kind": "lance_write",
                "dataset_path": self.path,
                "mode": self.mode,
                "enable_stable_row_ids": self.enable_stable_row_ids,
                "task_id": task.task_id,
                "fragment_index": index,
                "schema": schema_to_json(schema),
                "fragment": base64.b64encode(pickle.dumps(fragment)).decode("ascii"),
            }
            record_paths.append(
                write_lance_checkpoint_record(
                    self.commit_path,
                    record,
                    lance_checkpoint_record_id("lance_write", task.task_id, index),
                    checkpoint_storage_options,
                )
            )

        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=record_paths,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class LanceAnnotationWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    """Update existing Lance rows using metadata columns emitted by ``LanceReader``."""

    path: str
    commit_path: str
    schema: pa.Schema
    create_columns: bool = False
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    fields: list[str] | None = None
    name: str = "lance_annotation_writer"
    _prepared_version: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.write_kwargs = dict(self.write_kwargs or {})
        self.storage_options = self.write_kwargs.get("storage_options")
        self.checkpoint_storage_options = self.write_kwargs.get("checkpoint_storage_options")
        self.fields = list(self.schema.names if self.fields is None else self.fields)
        missing = [field for field in self.fields if field not in self.schema.names]
        if missing:
            msg = f"fields must be present in schema; missing {missing}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def prepare(self) -> int:
        """Create or validate annotation columns and pin the Lance version for the run."""
        import lance

        dataset = lance.dataset(self.path, storage_options=self.storage_options)
        target_schema = pa.schema([self.schema.field(field) for field in self.fields])
        missing_fields = [field for field in target_schema if field.name not in dataset.schema.names]
        if missing_fields:
            if not self.create_columns:
                missing = [field.name for field in missing_fields]
                msg = f"Lance annotation columns do not exist: {missing}"
                raise ValueError(msg)
            dataset.add_columns(pa.schema(missing_fields))
            dataset = lance.dataset(self.path, storage_options=self.storage_options)
        self._validate_target_columns(dataset)
        self._prepared_version = dataset.version
        return dataset.version

    def _validate_target_columns(self, dataset: object) -> None:
        missing = [field for field in self.fields if field not in dataset.schema.names]
        if missing:
            msg = f"Lance annotation columns do not exist: {missing}. Call prepare() before pipeline.run()."
            raise ValueError(msg)

    def _update_version(self) -> int:
        import lance

        if self._prepared_version is not None:
            return self._prepared_version
        logger.warning(
            "LanceAnnotationWriter is running without prepare(). "
            "This is only safe when target columns exist and the dataset does not change during the run."
        )
        dataset = lance.dataset(self.path, storage_options=self.storage_options)
        self._validate_target_columns(dataset)
        return dataset.version

    def _required_columns(self) -> list[str]:
        return [LANCE_ROWADDR_COLUMN, LANCE_FRAGID_COLUMN, *self.fields]

    def _validate_update_table(self, table: pa.Table) -> None:
        missing = [name for name in self._required_columns() if name not in table.column_names]
        if missing:
            msg = f"Lance annotation update table is missing required columns: {missing}"
            raise ValueError(msg)

    def _validate_unique_rowaddrs(self, table: pa.Table, fragment_id: int) -> None:
        counts = pc.value_counts(table["_rowaddr"].combine_chunks())
        duplicate_values = counts.field("values").filter(pc.greater(counts.field("counts"), 1))
        if len(duplicate_values) > 0:
            sample = ", ".join(str(value) for value in duplicate_values.slice(0, 5).to_pylist())
            msg = (
                f"Lance annotation update contains duplicate {LANCE_ROWADDR_COLUMN} values "
                f"for fragment {fragment_id}: {sample}"
            )
            raise ValueError(msg)

    def _update_table_for_fragment(self, table: pa.Table, fragment_id: int) -> pa.Table:
        fragids = table[LANCE_FRAGID_COLUMN].combine_chunks().cast(pa.uint64())
        mask = pc.equal(fragids, pa.scalar(fragment_id, type=pa.uint64()))
        fragment_table = table.filter(mask)
        update_table = fragment_table.select([LANCE_ROWADDR_COLUMN, *self.fields]).rename_columns(
            ["_rowaddr", *self.fields]
        )
        update_table = update_table.set_column(
            0,
            "_rowaddr",
            update_table["_rowaddr"].combine_chunks().cast(pa.uint64()),
        )
        self._validate_unique_rowaddrs(update_table, fragment_id)
        return update_table

    def process(self, task: DocumentBatch) -> FileGroupTask:
        import lance

        table = task.to_pyarrow()
        self._validate_update_table(table)
        version = self._update_version()
        dataset_kwargs: dict[str, Any] = {"version": version}
        if self.storage_options is not None:
            dataset_kwargs["storage_options"] = self.storage_options
        dataset = lance.dataset(self.path, **dataset_kwargs)

        record_paths = []
        fragment_ids = sorted(
            int(value) for value in pc.unique(table[LANCE_FRAGID_COLUMN].combine_chunks()).to_pylist()
        )
        for fragment_id in fragment_ids:
            update_table = self._update_table_for_fragment(table, fragment_id)
            fragment = dataset.get_fragment(fragment_id)
            updated_fragment, fields_modified = fragment.update_columns(
                update_table,
                left_on="_rowaddr",
                right_on="_rowaddr",
            )
            if not fields_modified:
                msg = f"update_columns returned no fields_modified for fragment {fragment_id}"
                raise RuntimeError(msg)
            record = {
                "kind": "lance_annotation_update",
                "dataset_path": self.path,
                "dataset_version": version,
                "task_id": task.task_id,
                "fragment_id": fragment_id,
                "fields_modified": sorted(set(fields_modified)),
                "updated_fragment": base64.b64encode(pickle.dumps(updated_fragment)).decode("ascii"),
            }
            record_paths.append(
                write_lance_checkpoint_record(
                    self.commit_path,
                    record,
                    lance_checkpoint_record_id("lance_annotation_update", task.task_id, fragment_id),
                    self.checkpoint_storage_options,
                )
            )

        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=record_paths,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def _validate_checkpoint_path(records: list[dict[str, Any]], path: str) -> None:
    dataset_paths = {record["dataset_path"] for record in records}
    if dataset_paths != {path}:
        msg = f"Checkpoint records are for {sorted(dataset_paths)}, not {path}"
        raise ValueError(msg)


def _single_checkpoint_value(records: list[dict[str, Any]], key: str, label: str) -> object:
    values = {record[key] for record in records}
    if len(values) != 1:
        msg = f"Expected one {label}; got {sorted(values)}"
        raise ValueError(msg)
    return next(iter(values))


def _decode_write_fragments(records: list[dict[str, Any]]) -> list[tuple[object, pa.Schema]]:
    from lance.schema import json_to_schema

    return [
        (pickle.loads(base64.b64decode(record["fragment"])), json_to_schema(record["schema"]))  # noqa: S301
        for record in sorted(
            records, key=lambda record: (str(record.get("task_id", "")), record.get("fragment_index", 0))
        )
    ]


def _annotation_records_by_fragment(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    records_by_fragment = {int(record["fragment_id"]): record for record in records}
    if len(records_by_fragment) != len(records):
        msg = "Ensure each Lance fragment is updated by at most one writer task."
        raise ValueError(msg)
    return records_by_fragment


def _decode_updated_fragments(records: list[dict[str, Any]]) -> list[object]:
    return [
        pickle.loads(base64.b64decode(record["updated_fragment"]))  # noqa: S301
        for record in records
    ]


def _fields_modified(records: list[dict[str, Any]]) -> list[str]:
    return sorted({field for record in records for field in record["fields_modified"]})


def commit_lance_checkpoint(
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> int:
    """Commit records written by ``LanceWriter`` and return the Lance version."""
    import lance
    from lance_ray import LanceFragmentCommitter

    records, committed_version = read_lance_checkpoint(commit_path, "lance_write", checkpoint_storage_options)
    if committed_version is not None:
        return committed_version

    _validate_checkpoint_path(records, path)
    mode = str(_single_checkpoint_value(records, "mode", "write mode"))
    stable_row_id_settings = {bool(record.get("enable_stable_row_ids", False)) for record in records}
    if len(stable_row_id_settings) != 1:
        msg = f"Expected one stable row ID setting; got {sorted(stable_row_id_settings)}"
        raise ValueError(msg)
    enable_stable_row_ids = next(iter(stable_row_id_settings))
    fragments = _decode_write_fragments(records)
    schema = fragments[0][1]

    if enable_stable_row_ids:
        if mode != "create":
            msg = "Stable row IDs can only be enabled while creating a new Lance dataset"
            raise ValueError(msg)
        operation = lance.LanceOperation.Overwrite(schema, [fragment for fragment, _ in fragments])
        lance.LanceDataset.commit(
            path,
            operation,
            read_version=0,
            storage_options=storage_options,
            enable_stable_row_ids=True,
        )
    else:
        committer = LanceFragmentCommitter(path, schema=schema, mode=mode, storage_options=storage_options)
        if mode == "append":
            committer.on_write_start(schema)
        fragment_payloads = [(pickle.dumps(fragment), pickle.dumps(schema)) for fragment, schema in fragments]
        committer.on_write_complete([fragment_payloads])
    dataset = lance.dataset(path, storage_options=storage_options)
    if enable_stable_row_ids and not dataset.has_stable_row_ids:
        msg = "Lance commit completed without enabling stable row IDs"
        raise RuntimeError(msg)
    version = dataset.version
    write_lance_checkpoint_marker(commit_path, version, checkpoint_storage_options)
    return version


def commit_lance_annotation_checkpoint(
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> int:
    """Commit records written by ``LanceAnnotationWriter`` and return the Lance version."""
    import lance

    records, committed_version = read_lance_checkpoint(
        commit_path, "lance_annotation_update", checkpoint_storage_options
    )
    if committed_version is not None:
        return committed_version

    _validate_checkpoint_path(records, path)
    read_version = int(_single_checkpoint_value(records, "dataset_version", "dataset version"))
    records_by_fragment = _annotation_records_by_fragment(records)
    fragment_records = [records_by_fragment[fragment_id] for fragment_id in sorted(records_by_fragment)]
    updated_fragments = _decode_updated_fragments(fragment_records)
    fields_modified = _fields_modified(fragment_records)
    operation = lance.LanceOperation.Update(updated_fragments=updated_fragments, fields_modified=fields_modified)
    version = lance.LanceDataset.commit(
        path,
        operation,
        read_version=read_version,
        storage_options=storage_options,
    ).version
    write_lance_checkpoint_marker(commit_path, version, checkpoint_storage_options)
    return version
