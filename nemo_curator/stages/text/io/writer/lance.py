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

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.lance import (
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
