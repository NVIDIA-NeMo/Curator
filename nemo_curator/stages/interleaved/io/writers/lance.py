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

import contextlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from loguru import logger

from nemo_curator.stages.interleaved.utils.schema import (
    deserialize_schema,
    reconcile_schema,
    serialize_schema,
)
from nemo_curator.tasks import FileGroupTask, InterleavedBatch

from .base import BaseInterleavedWriter

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class InterleavedLanceFragmentWriterStage(BaseInterleavedWriter):
    """Write interleaved rows as Lance fragments to a shared dataset path.

    Each task writes fragment data files via ``lance.fragment.write_fragments()``.
    After the pipeline finishes, call :func:`commit_lance_fragments` to assemble
    all fragments into a single LanceDB dataset.

    When *schema* is set (inherited from base), every fragment is aligned
    to it (missing columns become null arrays, extra columns dropped).  This
    prevents the lance parallel scanner crash caused by heterogeneous schemas.

    The ``binary_content`` column automatically uses Lance blob encoding
    (via ``lance-encoding:blob`` field metadata propagated from
    ``INTERLEAVED_SCHEMA``), which stores large binary data as separate blob
    files rather than columnar pages.
    """

    file_extension: str = "lance"
    name: str = "interleaved_lance_fragment_writer"

    def _write_dataframe(self, df: pd.DataFrame, file_path: str, write_kwargs: dict[str, Any]) -> None:
        pass

    def write_data(self, task: InterleavedBatch, file_path: str) -> None:  # noqa: ARG002
        import lance.fragment

        with self._time_metric("materialize_dataframe_total_s"):
            df = self._materialize_dataframe(task)
        df = self._align_output(df)

        table = pa.Table.from_pandas(df, preserve_index=False)
        write_schema = reconcile_schema(table.schema)
        table = table.cast(write_schema)

        with self._time_metric("lance_write_s"):
            fragments = lance.fragment.write_fragments(table, self.path, schema=write_schema)

        task._metadata["lance_fragments"] = [json.dumps(f.to_json()) for f in fragments]
        task._metadata["lance_schema"] = serialize_schema(write_schema)

    def process(self, task: InterleavedBatch) -> FileGroupTask:
        self.write_data(task, self.path)
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[self.path],
            _metadata={**task._metadata, "format": self.file_extension},
            _stage_perf=task._stage_perf,
        )


def commit_lance_fragments(
    path: str,
    output_tasks: list[FileGroupTask],
    mode: str = "overwrite",
) -> None:
    """Commit lance fragments written by InterleavedLanceFragmentWriterStage into a single dataset.

    Args:
        path: Lance dataset URI (same path passed to the writer stage).
        output_tasks: Output tasks from ``pipeline.run()``.
        mode: ``"overwrite"`` creates a new dataset; ``"append"`` adds to existing.
    """
    import lance

    all_fragment_jsons: list[str] = []
    schemas: list[pa.Schema] = []
    for task in output_tasks:
        all_fragment_jsons.extend(task._metadata.get("lance_fragments", []))
        if "lance_schema" in task._metadata:
            schemas.append(deserialize_schema(task._metadata["lance_schema"]))

    if not all_fragment_jsons:
        logger.warning("No lance fragments found in output tasks; nothing to commit")
        return

    if not schemas:
        msg = "No lance_schema found in any output task metadata"
        raise ValueError(msg)

    schema = pa.unify_schemas(schemas, promote_options="permissive")

    fragments = [lance.FragmentMetadata.from_json(fj) for fj in all_fragment_jsons]

    existing = None
    with contextlib.suppress(FileNotFoundError, ValueError, OSError):
        existing = lance.dataset(path)

    if mode not in ("overwrite", "append"):
        msg = f"commit_lance_fragments only supports mode='overwrite' or 'append', got {mode!r}"
        raise ValueError(msg)

    if existing is None or mode == "overwrite":
        op = lance.LanceOperation.Overwrite(schema, fragments)
        read_version = 0 if existing is None else existing.version
    else:
        op = lance.LanceOperation.Append(fragments)
        read_version = existing.version

    lance.LanceDataset.commit(path, op, read_version=read_version)
    ds = lance.dataset(path)
    logger.info("Committed {} fragments -> {} rows at {}", len(fragments), ds.count_rows(), path)
