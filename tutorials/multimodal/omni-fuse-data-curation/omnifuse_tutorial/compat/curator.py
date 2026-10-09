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

"""Small compatibility layer around NeMo Curator APIs."""

from __future__ import annotations

import copy
import inspect
from typing import Any

import pandas as pd
from nemo_curator.pipeline import Pipeline as CuratorPipeline
from nemo_curator.tasks import DocumentBatch, EmptyTask


def records_from_task(task: Any) -> list[dict[str, Any]]:
    """Return task data as records from a Curator task."""

    data = task.data
    if hasattr(data, "to_dict"):
        try:
            return [dict(row) for row in data.to_dict(orient="records")]
        except TypeError:
            return [dict(row) for row in data.to_dict("records")]
    if isinstance(data, list):
        return [dict(row) for row in data]
    raise TypeError(f"Unsupported task data type: {type(data)!r}")


def make_document_batch(
    task_id: str,
    dataset_name: str,
    records: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    stage_perf: list[Any] | None = None,
) -> DocumentBatch:
    """Construct a NeMo Curator DocumentBatch."""

    kwargs = {
        "dataset_name": dataset_name,
        "data": pd.DataFrame.from_records(records),
        "_metadata": metadata or {},
        "_stage_perf": stage_perf or [],
    }
    if "task_id" in inspect.signature(DocumentBatch).parameters:
        kwargs["task_id"] = task_id

    return DocumentBatch(**kwargs)


def make_empty_task() -> EmptyTask:
    if callable(EmptyTask):
        try:
            return EmptyTask(task_id="empty", dataset_name="omnifuse", data=None)
        except TypeError:
            return EmptyTask()
    return copy.deepcopy(EmptyTask)


def make_curator_pipeline(name: str, stages: list[Any], description: str | None = None) -> CuratorPipeline:
    return CuratorPipeline(name=name, description=description, stages=stages)
