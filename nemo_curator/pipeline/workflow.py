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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from nemo_curator.tasks import Task


@dataclass
class WorkflowRunResult:
    """Container returned by high-level workflows to expose pipeline outputs.

    Attributes:
        workflow_name: Human readable workflow identifier (e.g., "fuzzy_dedup").
        pipeline_tasks: Mapping of pipeline names to the ``Task`` objects they produced.
        metadata: Free-form dictionary for workflow specific timing or counters.
    """

    workflow_name: str
    pipeline_tasks: dict[str, list[Task]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_pipeline_tasks(self, pipeline_name: str, tasks: list[Task] | None) -> None:
        """Record the tasks emitted by a pipeline run (empty list if None)."""
        self.pipeline_tasks[pipeline_name] = list(tasks or [])

    def extend_metadata(self, updates: dict[str, Any] | None = None) -> None:
        """Update metadata dictionary in-place."""
        if updates:
            self.metadata.update(updates)

    def add_metadata(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Add a metadata key-value pair."""
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Any:  # noqa: ANN401
        """Get a metadata value."""
        return self.metadata.get(key)


class WorkflowBase(ABC):
    @abstractmethod
    def run(self, *args, **kwargs) -> WorkflowRunResult: ...
