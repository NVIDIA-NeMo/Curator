from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.tasks import Task


@dataclass
class WorkflowRunResult:
    """Container returned by high-level workflows to expose pipeline outputs.

    Attributes:
        workflow_name: Human readable workflow identifier (e.g., ``fuzzy_dedup``).
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

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation to preserve backward compatibility."""
        return {
            "workflow_name": self.workflow_name,
            "pipeline_tasks": self.pipeline_tasks,
            "metadata": self.metadata,
        }

    def __getitem__(self, key: str) -> object:
        """Allow dict-style access for transitionary callers."""
        if key == "workflow_name":
            return self.workflow_name
        if key == "pipeline_tasks":
            return self.pipeline_tasks
        if key == "metadata":
            return self.metadata
        raise KeyError(key)
