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

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from nemo_curator.utils.performance_utils import StagePerfStats

T = TypeVar("T")


@dataclass
class Task(ABC, Generic[T]):
    """Abstract base class for tasks in the pipeline.

    A task represents a batch of data to be processed. Different modalities
    (text, audio, video) can implement their own task types.

    Attributes:
        task_id: Deterministic identifier for this task. User-provided
            values are ALWAYS overwritten by the framework via
            ``_set_lineage`` once the task flows through any stage. Two
            runs of the same pipeline on the same inputs produce
            byte-identical ``task_id``s across all tasks.
        dataset_name: Name of the dataset this task belongs to.
        _stage_perf: List of stages perfs this task has passed through.
        _lineage_path: Underscore-joined path through the pipeline DAG
            (e.g. ``"abc123_0_5"`` = source ``abc123``, then child 0,
            then grandchild 5). Hashed into ``task_id``. Empty until
            ``_set_lineage`` runs.
    """

    task_id: str
    dataset_name: str
    data: T
    _stage_perf: list[StagePerfStats] = field(default_factory=list)
    _metadata: dict[str, Any] = field(default_factory=dict)
    _lineage_path: str = field(init=False, default="")

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        self.validate()

    @property
    @abstractmethod
    def num_items(self) -> int:
        """Get the number of items in this task."""

    def add_stage_perf(self, perf_stats: StagePerfStats) -> None:
        """Add performance stats for a stage."""
        self._stage_perf.append(perf_stats)

    def _set_lineage(self, parent_lineage_paths: list[str], child_segment: str | int) -> None:
        """Assign deterministic lineage to this task.

        Always overwrites ``_lineage_path`` and ``task_id``. There is no
        idempotency check — each stage transition re-hashes the task, so
        the same physical Python object passing through N stages gets N
        distinct ``task_id``s (one per stage boundary). The dedup keys
        used by resumability are captured BEFORE this method runs on a
        given output, so the rehash is safe.

        Args:
            parent_lineage_paths: Lineage paths of each parent. Empty
                strings are filtered out (so an EmptyTask parent doesn't
                contribute a leading ``"_"`` to the path).
            child_segment: Either a positional index (``int`` → coerced
                to ``str``) for plain emissions, or a string id (e.g. a
                content-based hash from :py:meth:`get_deterministic_id`)
                for source-stage emissions where stability across input
                reordering matters.
        """
        parts = [*[p for p in parent_lineage_paths if p], str(child_segment)]
        self._lineage_path = "_".join(parts)
        self.task_id = hashlib.sha256(self._lineage_path.encode()).hexdigest()[:32]

    def get_deterministic_id(self) -> str | None:
        """Return a content-based identifier for this task as a source,
        or ``None`` to fall back to the positional index.

        Override in subclasses that have stable content. The canonical
        example is :class:`FileGroupTask`, which hashes its sorted file
        paths so that adding or removing files between runs doesn't shift
        the identifiers of unchanged source partitions.

        Only called by source-stage adapters; non-source stages ignore
        this and always use positional indices."""
        return None

    def __repr__(self) -> str:
        subclass_name = self.__class__.__name__
        return f"{subclass_name}(task_id={self.task_id}, dataset_name={self.dataset_name})"

    @abstractmethod
    def validate(self) -> bool:
        """Validate the task data."""


@dataclass
class _EmptyTask(Task[None]):
    """Dummy task for testing."""

    @property
    def num_items(self) -> int:
        return 0

    def validate(self) -> bool:
        """Validate the task data."""
        return True


# Empty tasks are just used for `ls` stages
EmptyTask = _EmptyTask(task_id="empty", dataset_name="empty", data=None)
