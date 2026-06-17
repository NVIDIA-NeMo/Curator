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
from typing import Any, Generic, TypeVar

from nemo_curator.utils.performance_utils import StagePerfStats

T = TypeVar("T")


@dataclass
class Task(ABC, Generic[T]):
    """Abstract base class for tasks in the pipeline.

    A task represents a batch of data to be processed. Different modalities
    (text, audio, video) can implement their own task types.

    Attributes:
        task_id: Deterministic identifier for this task. NOT user-settable —
            the framework assigns it via ``_set_task_id`` at every stage
            boundary. It is an underscore-joined id path through the pipeline
            DAG — the parents' ids plus this task's own segment (e.g.
            ``"abc123_0_5"`` = source ``abc123``, then child 0, then
            grandchild 5). Using the readable path directly (rather than a
            hash of it) keeps task ids easy to debug. Empty string until the
            first stage runs; two runs of the same pipeline on the same
            inputs produce byte-identical ``task_id``s across all tasks.

            A ``task_id`` that starts with ``"r"`` (followed by a uuid) is a
            fallback assigned when the parent→child mapping could NOT be
            derived — e.g. a stage that overrides ``process_batch`` with an
            ambiguous batch fan-out (M inputs → K≠M outputs). Such ids are
            NON-deterministic (differ across runs).
        dataset_name: Name of the dataset this task belongs to.
        _stage_perf: List of stages perfs this task has passed through.
    """

    dataset_name: str
    data: T
    _stage_perf: list[StagePerfStats] = field(default_factory=list)
    _metadata: dict[str, Any] = field(default_factory=dict)
    task_id: str = field(init=False, default="")

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

    def _set_task_id(self, parent_task_id: str, current_task_id_suffix: str | int) -> None:
        """Assign this task's deterministic ``task_id`` from its parent.

        The ``task_id`` is the parent id and this task's own segment joined
        by ``"_"`` — e.g. parent ``"abc123"`` + suffix ``0`` →
        ``"abc123_0"``. Always overwrites ``task_id``; there is no
        idempotency check — each stage transition re-derives it, so the
        same physical Python object passing through N stages gets N
        distinct ``task_id``s (one per stage boundary). The dedup keys
        used by resumability are captured BEFORE this method runs on a
        given output, so the rewrite is safe.

        Only a single parent id is taken: the supported mappings (1→1,
        1→N fan-out, N→N positional) each give an output exactly one
        parent. N→1 aggregations don't track ancestry — those outputs get
        a random ``"r"``-prefixed id in the adapter instead of calling this.

        Args:
            parent_task_id: ``task_id`` of the parent. An empty string
                (an unassigned / EmptyTask parent) is dropped so it doesn't
                contribute a leading ``"_"`` to the path.
            current_task_id_suffix: This task's own segment of the id
                path — appended after the parent id. Either a positional
                index (``int`` → coerced to ``str``) for plain emissions,
                or a string id (e.g. a content-based hash from
                :py:meth:`get_deterministic_id`) for source-stage emissions
                where stability across input reordering matters.
        """
        if parent_task_id:
            self.task_id = f"{parent_task_id}_{current_task_id_suffix}"
        else:
            self.task_id = str(current_task_id_suffix)

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
