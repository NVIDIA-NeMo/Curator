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

import datetime
import hashlib
import json
import os
import socket
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.core.utils import ignore_ray_head_node
from nemo_curator.tasks import Task
from nemo_curator.tasks.sentinels import FailedTask, NoneTask
from nemo_curator.utils.performance_utils import StageTimer

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


FAILED_TASKS_DIR_ENV_VAR = "NEMO_CURATOR_FAILED_TASKS_DIR"
SLURM_ARRAY_ENABLED_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_ENABLED"
SLURM_ARRAY_SHARD_INDEX_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX"
SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS"
SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX"

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _get_int_or_env_var(input_value: int | str | None, default_name: str | None = None) -> int:
    if type(input_value) is int:
        return input_value

    if type(input_value) is str:
        try:
            return int(input_value)
        except ValueError:
            env_var = input_value
    else:
        env_var = default_name

    if env_var is None:
        msg = f"Invalid input value: {input_value}, must be an integer or a string"
        raise ValueError(msg)

    env_value = os.environ.get(env_var)
    if env_value is None:
        msg = f"Environment variable {env_var} is not set"
        raise ValueError(msg)
    try:
        return int(env_value)
    except ValueError as e:
        msg = f"Environment variable {env_var} must contain an integer, got {env_value!r}"
        raise ValueError(msg) from e


def _get_int_env_var(env_var: str, fallback_name: str | None = None, default: int | None = None) -> int:
    env_value = os.environ.get(env_var)
    if env_value is None:
        if fallback_name is not None:
            return _get_int_or_env_var(None, fallback_name)
        if default is not None:
            return default

        msg = f"Environment variable {env_var} is not set"
        raise ValueError(msg)

    try:
        return int(env_value)
    except ValueError as e:
        msg = f"Environment variable {env_var} must contain an integer, got {env_value!r}"
        raise ValueError(msg) from e


@dataclass
class SlurmArrayConfig:
    """Configuration for assigning source tasks to one Slurm array task."""

    shard_index: int | str | None = None
    total_shards: int | str | None = None
    minimum_shard_index: int | str = 0

    def resolve(self) -> "SlurmArrayConfig":
        """Resolve integer values from explicit integers or environment variables."""
        return SlurmArrayConfig(
            shard_index=_get_int_or_env_var(self.shard_index, "SLURM_ARRAY_TASK_ID"),
            total_shards=_get_int_or_env_var(self.total_shards, "SLURM_ARRAY_TASK_COUNT"),
            minimum_shard_index=_get_int_or_env_var(self.minimum_shard_index),
        )

    @classmethod
    def from_env(cls) -> "SlurmArrayConfig | None":
        """Return Slurm array config when source-task filtering is enabled."""
        enabled = os.environ.get(SLURM_ARRAY_ENABLED_ENV_VAR, "")
        if enabled.strip().lower() not in _TRUE_ENV_VALUES:
            return None

        return cls(
            shard_index=_get_int_env_var(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, "SLURM_ARRAY_TASK_ID"),
            total_shards=_get_int_env_var(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, "SLURM_ARRAY_TASK_COUNT"),
            minimum_shard_index=_get_int_env_var(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, default=0),
        )


def _safe_filename_token(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _write_failed_task_marker(marker_dir: Path, stage_name: str, task: FailedTask) -> None:
    created_at = datetime.datetime.now(datetime.UTC)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    payload: dict[str, str | int] = {
        "created_at": created_at.isoformat(),
        "stage_name": stage_name,
        "task_id": task.task_id,
        "dataset_name": task.dataset_name,
        "task_type": type(task).__name__,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }

    marker_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        "failed_task_"
        f"stage-{_safe_filename_token(stage_name)}_"
        f"task-{_safe_filename_token(task.task_id)}_"
        f"pid-{os.getpid()}_"
        f"{timestamp}_{uuid.uuid4().hex}.json"
    )
    final_path = marker_dir / filename

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=marker_dir,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.write("\n")

        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


@dataclass
class NodeInfo:
    """Generic node information for setup_on_node calls across backends.
    Simplified to match Xenna's structure.
    """

    node_id: str = ""


@dataclass
class WorkerMetadata:
    """Generic worker metadata for setup_on_node calls across backends.
    Simplified to match Xenna's structure. The allocation field can contain
    backend-specific allocation information.
    """

    worker_id: str = ""
    allocation: Any = None  # Backend-specific allocation info


class BaseExecutor(ABC):
    """Executor for a pipeline."""

    def __init__(self, config: dict[str, Any] | None = None, ignore_head_node: bool = False):
        self.config = config or {}
        self.ignore_head_node = ignore_head_node or ignore_ray_head_node()

    @abstractmethod
    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> None:
        """Execute the pipeline."""


class BaseStageAdapter:
    """Adapts ProcessingStage to an execution backend, if needed."""

    def __init__(self, stage: "ProcessingStage"):
        self.stage = stage

    def process_batch(self, tasks: list[Task]) -> list[Task]:
        """Process a batch of tasks.

        Args:
            tasks (list[Task]): List of tasks to process

        Returns:
            list[Task]: List of processed tasks
        """
        # Lazy initialize timer if needed
        if not hasattr(self, "_timer") or self._timer is None:
            self._timer = StageTimer(self.stage)

        # Calculate input data size for timer
        input_size = sum(task.num_items for task in tasks)
        # Initialize performance timer for this batch
        self._timer.reinit(input_size)

        with self._timer.time_process(input_size):
            # Use the batch processing logic
            results = self.stage.process_batch(tasks)

        # A returned ``None`` ("filter this slot") becomes a NoneTask so every
        # output is a real Task that gets a task_id. Sentinels (NoneTask /
        # FailedTask) carry no identity and are stripped again before this
        # method returns.
        results = [NoneTask() if r is None else r for r in results]

        # Guarantee every emitted task has a task_id (derived id, or uuid fallback).
        results = self._post_process_task_ids(tasks, results)

        self._record_failed_tasks([r for r in results if isinstance(r, FailedTask)])

        # Sentinels never propagate to the next stage.
        results = [r for r in results if not isinstance(r, (NoneTask, FailedTask))]

        results = self._filter_slurm_array_source_tasks(results)

        # Log performance stats and add to result tasks
        _, stage_perf_stats = self._timer.log_stats()
        # Consume and attach any custom metrics recorded by the stage during this call
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)
        for task in results:
            task.add_stage_perf(stage_perf_stats)

        return results

    def _record_failed_tasks(self, failed_tasks: list[FailedTask]) -> None:
        marker_dir = os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
        if not marker_dir or not failed_tasks:
            return

        marker_path = Path(marker_dir)
        for task in failed_tasks:
            try:
                _write_failed_task_marker(marker_path, self.stage.name, task)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to write FailedTask marker to {marker_path}: {e}")

    def _filter_slurm_array_source_tasks(self, tasks: list[Task]) -> list[Task]:
        """Keep only source tasks assigned to this Slurm array shard."""
        slurm_array = self._resolve_slurm_array_config()
        if slurm_array is None:
            return tasks

        nondeterministic_task_ids = [task.task_id for task in tasks if task.task_id.startswith("r")]
        if nondeterministic_task_ids:
            msg = (
                "Slurm array source filtering requires deterministic task IDs, but stage "
                f"{self.stage.name} emitted ambiguous source task IDs: {nondeterministic_task_ids[:5]}"
            )
            raise ValueError(msg)

        assigned_tasks = [
            task
            for task in tasks
            if self._slurm_array_shard_for_task(task, slurm_array) == slurm_array.shard_index
        ]

        msg = (
            f"Slurm array shard {slurm_array.shard_index}/{slurm_array.total_shards}: "
            f"assigned {len(assigned_tasks)} of {len(tasks)} source tasks for stage {self.stage.name}"
        )
        if len(assigned_tasks) == 0 and len(tasks) > 0:
            logger.warning(msg)
        else:
            logger.info(msg)

        return assigned_tasks

    def _resolve_slurm_array_config(self) -> SlurmArrayConfig | None:
        if not getattr(self.stage, "is_source_stage", False):
            return None

        if not hasattr(self, "_resolved_slurm_array"):
            resolved = SlurmArrayConfig.from_env()
            if resolved is not None:
                if resolved.total_shards <= 0:
                    msg = f"total_shards must be greater than 0, got {resolved.total_shards}"
                    raise ValueError(msg)

                min_assignable_shard_index = resolved.minimum_shard_index
                max_assignable_shard_index = resolved.minimum_shard_index + resolved.total_shards - 1
                if not min_assignable_shard_index <= resolved.shard_index <= max_assignable_shard_index:
                    logger.warning(
                        "shard_index={} is outside the assignable shard range [{}, {}]. "
                        "This task will not receive any source tasks.",
                        resolved.shard_index,
                        min_assignable_shard_index,
                        max_assignable_shard_index,
                    )
            self._resolved_slurm_array = resolved

        return self._resolved_slurm_array

    def _slurm_array_shard_for_task(self, task: Task, slurm_array: SlurmArrayConfig) -> int:
        digest = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % slurm_array.total_shards + slurm_array.minimum_shard_index

    def _post_process_task_ids(self, input_tasks: list[Task], output_tasks: list[Task | None]) -> list[Task]:
        """Assign a deterministic ``task_id`` to every emitted task.

        This is the single place task ids are assigned — it runs for every
        stage on every backend (all backend adapters subclass this), so it
        makes no difference whether a stage defines ``process`` or overrides
        ``process_batch``. ``task_id`` is the task's id path (parents + own segment); ids are
        re-derived at each stage boundary so the same object passing through
        N stages gets N ids.

        The input→output mapping decides each output's PARENT; whether the
        stage is a source decides each output's SEGMENT (content id vs index)
        — the two are independent. ``None`` outputs (Curator's "return None to
        filter") are NOT removed before the length check — keeping them in
        place preserves positional alignment for filter stages — and are then
        dropped from the returned list.

        - single input → every output is its child (fan-out): ``parent_<seg>``
        - ``len(output) == len(input)`` → positional 1:1: each ``parent_i_<seg>``;
          a ``None`` slot just means input ``i`` was filtered.
        - any other (ambiguous) cardinality across a batch → a random ``uuid``
          prefixed with ``"r"`` (e.g. ``"r3f9a…"``), so ``task_id`` is never
          empty even when a derived id is not possible. The ``"r"`` prefix flags
          the id as non-deterministic / ancestry-not-tracked (see
          ``Task.task_id`` docstring).

        ``seg`` is the output's content id (``Task.get_deterministic_id()``)
        for a source stage when available, else the positional index — so a
        source partition keeps a stable id across reorderings regardless of
        whether the source is 1→N or N→N.

        Note: a stage that BOTH filters and fans out within a single batch
        (returning a flat list rather than a per-input slot) cannot be mapped
        positionally; if its length happens to equal the input length the 1:1
        assumption may misattribute parents. That combination is unsupported
        unless the stage preserves an unambiguous input -> output mapping.
        """
        is_source = getattr(self.stage, "is_source_stage", False)

        if len(input_tasks) == 1:
            # Fan-out (incl. a source reading from EmptyTask): every non-None
            # output is a child of the single input.
            parent_id = input_tasks[0].task_id
            out: list[Task] = [t for t in output_tasks if t is not None]
            for i, task in enumerate(out):
                suffix = (task.get_deterministic_id() or i) if is_source else i
                task._set_task_id(parent_id, suffix)
            return out

        if len(output_tasks) == len(input_tasks):
            # Positional 1:1. None is kept above so a filtered slot still lines
            # up with its own parent; drop the None slots from the result.
            out = []
            for parent, task in zip(input_tasks, output_tasks, strict=True):
                if task is None:
                    continue
                suffix = (task.get_deterministic_id() or 0) if is_source else 0
                task._set_task_id(parent.task_id, suffix)
                out.append(task)
            return out

        # Ambiguous cardinality across a batch: a derived id is not possible. Use a
        # random "r"-prefixed uuid so task_id is non-empty but clearly flagged
        # non-deterministic.
        out = [t for t in output_tasks if t is not None]
        for task in out:
            task.task_id = "r" + uuid.uuid4().hex
        return out

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup the stage on a node.

        Args:
            node_info (NodeInfo, optional): Information about the node
            worker_metadata (WorkerMetadata, optional): Information about the worker
        """
        # Call the underlying stage's setup_on_node method
        # Some backends may provide node/worker info, others may not
        self.stage.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup the stage once per actor.

        Args:
            worker_metadata (WorkerMetadata, optional): Information about the worker
        """
        self.stage.setup(worker_metadata)

    def teardown(self) -> None:
        """Teardown the stage once per actor."""
        self.stage.teardown()
