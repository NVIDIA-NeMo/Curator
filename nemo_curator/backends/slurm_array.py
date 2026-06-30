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

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nemo_curator.tasks import Task
from nemo_curator.utils.retry_manifest import RetryManifest, RetryManifestRecord, read_retry_manifests

SLURM_ARRAY_ENABLED_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_ENABLED"
SLURM_ARRAY_SHARD_INDEX_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX"
SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS"
SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX"
SLURM_ARRAY_RETRY_MANIFEST_NAMESPACE = "slurm_array"
SLURM_ARRAY_RETRY_DIRNAME = ".slurm_array_retry"

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _get_int_env_var(env_var: str, fallback_name: str | None = None, default: int | None = None) -> int:
    """Read an integer env var, with optional fallback/default."""
    env_value = os.environ.get(env_var)
    if env_value is None:
        if fallback_name is not None:
            env_var = fallback_name
            env_value = os.environ.get(env_var)

        if env_value is None:
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
    """Source-task sharding settings for one Slurm array task."""

    shard_index: int
    total_shards: int
    minimum_shard_index: int = 0

    @classmethod
    def from_env(cls) -> "SlurmArrayConfig | None":
        """Build config from Curator or Slurm env vars unless explicitly disabled."""
        enabled = os.environ.get(SLURM_ARRAY_ENABLED_ENV_VAR, "1").strip().lower()
        if enabled in _FALSE_ENV_VALUES:
            return None
        if enabled not in _TRUE_ENV_VALUES:
            msg = (
                f"Environment variable {SLURM_ARRAY_ENABLED_ENV_VAR} must be one of "
                f"{sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES)}, got {enabled!r}"
            )
            raise ValueError(msg)

        has_shard_index = (
            SLURM_ARRAY_SHARD_INDEX_ENV_VAR in os.environ or "SLURM_ARRAY_TASK_ID" in os.environ
        )
        has_total_shards = (
            SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR in os.environ or "SLURM_ARRAY_TASK_COUNT" in os.environ
        )
        if not has_shard_index and not has_total_shards:
            return None

        return cls(
            shard_index=_get_int_env_var(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, "SLURM_ARRAY_TASK_ID"),
            total_shards=_get_int_env_var(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, "SLURM_ARRAY_TASK_COUNT"),
            minimum_shard_index=_get_int_env_var(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, default=0),
        )


@dataclass(frozen=True)
class SlurmArrayRetryPlan:
    """Outstanding shard IDs and the original logical shard configuration."""

    shard_indices: tuple[int, ...]
    total_shards: int
    minimum_shard_index: int


def configure_slurm_array_source_filtering(
    shard_index: int,
    total_shards: int,
    minimum_shard_index: int,
) -> None:
    """Set env vars consumed by source-stage filtering."""
    os.environ[SLURM_ARRAY_ENABLED_ENV_VAR] = "1"
    os.environ[SLURM_ARRAY_SHARD_INDEX_ENV_VAR] = str(shard_index)
    os.environ[SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR] = str(total_shards)
    os.environ[SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR] = str(minimum_shard_index)


def resolve_slurm_array_config(is_source_stage: bool) -> SlurmArrayConfig | None:
    """Resolve filtering config for source stages."""
    if not is_source_stage:
        return None

    resolved = SlurmArrayConfig.from_env()
    if resolved is None:
        return None

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

    return resolved


def slurm_array_shard_for_task(task: Task, slurm_array: SlurmArrayConfig) -> int:
    """Assign a task to a shard by hashing its deterministic task ID."""
    digest = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % slurm_array.total_shards + slurm_array.minimum_shard_index


def filter_slurm_array_source_tasks(
    tasks: list[Task],
    slurm_array: SlurmArrayConfig | None,
    stage_name: str,
) -> list[Task]:
    """Keep only source tasks assigned to the active Slurm array shard."""
    nondeterministic_task_ids = [task.task_id for task in tasks if task.task_id.startswith("r")]
    if nondeterministic_task_ids:
        msg = (
            "Slurm array source filtering requires deterministic task IDs, but stage "
            f"{stage_name} emitted ambiguous source task IDs: {nondeterministic_task_ids[:5]}"
        )
        raise ValueError(msg)

    assigned_tasks = [
        task for task in tasks if slurm_array_shard_for_task(task, slurm_array) == slurm_array.shard_index
    ]

    msg = (
        f"Slurm array shard {slurm_array.shard_index}/{slurm_array.total_shards}: "
        f"assigned {len(assigned_tasks)} of {len(tasks)} source tasks for stage {stage_name}"
    )
    if len(assigned_tasks) == 0 and len(tasks) > 0:
        logger.warning(msg)
    else:
        logger.info(msg)

    return assigned_tasks


def is_slurm_array_driver_process(use_slurm: bool) -> bool:
    """Return true for the process that owns retry metadata."""
    return not use_slurm or os.environ.get("SLURM_NODEID", "0") == "0"


def build_slurm_array_retry_manifest(
    checkpoint_path: str | None,
    shard_index: int,
    total_shards: int,
    minimum_shard_index: int,
) -> RetryManifest | None:
    """Create a retry manifest for one Slurm array shard."""
    if checkpoint_path is None:
        return None

    return RetryManifest(
        checkpoint_path=checkpoint_path,
        namespace=SLURM_ARRAY_RETRY_MANIFEST_NAMESPACE,
        retry_dirname=SLURM_ARRAY_RETRY_DIRNAME,
        identity={
            "minimum_shard_index": minimum_shard_index,
            "shard_index": shard_index,
            "total_shards": total_shards,
        },
        flatten_identity=True,
    )


def _require_manifest_int(record: RetryManifestRecord, field: str) -> int:
    value = record.payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"Slurm array retry manifest {record.path} must contain an integer {field}"
        raise ValueError(msg)
    return value


def find_slurm_array_retries(checkpoint_path: str | Path) -> SlurmArrayRetryPlan | None:
    """Build a retry plan from all outstanding Slurm array manifests."""
    records = read_retry_manifests(
        checkpoint_path,
        namespace=SLURM_ARRAY_RETRY_MANIFEST_NAMESPACE,
        retry_dirname=SLURM_ARRAY_RETRY_DIRNAME,
    )
    if not records:
        return None

    shard_indices = set()
    total_shards_values = set()
    minimum_shard_index_values = set()
    for record in records:
        shard_indices.add(_require_manifest_int(record, "shard_index"))
        total_shards_values.add(_require_manifest_int(record, "total_shards"))
        minimum_shard_index_values.add(_require_manifest_int(record, "minimum_shard_index"))

    if len(total_shards_values) != 1 or len(minimum_shard_index_values) != 1:
        msg = "Slurm array retry manifests contain multiple shard configurations; split them by logical run"
        raise ValueError(msg)

    total_shards = next(iter(total_shards_values))
    minimum_shard_index = next(iter(minimum_shard_index_values))
    if total_shards <= 0:
        msg = f"Slurm array retry manifests must have total_shards greater than 0, got {total_shards}"
        raise ValueError(msg)

    maximum_shard_index = minimum_shard_index + total_shards - 1
    invalid_shard_indices = sorted(
        shard_index
        for shard_index in shard_indices
        if not minimum_shard_index <= shard_index <= maximum_shard_index
    )
    if invalid_shard_indices:
        msg = (
            f"Slurm array retry shard indices {invalid_shard_indices} are outside the original shard range "
            f"[{minimum_shard_index}, {maximum_shard_index}]"
        )
        raise ValueError(msg)

    return SlurmArrayRetryPlan(
        shard_indices=tuple(sorted(shard_indices)),
        total_shards=total_shards,
        minimum_shard_index=minimum_shard_index,
    )


def format_slurm_array_indices(indices: Iterable[int]) -> str:
    """Format shard indices as a compact Slurm ``--array`` expression."""
    unique_indices = set(indices)
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in unique_indices):
        msg = "Slurm array indices must be non-negative integers"
        raise ValueError(msg)
    sorted_indices = sorted(unique_indices)
    if not sorted_indices:
        return ""

    ranges = []
    range_start = sorted_indices[0]
    range_end = range_start
    for index in sorted_indices[1:]:
        if index == range_end + 1:
            range_end = index
            continue

        ranges.append(str(range_start) if range_start == range_end else f"{range_start}-{range_end}")
        range_start = range_end = index

    ranges.append(str(range_start) if range_start == range_end else f"{range_start}-{range_end}")
    return ",".join(ranges)
