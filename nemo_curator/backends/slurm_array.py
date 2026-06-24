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
from dataclasses import dataclass

from loguru import logger

from nemo_curator.tasks import Task
from nemo_curator.tasks.sentinels import FailedTask
from nemo_curator.utils.retry_manifest import RetryManifest

SLURM_ARRAY_ENABLED_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_ENABLED"
SLURM_ARRAY_SHARD_INDEX_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX"
SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS"
SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX"
SLURM_ARRAY_RETRY_MANIFEST_NAMESPACE = "slurm_array"
SLURM_ARRAY_RETRY_DIRNAME = ".slurm_array_retry"

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


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
        """Build config from Curator env vars, falling back to Slurm env vars."""
        enabled = os.environ.get(SLURM_ARRAY_ENABLED_ENV_VAR, "")
        if enabled.strip().lower() not in _TRUE_ENV_VALUES:
            return None

        return cls(
            shard_index=_get_int_env_var(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, "SLURM_ARRAY_TASK_ID"),
            total_shards=_get_int_env_var(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, "SLURM_ARRAY_TASK_COUNT"),
            minimum_shard_index=_get_int_env_var(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, default=0),
        )


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
    if slurm_array is None:
        return tasks

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


def raise_for_failed_source_tasks_with_slurm_array(
    stage_name: str,
    failed_tasks: list[FailedTask],
    slurm_array: SlurmArrayConfig | None,
) -> None:
    """Reject source-stage FailedTasks, which cannot be retried by shard."""
    if failed_tasks and slurm_array is not None:
        msg = (
            f"Source stage {stage_name} emitted FailedTask while Slurm array filtering is enabled. "
            "This is not supported because the failed source task cannot be assigned to a retry shard "
            "reliably. Raise an exception from the source stage instead."
        )
        raise ValueError(msg)


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
