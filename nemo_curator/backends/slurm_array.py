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
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nemo_curator.tasks import Task
from nemo_curator.utils.atomic_io import write_json_atomically_if_absent
from nemo_curator.utils.retry_manifest import METADATA_DIRNAME, CompletionManifest, read_completion_manifests

SLURM_ARRAY_ENABLED_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_ENABLED"
SLURM_ARRAY_SHARD_INDEX_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX"
SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS"
SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR = "NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX"
SLURM_ARRAY_COMPLETION_MANIFEST_NAMESPACE = "slurm_array"
SLURM_ARRAY_COMPLETION_DIRNAME = ".slurm_array_completion"
SLURM_ARRAY_RUN_CONFIG_FILENAME = "run.json"

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _get_int_env_var(env_var: str, fallback_name: str | None = None, default: int | None = None) -> int:
    """Read an integer env var, with optional fallback/default."""
    resolved_var = env_var
    env_value = os.environ.get(env_var)
    if env_value is None and fallback_name is not None:
        resolved_var = fallback_name
        env_value = os.environ.get(fallback_name)

    if env_value is None:
        if default is not None:
            return default

        if fallback_name is not None:
            msg = f"Environment variable {env_var} (or {fallback_name}) is not set"
        else:
            msg = f"Environment variable {env_var} is not set"
        raise ValueError(msg)

    try:
        return int(env_value)
    except ValueError as e:
        msg = f"Environment variable {resolved_var} must contain an integer, got {env_value!r}"
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
        explicitly_configured = SLURM_ARRAY_ENABLED_ENV_VAR in os.environ
        enabled = os.environ.get(SLURM_ARRAY_ENABLED_ENV_VAR, "1").strip().lower()
        if enabled in _FALSE_ENV_VALUES:
            return None
        if enabled not in _TRUE_ENV_VALUES:
            msg = (
                f"Environment variable {SLURM_ARRAY_ENABLED_ENV_VAR} must be one of "
                f"{sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES)}, got {enabled!r}"
            )
            raise ValueError(msg)

        has_shard_index = SLURM_ARRAY_SHARD_INDEX_ENV_VAR in os.environ or "SLURM_ARRAY_TASK_ID" in os.environ
        has_total_shards = SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR in os.environ or "SLURM_ARRAY_TASK_COUNT" in os.environ
        if not explicitly_configured and not has_shard_index and not has_total_shards:
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


@dataclass(frozen=True)
class SlurmArrayRetrySubmission:
    """Physical Slurm array indices and their logical shard offset."""

    array_indices: tuple[int, ...]
    shard_index_offset: int


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
    if resolved.minimum_shard_index < 0:
        msg = f"minimum_shard_index must be non-negative, got {resolved.minimum_shard_index}"
        raise ValueError(msg)
    if resolved.shard_index < 0:
        msg = f"shard_index must be non-negative, got {resolved.shard_index}"
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


def is_slurm_array_driver_process() -> bool:
    """Return True for the process that owns retry metadata.

    The head node has ``SLURM_NODEID == 0``; the variable is absent on
    local / single-node runs, which are also treated as head.
    """
    return os.environ.get("SLURM_NODEID", "0") == "0"


def _slurm_array_completion_dir(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path, METADATA_DIRNAME, SLURM_ARRAY_COMPLETION_DIRNAME).absolute()


def _require_manifest_int(payload: Mapping[str, object], path: Path, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"Slurm array manifest {path} must contain an integer {field}"
        raise TypeError(msg)
    return value


def _read_slurm_array_run_config(checkpoint_path: str | Path) -> tuple[Path, dict[str, object]] | None:
    config_file = _slurm_array_completion_dir(checkpoint_path) / SLURM_ARRAY_RUN_CONFIG_FILENAME
    if not config_file.is_file():
        return None
    try:
        payload = json.loads(config_file.read_text())
    except (OSError, json.JSONDecodeError) as e:
        msg = f"Failed to read Slurm array run configuration {config_file}: {e}"
        raise ValueError(msg) from e
    if not isinstance(payload, dict):
        msg = f"Slurm array run configuration must contain a JSON object: {config_file}"
        raise TypeError(msg)
    return config_file, payload


def _ensure_slurm_array_run_config(
    checkpoint_path: str | Path,
    total_shards: int,
    minimum_shard_index: int,
) -> Path:
    expected = {
        "minimum_shard_index": minimum_shard_index,
        "total_shards": total_shards,
    }
    existing = _read_slurm_array_run_config(checkpoint_path)
    if existing is not None:
        config_file, payload = existing
        if payload != expected:
            msg = (
                f"Slurm array run configuration at {config_file} is {payload}, expected {expected}; "
                "use a separate checkpoint path for each logical run"
            )
            raise ValueError(msg)
        return config_file

    config_file = _slurm_array_completion_dir(checkpoint_path) / SLURM_ARRAY_RUN_CONFIG_FILENAME
    created = write_json_atomically_if_absent(config_file, expected, separators=(",", ":"), sort_keys=True)
    if created:
        return config_file

    existing = _read_slurm_array_run_config(checkpoint_path)
    if existing is None:
        msg = f"Slurm array run configuration disappeared while initializing {config_file}"
        raise RuntimeError(msg)
    config_file, payload = existing
    if payload != expected:
        msg = (
            f"Slurm array run configuration at {config_file} is {payload}, expected {expected}; "
            "use a separate checkpoint path for each logical run"
        )
        raise ValueError(msg)
    return config_file


def build_slurm_array_completion_manifest(
    checkpoint_path: str | Path | None,
    shard_index: int,
    total_shards: int,
    minimum_shard_index: int,
) -> CompletionManifest | None:
    """Create durable completion tracking for one Slurm array shard."""
    if checkpoint_path is None:
        return None

    if total_shards <= 0:
        msg = f"total_shards must be greater than 0, got {total_shards}"
        raise ValueError(msg)
    if minimum_shard_index < 0:
        msg = f"minimum_shard_index must be non-negative, got {minimum_shard_index}"
        raise ValueError(msg)
    if shard_index < 0:
        msg = f"shard_index must be non-negative, got {shard_index}"
        raise ValueError(msg)
    maximum_shard_index = minimum_shard_index + total_shards - 1
    if not minimum_shard_index <= shard_index <= maximum_shard_index:
        msg = f"shard_index {shard_index} is outside the shard range [{minimum_shard_index}, {maximum_shard_index}]"
        raise ValueError(msg)

    _ensure_slurm_array_run_config(checkpoint_path, total_shards, minimum_shard_index)
    return CompletionManifest(
        checkpoint_path=checkpoint_path,
        namespace=SLURM_ARRAY_COMPLETION_MANIFEST_NAMESPACE,
        completion_dirname=SLURM_ARRAY_COMPLETION_DIRNAME,
        identity={
            "minimum_shard_index": minimum_shard_index,
            "shard_index": shard_index,
            "total_shards": total_shards,
        },
        flatten_identity=True,
    )


def find_slurm_array_retries(checkpoint_path: str | Path) -> SlurmArrayRetryPlan | None:
    """Return expected shard IDs that have no completion manifest."""
    records = read_completion_manifests(
        checkpoint_path,
        namespace=SLURM_ARRAY_COMPLETION_MANIFEST_NAMESPACE,
        completion_dirname=SLURM_ARRAY_COMPLETION_DIRNAME,
    )
    run_config = _read_slurm_array_run_config(checkpoint_path)
    if run_config is None:
        if records:
            msg = "Slurm array completion manifests exist without run.json"
            raise ValueError(msg)
        return None

    config_file, config_payload = run_config
    total_shards = _require_manifest_int(config_payload, config_file, "total_shards")
    minimum_shard_index = _require_manifest_int(config_payload, config_file, "minimum_shard_index")
    if total_shards <= 0:
        msg = f"Slurm array run configuration must have total_shards greater than 0, got {total_shards}"
        raise ValueError(msg)
    if minimum_shard_index < 0:
        msg = (
            "Slurm array run configuration must have a non-negative minimum_shard_index, "
            f"got {minimum_shard_index}"
        )
        raise ValueError(msg)

    completed_shard_indices = set()
    for record in records:
        record_total_shards = _require_manifest_int(record.payload, record.path, "total_shards")
        record_minimum_shard_index = _require_manifest_int(record.payload, record.path, "minimum_shard_index")
        if record_total_shards != total_shards or record_minimum_shard_index != minimum_shard_index:
            msg = f"Slurm array completion manifest {record.path} does not match the run configuration"
            raise ValueError(msg)
        completed_shard_indices.add(_require_manifest_int(record.payload, record.path, "shard_index"))

    maximum_shard_index = minimum_shard_index + total_shards - 1
    invalid_shard_indices = sorted(
        shard_index
        for shard_index in completed_shard_indices
        if not minimum_shard_index <= shard_index <= maximum_shard_index
    )
    if invalid_shard_indices:
        msg = (
            f"Completed Slurm array shard indices {invalid_shard_indices} are outside the original shard range "
            f"[{minimum_shard_index}, {maximum_shard_index}]"
        )
        raise ValueError(msg)

    expected_shard_indices = set(range(minimum_shard_index, maximum_shard_index + 1))
    return SlurmArrayRetryPlan(
        shard_indices=tuple(sorted(expected_shard_indices - completed_shard_indices)),
        total_shards=total_shards,
        minimum_shard_index=minimum_shard_index,
    )


def build_slurm_array_retry_submissions(
    retry_plan: SlurmArrayRetryPlan,
    max_array_size: int | None = None,
) -> tuple[SlurmArrayRetrySubmission, ...]:
    """Map missing logical shards to one or more physical Slurm arrays."""
    if max_array_size is not None and (isinstance(max_array_size, bool) or not isinstance(max_array_size, int)):
        msg = "max_array_size must be an integer"
        raise TypeError(msg)
    if max_array_size is not None and max_array_size <= 0:
        msg = "max_array_size must be greater than 0"
        raise ValueError(msg)
    if any(shard_index < 0 for shard_index in retry_plan.shard_indices):
        msg = "Slurm array shard indices must be non-negative"
        raise ValueError(msg)
    if not retry_plan.shard_indices:
        return ()
    if max_array_size is None:
        return (SlurmArrayRetrySubmission(array_indices=retry_plan.shard_indices, shard_index_offset=0),)

    indices_by_offset: dict[int, list[int]] = {}
    for shard_index in retry_plan.shard_indices:
        shard_index_offset = shard_index // max_array_size * max_array_size
        indices_by_offset.setdefault(shard_index_offset, []).append(shard_index - shard_index_offset)

    return tuple(
        SlurmArrayRetrySubmission(array_indices=tuple(indices), shard_index_offset=shard_index_offset)
        for shard_index_offset, indices in sorted(indices_by_offset.items())
    )


def format_slurm_array_indices(indices: Iterable[int]) -> str:
    """Format shard indices as a compact Slurm ``--array`` expression."""
    unique_indices = set(indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in unique_indices):
        msg = "Slurm array indices must be integers"
        raise TypeError(msg)
    if any(index < 0 for index in unique_indices):
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
