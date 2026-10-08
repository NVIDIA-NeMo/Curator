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
import math
import os
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from operator import eq, ge, gt, le, lt, ne
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

import soundfile
import torch
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, Role, StageContract, StaticHints
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask

_VALUE_OPERATORS = {"lt": lt, "le": le, "eq": eq, "ne": ne, "ge": ge, "gt": gt}


def get_audio_duration(audio_filepath: str) -> float:
    """Get the duration of the audio file in seconds."""
    try:
        info = soundfile.info(audio_filepath)
        return info.frames / info.samplerate
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to get duration for audio file {audio_filepath}: {e}")
        return -1.0


@dataclass
class GetAudioDurationStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Compute audio duration from the file at *audio_filepath_key* and
    store the result under *duration_key*.

    Args:
        audio_filepath_key: Key to get path to wav file.
        duration_key: Key to put audio duration.
        waveform_key: Key for an in-memory waveform tensor.
        sample_rate_key: Key for the in-memory waveform sample rate.
        input_residency: Which input to use — "file" (audio_filepath only; default,
            unchanged), "waveform" (in-memory only), or "auto" (waveform first, file fallback).
    """

    name: str = "GetAudioDurationStage"
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    _: KW_ONLY
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    input_residency: Literal["file", "waveform", "auto"] = "file"

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        import soundfile

        self._soundfile = soundfile

    def inputs(self) -> tuple[list[str], list[str]]:
        if self.input_residency == "waveform":
            return [], [self.waveform_key, self.sample_rate_key]
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.duration_key]

    def describe(self) -> StageContract:
        # Lazy import avoids a module-level cycle (_residency imports from common).
        from nemo_curator.stages.audio._agent._residency import residency_read_specs

        return StageContract(
            preferred_reads=(
                IOSpec(data_keys=[self.waveform_key, self.sample_rate_key], accepts=["waveform"])
                if self.input_residency == "auto"
                else None
            ),
            reads_one_of=residency_read_specs(
                self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=[self.duration_key]),
            key_roles={self.duration_key: "duration"},
            gates=Gates(per_row_independent=True),
        )

    def __post_init__(self) -> None:
        from nemo_curator.stages.audio._agent._residency import validate_input_residency

        validate_input_residency(self.input_residency, stage_name=self.name)
        if self.input_residency != "file":
            from nemo_curator.stages.audio._agent._residency import validate_audio_key_configuration

            inputs = {"waveform_key": self.waveform_key, "sample_rate_key": self.sample_rate_key}
            if self.input_residency == "auto":
                inputs["audio_filepath_key"] = self.audio_filepath_key
            validate_audio_key_configuration(
                self.name, input_keys=inputs, output_keys={"duration_key": self.duration_key}
            )

    def validate_input(self, task: AudioTask) -> bool:
        if not super().validate_input(task):
            return False
        return self.input_residency != "waveform" or self._resident_rate(task.data) is not None

    def input_spec_for_task(self, task: AudioTask) -> tuple[list[str], list[str]]:
        attrs, keys = super().input_spec_for_task(task)
        if self.input_residency == "auto" and self._resident_rate(task.data) is not None:
            keys = [key for key in keys if key != self.audio_filepath_key]
            keys.extend([self.waveform_key, self.sample_rate_key])
        return attrs, keys

    def _resident_rate(self, data: dict[str, Any]) -> int | None:
        from nemo_curator.stages.audio._agent._residency import resident_sample_rate

        if data.get(self.waveform_key) is None or data.get(self.sample_rate_key) is None:
            return None
        try:
            return resident_sample_rate(
                data[self.sample_rate_key], sample_rate_key=self.sample_rate_key, stage_name=self.name
            )
        except ValueError:
            return None

    def _resolve_duration(self, data: dict[str, Any]) -> float:
        """Prefer valid resident audio only in the explicitly selected residency modes."""
        if self.input_residency != "file":
            rate = self._resident_rate(data)
            if rate is not None:
                return ensure_waveform_2d(data[self.waveform_key]).shape[-1] / rate
            if self.input_residency == "waveform":
                msg = f"[{self.name}] Missing waveform or invalid resident sample rate"
                raise ValueError(msg)
        return get_audio_duration(data[self.audio_filepath_key])

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        duration = self._resolve_duration(task.data)
        task.data[self.duration_key] = duration
        self._log_metrics({"process_time": time.perf_counter() - t0, "duration": max(duration, 0.0)})
        return task


class PreserveByValueStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Filter entries by comparing *input_value_key* against *target_value*.

    Returns ``None`` from ``process()`` to drop entries that fail the
    comparison, matching the text-modality filter convention.

    Args:
        input_value_key: The field in the dataset entries to evaluate.
        target_value: The value to compare with.
        operator: Comparison operator (lt, le, eq, ne, ge, gt).
        missing_value_policy: ``"error"`` (default) preserves the historical
            validation error for a missing input key; ``"drop"`` removes that row.
    """

    name: str = "PreserveByValueStage"
    BATCH_ONLY = True  # process() raises; only process_batch is implemented (agent-discovery hint)

    def __init__(
        self,
        input_value_key: str,
        target_value: float | str,
        operator: str = "eq",
        missing_value_policy: Literal["error", "drop"] = "error",
    ):
        self.input_value_key = input_value_key
        self.target_value = target_value
        if operator not in _VALUE_OPERATORS:
            msg = f"Operator must be one of: {', '.join(_VALUE_OPERATORS)}"
            raise ValueError(msg)
        if missing_value_policy not in {"error", "drop"}:
            msg = "missing_value_policy must be 'error' or 'drop'"
            raise ValueError(msg)
        self.operator = _VALUE_OPERATORS[operator]
        self.missing_value_policy = missing_value_policy

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.input_value_key] if self.missing_value_policy == "error" else []),
            optional_reads=IOSpec(data_keys=[self.input_value_key] if self.missing_value_policy == "drop" else []),
            cardinality="filter",
            # Compares one row against a fixed target value, so batching changes throughput
            # rather than the verdict -- no row's fate depends on the rows beside it.
            gates=Gates(per_row_independent=True),
        )

    def process(self, task: AudioTask) -> AudioTask | None:
        msg = "PreserveByValueStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        t0 = time.perf_counter()
        results = []
        for task in tasks:
            if not self.validate_input(task):
                if self.missing_value_policy == "drop":
                    continue
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)
            if self.operator(task.data[self.input_value_key], self.target_value):
                results.append(task)
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "input_count": len(tasks),
                "output_count": len(results),
                "filtered_count": len(tasks) - len(results),
            }
        )
        return results


@dataclass(frozen=True)
class _ValueCondition:
    input_value_key: str
    target_value: float | str | bool
    operator: str


class PreserveByValueConditionsStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Keep a row or nested child according to flat scalar conditions.

    ``conditions`` accepts either a list of
    ``{"input_value_key", "target_value", "operator"}`` mappings or a mapping
    from input key to ``{"target_value", "operator"}``. A scalar mapping value
    is shorthand for an equality condition. Conditions use AND semantics by
    default; ``condition_logic="or"`` keeps an item when at least one condition
    passes. Missing condition keys always fail closed before logic is applied.

    Args:
        conditions: Non-empty list or mapping of scalar comparisons.
        missing_value_policy: ``"error"`` raises when any condition key is
            absent; ``"drop"`` removes rows with any absent condition key.
        items_key: Optional top-level ``task.data`` key containing a list of
            mapping-like children. When set, conditions filter that list in
            place of filtering top-level rows. This is a single-level operation
            and never recursively descends into child values.
        drop_parent_if_empty: In nested mode, drop the parent AudioTask when no
            child survives. Ignored when ``items_key`` is ``None``.
        condition_logic: ``"and"`` (default) requires every condition to pass;
            ``"or"`` requires at least one condition to pass.
    """

    name: str = "PreserveByValueConditionsStage"
    BATCH_ONLY = True

    def __init__(
        self,
        conditions: list[Mapping[str, Any]] | Mapping[str, Any],
        missing_value_policy: Literal["error", "drop"] = "error",
        items_key: str | None = None,
        drop_parent_if_empty: bool = True,
        condition_logic: Literal["and", "or"] = "and",
    ):
        if missing_value_policy not in {"error", "drop"}:
            msg = "missing_value_policy must be 'error' or 'drop'"
            raise ValueError(msg)
        if items_key is not None and (not isinstance(items_key, str) or not items_key):
            msg = "items_key must be None or a non-empty string"
            raise ValueError(msg)
        if not isinstance(drop_parent_if_empty, bool):
            msg = "drop_parent_if_empty must be a boolean"
            raise TypeError(msg)
        if condition_logic not in {"and", "or"}:
            msg = "condition_logic must be 'and' or 'or'"
            raise ValueError(msg)
        self.missing_value_policy = missing_value_policy
        self.items_key = items_key
        self.drop_parent_if_empty = drop_parent_if_empty
        self.condition_logic = condition_logic
        self._conditions = self._normalize_conditions(conditions)

    @staticmethod
    def _normalize_conditions(  # noqa: C901, PLR0912 - one validation branch per accepted condition shape
        conditions: list[Mapping[str, Any]] | Mapping[str, Any],
    ) -> tuple[_ValueCondition, ...]:
        raw_conditions: list[Mapping[str, Any]]
        if isinstance(conditions, Mapping):
            raw_conditions = []
            for key, value in conditions.items():
                if isinstance(value, Mapping):
                    raw_conditions.append(
                        {
                            "input_value_key": key,
                            "target_value": value.get("target_value"),
                            "operator": value.get("operator", "eq"),
                            "_has_target": "target_value" in value,
                        }
                    )
                else:
                    raw_conditions.append(
                        {
                            "input_value_key": key,
                            "target_value": value,
                            "operator": "eq",
                            "_has_target": True,
                        }
                    )
        elif isinstance(conditions, list):
            raw_conditions = list(conditions)
        else:
            msg = "conditions must be a non-empty list or mapping"
            raise TypeError(msg)
        if not raw_conditions:
            msg = "conditions must contain at least one scalar comparison"
            raise ValueError(msg)

        normalized: list[_ValueCondition] = []
        for index, condition in enumerate(raw_conditions):
            if not isinstance(condition, Mapping):
                msg = f"conditions[{index}] must be a mapping"
                raise TypeError(msg)
            key = condition.get("input_value_key")
            if not isinstance(key, str) or not key:
                msg = f"conditions[{index}].input_value_key must be a non-empty string"
                raise ValueError(msg)
            if not condition.get("_has_target", "target_value" in condition):
                msg = f"conditions[{index}] must define target_value"
                raise ValueError(msg)
            target = condition.get("target_value")
            if isinstance(target, float) and not math.isfinite(target):
                msg = f"conditions[{index}].target_value must be finite"
                raise ValueError(msg)
            if not isinstance(target, (bool, int, float, str)):
                msg = f"conditions[{index}].target_value must be a JSON scalar"
                raise TypeError(msg)
            operator = condition.get("operator", "eq")
            if operator not in _VALUE_OPERATORS:
                msg = f"conditions[{index}].operator must be one of: {', '.join(_VALUE_OPERATORS)}"
                raise ValueError(msg)
            normalized.append(
                _ValueCondition(
                    input_value_key=key,
                    target_value=target,
                    operator=str(operator),
                )
            )
        return tuple(normalized)

    @property
    def conditions(self) -> list[dict[str, Any]]:
        """Configured comparisons as an independent snapshot of the executed semantics."""
        return list(self.normalized_conditions)

    @property
    def normalized_conditions(self) -> tuple[dict[str, Any], ...]:
        """Canonical conditions for deterministic planning and comparison."""
        return tuple(
            {
                "input_value_key": condition.input_value_key,
                "target_value": condition.target_value,
                "operator": condition.operator,
            }
            for condition in self._conditions
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        if self.items_key is not None:
            return [], [self.items_key]
        return [], [condition.input_value_key for condition in self._conditions]

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.items_key is not None:
            return [], [self.items_key]
        return [], [condition.input_value_key for condition in self._conditions]

    def describe(self) -> StageContract:
        logic = self.condition_logic.upper()
        if self.items_key is not None:
            return StageContract(
                reads=IOSpec(
                    data_keys=[self.items_key],
                    segment_data_keys=(
                        [c.input_value_key for c in self._conditions] if self.missing_value_policy == "error" else []
                    ),
                ),
                optional_reads=IOSpec(
                    segment_data_keys=(
                        [c.input_value_key for c in self._conditions] if self.missing_value_policy == "drop" else []
                    )
                ),
                writes=IOSpec(data_keys=[self.items_key]),
                cardinality="filter" if self.drop_parent_if_empty else "1:1 nested-list",
                cardinality_options=["filter", "1:1 nested-list"],
                iteration_key=None if self.drop_parent_if_empty else self.items_key,
                description=(
                    f"Filter mapping-like children in task.data[{self.items_key!r}] with "
                    f"one-level {logic} conditions; nested values are not traversed."
                ),
                gates=Gates(per_row_independent=True),
            )
        keys = [condition.input_value_key for condition in self._conditions]
        return StageContract(
            reads=IOSpec(data_keys=keys if self.missing_value_policy == "error" else []),
            optional_reads=IOSpec(data_keys=keys if self.missing_value_policy == "drop" else []),
            cardinality="filter",
            description=f"Filter top-level AudioTask rows with {logic} conditions.",
            gates=Gates(per_row_independent=True),
        )

    def process(self, task: AudioTask) -> AudioTask | None:
        msg = "PreserveByValueConditionsStage only supports process_batch"
        raise NotImplementedError(msg)

    def _nested_items(
        self,
        task: AudioTask,
        items_key: str,
    ) -> list[Mapping[str, Any]]:
        """Return and structurally validate the configured one-level child list."""
        if items_key not in task.data:
            msg = f"Task {task!s} is missing nested items_key {items_key!r}"
            raise ValueError(msg)
        items = task.data[items_key]
        if not isinstance(items, list):
            msg = f"Task {task!s} nested items_key {items_key!r} must contain a list, got {type(items).__name__}"
            raise TypeError(msg)
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                msg = (
                    f"Task {task!s} nested items_key {items_key!r} "
                    f"child {index} must be mapping-like, got {type(item).__name__}"
                )
                raise TypeError(msg)
        return items

    def _nested_item_passes(
        self,
        task: AudioTask,
        index: int,
        item: Mapping[str, Any],
        items_key: str,
    ) -> bool:
        """Apply configured conditions to one validated direct child."""
        for condition in self._conditions:
            if condition.input_value_key not in item:
                if self.missing_value_policy == "drop":
                    return False
                msg = (
                    f"Task {task!s} nested items_key {items_key!r} child "
                    f"{index} is missing condition key {condition.input_value_key!r}"
                )
                raise ValueError(msg)
        results = (
            _VALUE_OPERATORS[condition.operator](
                item[condition.input_value_key],
                condition.target_value,
            )
            for condition in self._conditions
        )
        return all(results) if self.condition_logic == "and" else any(results)

    def _process_nested_batch(
        self,
        tasks: list[AudioTask],
        items_key: str,
    ) -> list[AudioTask]:
        """Filter direct children, replacing only the configured list field."""
        results: list[AudioTask] = []
        for task in tasks:
            items = self._nested_items(task, items_key)
            survivors = [
                item for index, item in enumerate(items) if self._nested_item_passes(task, index, item, items_key)
            ]
            task.data[items_key] = survivors
            if survivors or not self.drop_parent_if_empty:
                results.append(task)
        return results

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        t0 = time.perf_counter()
        if self.items_key is not None:
            results = self._process_nested_batch(tasks, self.items_key)
        else:
            results = []
            for task in tasks:
                for condition in self._conditions:
                    if condition.input_value_key not in task.data:
                        if self.missing_value_policy == "drop":
                            break
                        msg = f"Task {task!s} failed validation for stage {self}"
                        raise ValueError(msg)
                else:
                    condition_results = (
                        _VALUE_OPERATORS[condition.operator](
                            task.data[condition.input_value_key],
                            condition.target_value,
                        )
                        for condition in self._conditions
                    )
                    keep = all(condition_results) if self.condition_logic == "and" else any(condition_results)
                    if keep:
                        results.append(task)
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "input_count": len(tasks),
                "output_count": len(results),
                "filtered_count": len(tasks) - len(results),
            }
        )
        return results


def _row_names_file(row: dict[str, Any], key: str, wanted: set[str]) -> bool:
    """Whether a manifest row's audio path is one of ``wanted`` (absolute-path comparison)."""
    value = row.get(key)
    return isinstance(value, str) and os.path.abspath(os.path.expanduser(value)) in wanted


@dataclass
class ManifestReaderStage(AgentReady, ProcessingStage[FileGroupTask, AudioTask]):
    """Read JSONL manifest files from a FileGroupTask and emit one AudioTask per line.

    Uses line-by-line streaming via fsspec (no Pandas) to keep memory at ~1x file size.
    Supports local and cloud paths (S3, GCS).

    Args:
        include_files: Emit only rows whose audio path (under ``include_files_key``) is one of
            these files, comparing absolute paths. ``None`` -- the default -- reads every row.
            It restricts the same reader over the same manifest rather than pointing a delta
            run at a filtered copy, so the rows a partial run emits are the rows a full run
            would have emitted.
        include_files_key: Which row column holds that path.
    """

    name: str = "manifest_reader_stage"
    _: KW_ONLY
    include_files: list[str] | None = None
    include_files_key: str = "audio_filepath"
    # Declared statically as well as in describe(): a narrowable source has to be readable as
    # safe-to-narrow without constructing it, which is how the instance-free conformance sweep
    # sees it.
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(lifecycle_side_effects=True, per_row_independent=True)
    )
    # It points at the column holding a row's audio path, which is an existing role rather
    # than a new one -- a filter comparing something else would not be filtering by file.
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {"include_files_key": "audio_filepath"}

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        paths = task.data
        results: list[AudioTask] = []
        count = 0
        wanted = (
            None
            if self.include_files is None
            else {os.path.abspath(os.path.expanduser(p)) for p in self.include_files}
        )
        for manifest in paths:
            fs, resolved = url_to_fs(manifest)
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line.strip())
                        if wanted is not None and not _row_names_file(row, self.include_files_key, wanted):
                            continue
                        results.append(
                            AudioTask(
                                dataset_name=task.dataset_name,
                                data=row,
                                _metadata=task._metadata,
                                _stage_perf=list(task._stage_perf),
                            )
                        )
                        count += 1
            logger.info(f"ManifestReaderStage: loaded {count} entries from {manifest}")
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "manifests_read": len(paths),
                "entries_read": len(results),
            }
        )
        return results

    def num_workers(self) -> int | None:
        return 1

    def describe(self) -> StageContract:
        return StageContract(
            # The row is emitted verbatim, so the path column this reader was pointed at IS
            # the column downstream sees. Hard-coding ``audio_filepath`` claimed a key the
            # rows do not carry whenever ``include_files_key`` is configured: a manifest of
            # ``recording_path`` validated clean against a default consumer, which then found
            # no path at runtime.
            writes=IOSpec(data_keys=[self.include_files_key]),
            cardinality="1:N fan-out",
            gates=Gates(lifecycle_side_effects=True, per_row_independent=True),
        )


@dataclass
class ManifestReader(AgentReady, CompositeStage[EmptyTask, AudioTask]):
    """Composite stage for reading JSONL manifests.

    Decomposes into:
    1. FilePartitioningStage — discovers and partitions manifest files
    2. ManifestReaderStage — reads each partition line-by-line (no Pandas)

    Args:
        manifest_path: Path or list of paths to JSONL manifests (local or cloud).
        files_per_partition: Number of manifest files per partition. Defaults to 1.
        blocksize: Target size per partition (e.g., "100MB"). Ignored if files_per_partition is set.
        file_extensions: File extensions to filter. Defaults to [".jsonl", ".json"].
        storage_options: Storage options for cloud paths (S3, GCS credentials, endpoints).
        include_files: Read only the rows naming these audio files (see ``ManifestReaderStage``).
        include_files_key: Which row column holds the audio path.
    """

    manifest_path: str | list[str]
    name: str = "manifest_reader"
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".jsonl", ".json"])
    storage_options: dict[str, Any] | None = None
    _: KW_ONLY
    include_files: list[str] | None = None
    include_files_key: str = "audio_filepath"
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(gates=Gates(per_row_independent=True))
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {"include_files_key": "audio_filepath"}

    def __post_init__(self) -> None:
        super().__init__()
        if not self.manifest_path:
            msg = "manifest_path is required for ManifestReader"
            raise ValueError(msg)

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.manifest_path,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.storage_options,
            ),
            ManifestReaderStage(
                include_files=self.include_files,
                include_files_key=self.include_files_key,
            ),
        ]

    def get_description(self) -> str:
        parts = [f"Read JSONL manifests from {self.manifest_path}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        return ", ".join(parts)

    def describe(self) -> StageContract:
        return StageContract(
            cardinality="1:N fan-out",
            wrappable=False,
            gates=Gates(per_row_independent=True),
        )


class _AudioFolderTask(AudioTask):
    """Folder source identity independent of enumeration and the selected cohort."""

    def get_deterministic_id(self) -> str:
        return self._metadata["audio_folder_source_id"]


@dataclass
class CreateInitialManifestAudioFolderStage(AgentReady, ProcessingStage[EmptyTask, AudioTask]):
    """Create an initial manifest from any local folder of audio files.

    Recursively scans ``data_dir`` for audio files and emits one AudioTask per file with its
    path under ``audio_filepath_key`` (plus a filename-derived ``audio_item_id``). A generic,
    dataset-agnostic source: no download, no transcripts, and no dataset-specific filename
    parsing -- unlike ``CreateInitialManifest{ReadSpeech,Fleurs}Stage``. Use it to start a
    pipeline from a plain folder of WAV/FLAC/MP3/... when there is no JSONL manifest (use
    ``ManifestReader`` when a manifest already exists).

    Args:
        data_dir: Local folder to scan for audio files.
        extensions: Audio file extensions to include (case-insensitive).
        recursive: Recurse into subfolders (default True).
        max_samples: Maximum number of files to include (-1 for all).
        include_files: Process only these files (absolute paths), skipping the rest of the
            folder. ``None`` -- the default -- means the whole folder, exactly as before.
            Restricting the file list rather than swapping in a different source stage is what
            lets a delta run over new files produce rows identical to a full run's.
    """

    data_dir: str
    extensions: list[str] = field(default_factory=lambda: [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"])
    recursive: bool = True
    max_samples: int = -1
    include_files: list[str] | None = None
    audio_filepath_key: str = "audio_filepath"
    audio_item_id_key: str = "audio_item_id"
    name: str = "CreateInitialManifestAudioFolder"
    batch_size: int = 1
    # See ManifestReaderStage: the narrowing claim has to survive being read off the class.
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(gates=Gates(per_row_independent=False))

    def __post_init__(self) -> None:
        super().__init__()
        if not self.data_dir:
            msg = "data_dir is required for CreateInitialManifestAudioFolderStage"
            raise ValueError(msg)
        from nemo_curator.stages.audio._agent._residency import validate_audio_key_configuration

        validate_audio_key_configuration(
            self.name,
            input_keys={},
            output_keys={
                "audio_filepath_key": self.audio_filepath_key,
                "audio_item_id_key": self.audio_item_id_key,
            },
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.audio_item_id_key]

    def describe(self) -> StageContract:
        return StageContract(
            # No ``produces``: the audio already exists on disk, this stage only points at it
            # (unlike the dataset CreateInitialManifest*Stage sources, which download and write).
            writes=IOSpec(data_keys=[self.audio_filepath_key, self.audio_item_id_key]),
            cardinality="1:N fan-out",
            # A bounded sorted listing depends on the whole folder: adding an earlier path can
            # change which existing rows belong to the first N. Unbounded scans remain safely
            # narrowable one file at a time.
            gates=Gates(per_row_independent=self.max_samples is None or self.max_samples < 0),
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}

    def num_workers(self) -> int | None:
        return 1

    @staticmethod
    def _item_id(relative_path: str) -> str:
        """Flatten a relative stem without confusing separators with filename text."""
        stem, extension = os.path.splitext(relative_path)
        encoded_components: list[str] = []
        for component in stem.split(os.sep):
            encoded: list[str] = []
            last = len(component) - 1
            for index, char in enumerate(component):
                if char == "~":
                    # Reserve ``~`` for the underscore escape below.
                    encoded.append("~~")
                elif char == "_" and (
                    index in (0, last) or component[index - 1] == "_" or component[index + 1] == "_"
                ):
                    # Encoded components must neither contain ``__`` nor touch a
                    # separator with ``_``; otherwise two different component
                    # boundaries can flatten to the same string.
                    encoded.append("~u")
                else:
                    encoded.append(char)
            encoded_components.append("".join(encoded))
        return "__".join(encoded_components) + "~e" + extension[1:].replace("~", "~~")

    def _collect_audio_files(self) -> list[str]:
        exts = tuple((e if e.startswith(".") else f".{e}").lower() for e in self.extensions)
        if not os.path.isdir(self.data_dir):
            logger.error(f"[{self.name}] data_dir not found: {self.data_dir}")
            return []
        found: list[str] = []
        if self.recursive:
            for root, _dirs, files in os.walk(self.data_dir):
                found.extend(os.path.join(root, f) for f in files if f.lower().endswith(exts))
        else:
            found = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.lower().endswith(exts) and os.path.isfile(os.path.join(self.data_dir, f))
            ]
        if self.include_files is not None:
            wanted = {os.path.abspath(os.path.expanduser(p)) for p in self.include_files}
            found = [p for p in found if os.path.abspath(p) in wanted]
            missing = wanted - {os.path.abspath(p) for p in found}
            if missing:
                # Named rather than silently skipped: a caller that asked for specific files and
                # got fewer would otherwise read the short result as "those files held nothing".
                logger.warning(
                    f"[{self.name}] include_files named {len(missing)} file(s) not found under {self.data_dir}"
                )
        return sorted(found)

    def process(self, task: EmptyTask | None) -> list[AudioTask]:
        """Emit one AudioTask per audio file found under ``data_dir``."""
        paths = self._collect_audio_files()
        if self.max_samples is not None and self.max_samples >= 0:
            paths = paths[: self.max_samples]
        if not paths:
            logger.warning(f"[{self.name}] no audio files {self.extensions} under {self.data_dir}")
            return []
        tasks: list[AudioTask] = []
        for path in paths:
            abspath = os.path.abspath(path)
            # Relpath, not basename: ``recursive`` defaults True and speaker-per-folder is the
            # standard layout, so a basename id gives spk1/utt1.wav and spk2/utt1.wav the same
            # id -- and downstream that id becomes an output filename. A flat corpus is
            # unaffected unless its name needs escaping to remain distinct from a path separator.
            rel = os.path.relpath(abspath, os.path.abspath(self.data_dir))
            item_id = self._item_id(rel)
            tasks.append(
                _AudioFolderTask(
                    dataset_name="local-audio-folder",
                    data={self.audio_filepath_key: abspath, self.audio_item_id_key: item_id},
                    filepath_key=self.audio_filepath_key,
                    _metadata={
                        **({} if task is None else task._metadata),
                        "audio_folder_source_id": hashlib.sha256(abspath.encode("utf-8")).hexdigest(),
                    },
                    _stage_perf=[] if task is None else list(task._stage_perf),
                )
            )
        logger.info(f"[{self.name}] created {len(tasks)} AudioTask(s) from {self.data_dir}")
        return tasks


@dataclass
class ManifestWriterStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Append AudioTasks to a driver-prepared JSONL manifest.

    Audio-agent preparation truncates once before workers are serialized. Worker and
    node setup only verify the shared run reservation. Standalone setup() retains
    its historical truncate behavior when no driver reservation was prepared.
    """

    output_path: str
    name: str = "manifest_writer"

    # ``output_path`` is required, so static discovery cannot instantiate this
    # stage. Publish invariant sink behavior so an agent never mistakes it for
    # a pure pass-through before it has configuration values.
    # Agent runners stage terminal output and promote it only after verified success.
    AGENT_ATOMIC_OUTPUT_PARAMS: ClassVar[tuple[str, ...]] = ("output_path",)

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(
            writes_to_disk=True,
            output_path_params=["output_path"],
            lifecycle_side_effects=True,
            requires_serializable_input=True,
            per_row_independent=True,
        )
    )

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for ManifestWriterStage"
            raise ValueError(msg)

    def _resolve_output(self) -> None:
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)

    def _output_digest(self) -> str:
        digest = hashlib.sha256()
        with self._fs.open(self._path, "rb") as output:
            for chunk in iter(lambda: output.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _checkpoint_digest(self) -> str | None:
        from nemo_curator.utils.resumability_actor import METADATA_DIRNAME

        files = sorted((Path(self._checkpoint_path) / METADATA_DIRNAME).glob("*.mdb"))
        if not files:
            return None
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode())
            with path.open("rb") as state:
                for chunk in iter(lambda: state.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _bind_checkpoint(self) -> str:
        from nemo_curator.utils.resumability_actor import METADATA_DIRNAME

        marker = Path(self._checkpoint_path) / METADATA_DIRNAME / "audio_manifest_pipeline.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        binding = {"pipeline": self._pipeline_identity, "token": uuid.uuid4().hex}
        try:
            with marker.open("x", encoding="utf-8") as output:
                json.dump(binding, output)
                output.flush()
                os.fsync(output.fileno())
        except FileExistsError:
            with marker.open(encoding="utf-8") as source:
                binding = json.load(source)
        if binding.get("pipeline") != self._pipeline_identity or not binding.get("token"):
            msg = "ManifestWriterStage checkpoint belongs to a different pipeline"
            raise RuntimeError(msg)
        return binding["token"]

    def _resume_provenance(self) -> dict[str, str | None]:
        return {
            "checkpoint": self._checkpoint_path,
            "checkpoint_token": self._checkpoint_token,
            "pipeline": self._pipeline_identity,
            "state": self._checkpoint_digest(),
            "output": self._output_digest(),
        }

    def prepare_on_driver(
        self, *, checkpoint_path: str | Path | None = None, pipeline_identity: str | None = None
    ) -> None:
        """Reserve output; append only to a verified successful run of this pipeline."""
        self._resolve_output()
        token = uuid.uuid4().hex
        with self._fs.open(f"{self._path}._RUN", "x", encoding="utf-8") as owner:
            owner.write(token)
        self._run_token = token
        self._checkpoint_path = str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None
        self._pipeline_identity = pipeline_identity
        try:
            resume = False
            if checkpoint_path is not None:
                state = self._checkpoint_digest()
                exists = self._fs.exists(self._path)
                occupied = exists and self._fs.size(self._path) > 0
                if not pipeline_identity:
                    msg = "ManifestWriterStage needs pipeline_identity with checkpoint_path"
                    raise ValueError(msg)  # noqa: TRY301 - release the owned reservation
                marker = Path(self._checkpoint_path) / ".nemo_curator_metadata" / "audio_manifest_pipeline.json"
                self._checkpoint_token = None
                if marker.exists():
                    with marker.open(encoding="utf-8") as source:
                        self._checkpoint_token = json.load(source).get("token")
                if state is not None or occupied:
                    provenance = None
                    if self._fs.exists(f"{self._path}._RESUME"):
                        with self._fs.open(f"{self._path}._RESUME", "r", encoding="utf-8") as record:
                            provenance = json.load(record)
                    if not exists or not pipeline_identity or state is None or provenance != self._resume_provenance():
                        msg = "ManifestWriterStage cannot authenticate checkpoint, pipeline, and output for resume; use fresh output and checkpoint paths"
                        raise RuntimeError(msg)  # noqa: TRY301 - release the owned reservation on failure
                    resume = True
                self._checkpoint_token = self._bind_checkpoint()
            with self._fs.open(self._path, "a" if resume else "w", encoding="utf-8"):
                pass
            # A prior successful run must not authenticate partially rewritten output.
            if self._fs.exists(f"{self._path}._RESUME"):
                self._fs.rm(f"{self._path}._RESUME")
            self._write_run_owner()
        except Exception:
            self.abort_on_driver()
            raise

    def _output_identity(self) -> dict[str, str]:
        protocols = self._fs.protocol
        protocols = (protocols,) if isinstance(protocols, str) else protocols
        if {"file", "local"}.intersection(protocols):
            stat = os.stat(self._path)
            # Shared mounts can expose different device numbers on different hosts.
            return {"inode": str(stat.st_ino), "ctime_ns": str(stat.st_ctime_ns), "size": str(stat.st_size)}
        self._fs.invalidate_cache(self._path)
        info = self._fs.info(self._path)
        identity = {
            key: str(info[key])
            for key in ("size", "etag", "ETag", "version_id", "VersionId", "generation", "created")
            if key in info and info[key] not in (None, "")
        }
        if not set(identity) - {"size"}:
            msg = "ManifestWriterStage filesystem lacks artifact identity metadata for driver-prepared output"
            raise RuntimeError(msg)
        return identity

    def _read_run_owner(self) -> dict[str, Any]:
        try:
            with self._fs.open(f"{self._path}._RUN", "r", encoding="utf-8") as owner:
                text = owner.read()
        except FileNotFoundError:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            # Initial reservation contains only the token, before output preparation.
            return {"token": text}
        return value if isinstance(value, dict) else {}

    def _write_run_owner(self) -> None:
        payload = {"token": self._run_token, "artifact": self._output_identity()}
        with self._fs.open(f"{self._path}._RUN", "w", encoding="utf-8") as owner:
            json.dump(payload, owner)

    def _verify_run(self) -> None:
        try:
            owner = self._read_run_owner()
            if owner.get("token") == self._run_token and owner.get("artifact") == self._output_identity():
                return
        except FileNotFoundError:
            pass
        msg = f"ManifestWriterStage cannot see its driver-prepared shared output at {self.output_path!r}"
        raise RuntimeError(msg)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._resolve_output()
        if getattr(self, "_run_token", None) is not None:
            self._verify_run()
        else:
            if self._fs.exists(f"{self._path}._RUN"):
                msg = "ManifestWriterStage cannot truncate output owned by an active pipeline run"
                raise RuntimeError(msg)
            with self._fs.open(self._path, "w", encoding="utf-8"):
                pass
        logger.info(f"ManifestWriterStage: writing to {self.output_path}")

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        self._resolve_output()
        if getattr(self, "_run_token", None) is not None:
            self._verify_run()

    def finalize(self) -> None:
        if getattr(self, "_run_token", None) is not None:
            self._verify_run()
            if self._checkpoint_path is not None and self._pipeline_identity is not None:
                provenance = self._resume_provenance()
                if provenance["state"] is not None:
                    with self._fs.open(f"{self._path}._RESUME", "w", encoding="utf-8") as record:
                        json.dump(provenance, record, sort_keys=True)
            self.abort_on_driver()

    def reset_for_retry(self) -> None:
        """Release this agent run before the driver prepares its next attempt."""
        self.abort_on_driver()

    def abort_on_driver(self) -> None:
        """Release only this run's ownership; retain partial output on failure."""
        if getattr(self, "_run_token", None) is None:
            return
        self._resolve_output()
        owner_path = f"{self._path}._RUN"
        if self._fs.exists(owner_path):
            if self._read_run_owner().get("token") != self._run_token:
                return
            self._fs.rm(owner_path)
        self._run_token = None

    def write_jsonl_batch(self, records: str) -> None:
        """Append serialized JSONL records, verifying ownership once per batch.

        Callers provide newline-terminated records produced by their serializer.
        Empty batches leave both the output and its ownership marker untouched.
        """
        if not records:
            return
        if getattr(self, "_run_token", None) is not None:
            self._verify_run()
        with self._fs.open(self._path, "a", encoding="utf-8") as output:
            output.write(records)
        if getattr(self, "_run_token", None) is not None:
            self._write_run_owner()

    def process(self, task: AudioTask) -> AudioTask:
        self.write_jsonl_batch(json.dumps(task.data, ensure_ascii=False) + "\n")
        return AudioTask(
            dataset_name=task.dataset_name,
            data=task.data,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )

    def num_workers(self) -> int | None:
        return 1

    def describe(self) -> StageContract:
        return StageContract(
            gates=Gates(
                writes_to_disk=True,
                output_path_params=["output_path"],
                lifecycle_side_effects=True,
                # Serializes task.data as-is via json.dumps; a resident tensor
                # (e.g. a waveform) will crash it. Stop carrying the tensor
                # before this AudioTask sink, or convert to a DocumentBatch and
                # use DocumentBatchJsonlWriterStage instead.
                requires_serializable_input=True,
                # Appends each row as it arrives; a row's line is its own contents.
                per_row_independent=True,
            ),
        )


@dataclass
class ManifestCheckpointStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Persist a reusable, metadata-only AudioTask boundary as JSONL.

    This stage is an intermediate checkpoint, not a terminal user deliverable. It
    serializes complete ``task.data`` rows and passes each task downstream with
    its dataset name, metadata, and performance records preserved. Waveform
    tensors and other non-JSON values are illegal at this boundary.

    The checkpoint is local-only and single-worker. ``setup()`` exclusively
    reserves a new destination and refuses to overwrite any existing file.

    Args:
        output_path: Required local destination for checkpoint JSONL.
        retention_sec: Advisory retention in seconds. Defaults to 0, meaning
            user-managed with no automatic expiry; must be non-negative.
        owner: Ownership recorded with the checkpoint policy. ``"user"`` is the
            conservative default; ``"project"`` means the project operator owns
            retention. Neither value enables automatic deletion.
        planning_provenance: Internal marker for a reusable-pipeline candidate.
            Such recipes require exact-hash approval and authoritative smoke.
    """

    REUSABLE_PIPELINE_PROVENANCE: ClassVar[str] = "reusable_pipeline_v1"

    output_path: str
    retention_sec: int = 0
    owner: Literal["user", "project"] = "user"
    planning_provenance: Literal["reusable_pipeline_v1"] | None = None
    name: str = "manifest_checkpoint"

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(
            writes_to_disk=True,
            output_path_params=["output_path"],
            lifecycle_side_effects=True,
            requires_serializable_input=True,
            per_row_independent=True,
        ),
        description="Persist a complete metadata checkpoint without ending the pipeline",
    )

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for ManifestCheckpointStage"
            raise ValueError(msg)
        if urlsplit(self.output_path).scheme:
            msg = "ManifestCheckpointStage output_path must be a plain local path, not a URI"
            raise ValueError(msg)
        if isinstance(self.retention_sec, bool) or not isinstance(self.retention_sec, int) or self.retention_sec < 0:
            msg = "retention_sec must be a non-negative integer"
            raise ValueError(msg)
        if self.owner not in {"user", "project"}:
            msg = "owner must be 'user' or 'project'"
            raise ValueError(msg)
        if self.planning_provenance not in {None, self.REUSABLE_PIPELINE_PROVENANCE}:
            msg = f"planning_provenance must be None or {self.REUSABLE_PIPELINE_PROVENANCE!r}"
            raise ValueError(msg)
        self._reservation_owned = False
        self._reservation_identity: tuple[int, int, int] | None = None
        self._reservation_token = uuid.uuid4().hex
        self._checkpoint_rows_written = 0
        self._checkpoint_bytes_written = 0

    def _resolve_output(self) -> None:
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)

    def prepare_on_driver(self, **_run_context: object) -> None:
        self._reserve_checkpoint()
        self._driver_prepared = True

    def _verify_reservation(self) -> None:
        self._resolve_output()
        owner = self._read_retry_owner()
        if owner is not None and owner.get("token") == self._reservation_token and self._fs.exists(self._path):
            stat = os.stat(self._path)
            # Device numbers can differ across hosts mounting the same shared filesystem.
            if (stat.st_ino, stat.st_ctime_ns, stat.st_size) == (
                owner.get("st_ino"),
                owner.get("st_ctime_ns"),
                owner.get("st_size"),
            ):
                self._reservation_owned = True
                self._reservation_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
                return
        msg = f"ManifestCheckpointStage cannot verify its shared reservation at {self.output_path!r}"
        raise RuntimeError(msg)

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        if getattr(self, "_driver_prepared", False):
            self._verify_reservation()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if getattr(self, "_driver_prepared", False):
            self._verify_reservation()
        else:
            self._reserve_checkpoint()

    def finalize(self) -> None:
        self.release_retry_reservation()

    def _reserve_checkpoint(self) -> None:
        """Atomically reserve a new checkpoint without overwriting retained work."""
        self._resolve_output()
        if self._reservation_owned:
            msg = (
                "ManifestCheckpointStage setup was called again while this stage "
                "still owns an incomplete reservation; reset_for_retry() is required"
            )
            raise RuntimeError(msg)
        marker_path = f"{self._path}._COMPLETE"
        if self._fs.exists(marker_path):
            msg = (
                "ManifestCheckpointStage refuses to create a checkpoint beside an "
                f"existing completion marker at {self.output_path!r}"
            )
            raise FileExistsError(msg)
        owner_path = self._retry_owner_path()
        try:
            with self._fs.open(owner_path, "xb") as owner:
                owner.write(json.dumps({"token": self._reservation_token}).encode("utf-8"))
        except FileExistsError as exc:
            msg = (
                "ManifestCheckpointStage refuses to replace an existing retry "
                f"ownership record at {self.output_path!r}"
            )
            raise FileExistsError(msg) from exc
        try:
            with self._fs.open(self._path, "xb"):
                pass
        except FileExistsError as exc:
            self._remove_retry_owner_if_owned()
            msg = f"ManifestCheckpointStage refuses to overwrite an existing checkpoint at {self.output_path!r}"
            raise FileExistsError(msg) from exc
        except OSError:
            self._remove_retry_owner_if_owned()
            raise
        try:
            stat = os.stat(self._path)
        except OSError:
            # The exclusive create above belongs to this stage instance. If its
            # identity cannot be recorded, remove only that empty reservation
            # and fail rather than creating an unprovable retry owner.
            self._fs.rm(self._path)
            self._remove_retry_owner_if_owned()
            raise
        self._reservation_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
        self._reservation_owned = True
        self._checkpoint_rows_written = 0
        self._checkpoint_bytes_written = 0
        try:
            self._write_retry_owner(stat)
        except OSError:
            # The path is still the empty reservation whose identity was just
            # recorded. Remove it and its token rather than leave state that a
            # driver-side retry cannot prove.
            current = os.stat(self._path)
            current_identity = (
                current.st_dev,
                current.st_ino,
                current.st_ctime_ns,
            )
            if current_identity == self._reservation_identity and current.st_size == 0:
                self._fs.rm(self._path)
            self._remove_retry_owner_if_owned()
            self._reset_retry_state()
            raise
        logger.info(f"ManifestCheckpointStage: writing metadata to {self.output_path}")

    def reset_for_retry(self) -> None:
        """Remove only this instance's incomplete reservation before an automatic retry."""
        self._resolve_output()
        marker_path = f"{self._path}._COMPLETE"
        if self._fs.exists(marker_path):
            msg = (
                "ManifestCheckpointStage refuses retry reset because a completion "
                f"marker exists at {self.output_path!r}"
            )
            raise FileExistsError(msg)
        owner = self._read_retry_owner()
        if owner is None or owner.get("token") != self._reservation_token:
            if self._fs.exists(self._path) or owner is not None:
                msg = (
                    "ManifestCheckpointStage refuses retry reset of a checkpoint "
                    f"it did not reserve for this run at {self.output_path!r}"
                )
                raise FileExistsError(msg)
            self._reset_retry_state()
            return
        if self._fs.exists(self._path):
            try:
                stat = os.stat(self._path)
            except OSError as exc:
                msg = f"ManifestCheckpointStage could not verify its retry reservation at {self.output_path!r}"
                raise RuntimeError(msg) from exc
            # Match shared-file verification: device numbers are local to each host.
            identity = (stat.st_ino, stat.st_ctime_ns)
            recorded_identity = (
                owner.get("st_ino"),
                owner.get("st_ctime_ns"),
            )
            if identity != recorded_identity or stat.st_size != owner.get("st_size"):
                msg = (
                    "ManifestCheckpointStage refuses retry reset because the checkpoint "
                    f"at {self.output_path!r} is no longer its exact reservation"
                )
                raise FileExistsError(msg)
            self._fs.rm(self._path)
        self._remove_retry_owner_if_owned()
        self._reset_retry_state()

    def release_retry_reservation(self) -> None:
        """Publish completion only after verifying this run's durable reservation."""
        self._resolve_output()
        marker_path = f"{self._path}._COMPLETE"
        if self._fs.exists(marker_path):
            with self._fs.open(marker_path, encoding="utf-8") as marker:
                if json.load(marker).get("token") == self._reservation_token:
                    return
            msg = f"ManifestCheckpointStage refuses to replace completion marker at {self.output_path!r}"
            raise FileExistsError(msg)
        self._verify_reservation()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(self._path),
            prefix=os.path.basename(marker_path) + ".",
            delete=False,
        ) as marker:
            temporary_path = marker.name
            try:
                json.dump(self._read_retry_owner(), marker)
                marker.flush()
                os.fsync(marker.fileno())
                os.link(temporary_path, marker_path)
            finally:
                os.unlink(temporary_path)
        self._remove_retry_owner_if_owned()
        self._reservation_owned = False
        self._reservation_identity = None

    def _retry_owner_path(self) -> str:
        return f"{self._path}._RETRY_OWNER"

    def _read_retry_owner(self) -> dict[str, Any] | None:
        owner_path = self._retry_owner_path()
        if not self._fs.exists(owner_path):
            return None
        try:
            with self._fs.open(owner_path, "rb") as owner:
                value = json.loads(owner.read().decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_retry_owner(self, stat: os.stat_result) -> None:
        payload = {
            "token": self._reservation_token,
            "st_dev": stat.st_dev,
            "st_ino": stat.st_ino,
            "st_ctime_ns": stat.st_ctime_ns,
            "st_size": stat.st_size,
        }
        with self._fs.open(self._retry_owner_path(), "wb") as owner:
            owner.write(json.dumps(payload, sort_keys=True).encode("utf-8"))

    def _remove_retry_owner_if_owned(self) -> None:
        owner = self._read_retry_owner()
        if owner is None or owner.get("token") != self._reservation_token:
            return
        self._fs.rm(self._retry_owner_path())

    def _reset_retry_state(self) -> None:
        self._driver_prepared = False
        self._reservation_owned = False
        self._reservation_identity = None
        self._checkpoint_rows_written = 0
        self._checkpoint_bytes_written = 0
        self._custom_metrics = {}

    def process(self, task: AudioTask) -> AudioTask:
        if not self._reservation_owned:
            msg = "ManifestCheckpointStage cannot write without an owned setup reservation"
            raise RuntimeError(msg)
        self._verify_reservation()
        t0 = time.perf_counter()
        row = (json.dumps(task.data, ensure_ascii=False) + "\n").encode("utf-8")
        with self._fs.open(self._path, "ab") as f:
            f.write(row)
        self._checkpoint_rows_written += 1
        self._checkpoint_bytes_written += len(row)
        stat = os.stat(self._path)
        self._reservation_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
        self._write_retry_owner(stat)
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "checkpoint_rows_written": 1,
                "checkpoint_bytes_written": len(row),
            }
        )
        return AudioTask(
            dataset_name=task.dataset_name,
            data=task.data,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )

    def num_workers(self) -> int | None:
        return 1

    def describe(self) -> StageContract:
        return StageContract(
            gates=Gates(
                writes_to_disk=True,
                output_path_params=["output_path"],
                lifecycle_side_effects=True,
                requires_serializable_input=True,
                per_row_independent=True,
            ),
        )


def load_audio_file(audio_path: str, mono: bool = True) -> tuple[torch.Tensor, int]:
    """Load audio file and return waveform tensor (channels, samples) and sample rate."""
    data, sample_rate = soundfile.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data)
    waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform.T
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def ensure_waveform_2d(waveform: Any) -> torch.Tensor:  # noqa: ANN401
    """Ensure waveform is a torch.Tensor in 2D (channels, samples) format."""
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert multi-channel waveform to mono. Assumes 2D (channels, samples) input."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def resolve_waveform_from_item(
    item: dict[str, Any], task_id: str, mono: bool = True
) -> tuple[torch.Tensor, int] | None:
    """
    Resolve (waveform, sample_rate) from an item dict, loading from file if needed.

    Checks item['waveform'] + item['sample_rate'], falls back to loading from
    item['audio_filepath'], resolves missing sample_rate from file header.
    Updates item in-place when loading from file.
    Returns None if resolution fails.

    .. note::
       The canonical resolver is :func:`nemo_curator.stages.audio._agent._residency.resolve_audio`.
       This helper is retained for its unique behavior — reading ``sample_rate`` from the
       file header *without* reloading an already-present waveform, and writing the loaded
       waveform/sample_rate back into ``item`` — which ``resolve_audio`` does not replicate.
       Prefer ``resolve_audio`` in new code.
    """
    waveform = item.get("waveform")
    sample_rate = item.get("sample_rate")

    if waveform is None:
        audio_filepath = item.get("audio_filepath")
        if audio_filepath and os.path.exists(audio_filepath):
            try:
                waveform, sample_rate = load_audio_file(audio_filepath, mono=mono)
                item["waveform"] = waveform
                item["sample_rate"] = sample_rate
            except (OSError, RuntimeError, soundfile.SoundFileError) as e:
                logger.error(f"[{task_id}] Failed to load audio file: {e}")
                return None
        else:
            logger.warning(f"[{task_id}] No waveform or valid audio_filepath found")
            return None
    elif sample_rate is None:
        audio_filepath = item.get("audio_filepath")
        if audio_filepath and os.path.exists(audio_filepath):
            try:
                info = soundfile.info(audio_filepath)
                sample_rate = info.samplerate
                item["sample_rate"] = sample_rate
            except (OSError, RuntimeError, soundfile.SoundFileError) as e:
                logger.error(
                    f"[{task_id}] Waveform present but sample_rate missing "
                    f"and could not read from '{audio_filepath}': {e}"
                )
                return None
        else:
            logger.error(f"[{task_id}] Waveform present but 'sample_rate' missing and no audio_filepath available.")
            return None

    waveform = ensure_waveform_2d(waveform)
    if mono:
        waveform = ensure_mono(waveform)

    return waveform, sample_rate


def resolve_model_path(model_path: str, reference_file: str, module_subdir: str) -> str:
    """Resolve a relative model path using the reference file's directory and module subdirectory."""
    if os.path.isabs(model_path):
        return model_path
    current_dir = os.path.dirname(os.path.abspath(reference_file))
    module_dir = os.path.join(current_dir, module_subdir)
    for base in (module_dir, current_dir):
        resolved = os.path.join(base, model_path)
        if os.path.exists(resolved):
            return resolved
    return os.path.join(module_dir, model_path)
