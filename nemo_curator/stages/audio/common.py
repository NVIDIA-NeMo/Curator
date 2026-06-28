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
# ruff: noqa: ANN401, BLE001, C901, N806, PERF403, PLR0911, PLR0912, PLR0913, S110, TRY300, TRY301

import contextlib
import copy
import json
import math
import os
import pickle
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from numbers import Real
from operator import eq, ge, gt, le, lt, ne
from typing import Any

import soundfile
import torch
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.io.manifest_writer_utils import AudioManifestWriterMetrics, manifest_lines
from nemo_curator.stages.audio.model_input_segmentation import (
    duration_to_num_samples,
    plan_audio_segments,
    resolve_max_model_input_duration,
)
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask
from nemo_curator.tasks.task_terminals import (
    TERMINAL_COUNT_KEY,
    TERMINAL_DROPPED_BY_STAGE_KEY,
    TERMINAL_DROPPED_KEY,
    TERMINAL_GROUP_ID_KEY,
    TERMINAL_INDEX_KEY,
    TERMINAL_SOURCE_INDEX_KEY,
)


def _single_global_owner_stage(value: Any) -> str:
    """Return the single stage selector that owns global duration planning."""
    if value is None:
        msg = "ManifestReader(enable_global_bucketing=True) requires owner_stage"
        raise ValueError(msg)
    if isinstance(value, str):
        values = [value.strip()]
    else:
        try:
            values = [str(item).strip() for item in value]
        except TypeError as exc:
            msg = f"owner_stage must be a string stage selector, got {type(value).__name__}"
            raise TypeError(msg) from exc
    values = [item for item in values if item]
    if len(values) != 1:
        msg = f"owner_stage must contain exactly one stage selector for global bucketing, got {values}"
        raise ValueError(msg)
    return values[0]


def _as_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    get = getattr(config, "get", None)
    if callable(get):
        return get(key, default)
    return default


def _config_section(config: Any, key: str) -> dict[str, Any]:
    value = _as_container(_config_get(config, key, {}))
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f"{key} must be a mapping when configured, got {type(value).__name__}"
        raise TypeError(msg)
    return dict(value)


def _normalise_string_list(value: Any, *, key: str) -> list[str]:
    value = _as_container(value)
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    result = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    if not result:
        msg = f"{key} must contain at least one non-empty value"
        raise ValueError(msg)
    return result


def _normalise_optional_string_list(value: Any, *, key: str) -> list[str]:
    if value is None:
        return []
    return _normalise_string_list(value, key=key)


def _first_stage_attr(stages: list[ProcessingStage], attr: str, default: Any) -> Any:
    for stage in stages:
        value = getattr(stage, attr, None)
        if value not in (None, ""):
            return value
    return default


def _infer_global_text_keys(consumers: list[ProcessingStage]) -> list[str]:
    text_keys: list[str] = []
    for stage in consumers:
        for attr in ("pred_text_key", "disfluency_text_key"):
            value = getattr(stage, attr, None)
            if value and value not in text_keys:
                text_keys.append(str(value))
    return text_keys


def _assembler_skip_keys(assembler_cfg: dict[str, Any], consumers: list[ProcessingStage]) -> list[str]:
    configured = assembler_cfg.get("skip_me_keys")
    keys = (
        _normalise_string_list(configured, key="global_segment_assembler.skip_me_keys")
        if configured is not None
        else []
    )
    single_key = assembler_cfg.get("skip_me_key") or _first_stage_attr(consumers, "skip_me_key", "_skip_me")
    keys.append(str(single_key))
    for consumer in consumers:
        value = getattr(consumer, "skip_me_key", None)
        if value:
            keys.append(str(value))
    keys.append(TERMINAL_DROPPED_KEY)
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        text = str(key).strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def get_audio_duration(audio_filepath: str) -> float:
    """Get the duration of the audio file in seconds."""
    try:
        info = soundfile.info(audio_filepath)
        return info.frames / info.samplerate
    except Exception as e:
        logger.warning(f"Failed to get duration for audio file {audio_filepath}: {e}")
        return -1.0


@dataclass
class GetAudioDurationStage(ProcessingStage[AudioTask, AudioTask]):
    """Compute audio duration from the file at *audio_filepath_key* and
    store the result under *duration_key*.

    Args:
        audio_filepath_key: Key to get path to wav file.
        duration_key: Key to put audio duration.
    """

    name: str = "GetAudioDurationStage"
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        import soundfile

        self._soundfile = soundfile

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.duration_key]

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        audio_filepath = task.data[self.audio_filepath_key]
        duration = get_audio_duration(audio_filepath)
        task.data[self.duration_key] = duration
        self._log_metrics({"process_time": time.perf_counter() - t0, "duration": max(duration, 0.0)})
        return task


class PreserveByValueStage(ProcessingStage[AudioTask, AudioTask]):
    """Filter entries by comparing *input_value_key* against *target_value*.

    Returns ``None`` from ``process()`` to drop entries that fail the
    comparison, matching the text-modality filter convention.

    Args:
        input_value_key: The field in the dataset entries to evaluate.
        target_value: The value to compare with.
        operator: Comparison operator (lt, le, eq, ne, ge, gt).
    """

    name: str = "PreserveByValueStage"

    def __init__(
        self,
        input_value_key: str,
        target_value: int | str,
        operator: str = "eq",
    ):
        self.input_value_key = input_value_key
        self.target_value = target_value
        ops = {"lt": lt, "le": le, "eq": eq, "ne": ne, "ge": ge, "gt": gt}
        if operator not in ops:
            msg = f"Operator must be one of: {', '.join(ops)}"
            raise ValueError(msg)
        self.operator = ops[operator]

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def process(self, task: AudioTask) -> AudioTask | None:
        msg = "PreserveByValueStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        t0 = time.perf_counter()
        results = []
        for task in tasks:
            if not self.validate_input(task):
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


@dataclass
class ManifestReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read JSONL manifest files from a FileGroupTask and emit one AudioTask per line.

    Uses line-by-line streaming via fsspec (no Pandas) to keep memory at ~1x file size.
    Supports local and cloud paths (S3, GCS).
    """

    name: str = "manifest_reader_stage"
    storage_options: dict[str, Any] | None = None

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        paths = task.data
        results: list[AudioTask] = []
        count = 0
        for manifest in paths:
            fs, resolved = url_to_fs(manifest, **(self.storage_options or {}))
            manifest_count = 0
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        audio_task = AudioTask(
                            dataset_name=task.dataset_name,
                            data=json.loads(line.strip()),
                            _metadata=task._metadata,
                            _stage_perf=list(task._stage_perf),
                        )
                        results.append(audio_task)
                        count += 1
                        manifest_count += 1
            logger.info(f"ManifestReaderStage: loaded {manifest_count} entries from {manifest}")
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "manifests_read": len(paths),
                "entries_read": len(results),
            }
        )
        return results

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}

    def num_workers(self) -> int | None:
        return 1


@dataclass
class ManifestReader(CompositeStage[EmptyTask, AudioTask]):
    """Composite stage for reading JSONL manifests.

    By default, decomposes into:
    1. FilePartitioningStage — discovers and partitions manifest files
    2. ManifestReaderStage — reads each partition line-by-line (no Pandas)

    When ``enable_global_bucketing`` is true, the same reader entry point uses a
    metadata-only full-manifest planner and emits reordered ordinary
    ``AudioTask`` rows. Bucket-off mode therefore keeps the original streaming
    reader behavior.
    Global planning has exactly one owner stage: ``owner_stage`` names the
    downstream consumer whose model-input duration ceiling and bucket policy
    define the emitted segment rows. The owner must be the payload consumer with
    the largest ``max_inference_duration_s`` because the planner emits one
    full-manifest segment plan. Other payload consumers can share those rows,
    but they do not receive an independent global plan.

    Args:
        manifest_path: Path or list of paths to JSONL manifests (local or cloud).
        files_per_partition: Number of manifest files per partition. Defaults to 1.
        blocksize: Target size per partition (e.g., "100MB"). Ignored if files_per_partition is set.
        file_extensions: File extensions to filter. Defaults to [".jsonl", ".json"].
        storage_options: Storage options for cloud paths (S3, GCS credentials, endpoints).
    """

    manifest_path: str | list[str]
    name: str = "manifest_reader"
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".jsonl", ".json"])
    storage_options: dict[str, Any] | None = None
    enable_global_bucketing: bool = False
    owner_stage: str | None = None
    parent_coalescing: bool = True
    duration_key: str = "duration"
    fallback_duration_keys: list[str] = field(default_factory=lambda: ["actual_duration", "duration_sec"])
    sample_rate_key: str = "sample_rate"
    num_samples_key: str = "num_samples"
    target_sample_rate: int = 16000
    target_nchannels: int = 1
    waveform_dtype_bytes: int = 4
    max_inference_duration_s: float = 120.0
    buckets_sec: list[float] = field(default_factory=lambda: [0.0, 30.0, 60.0, 120.0])
    max_items_per_batch_by_bucket: list[int] | None = None
    max_audio_sec_per_batch: float | None = None
    target_ready_batches_per_bucket: int = 4
    annotate_plan_metadata: bool = True
    annotate_segment_plan: bool = False
    segment_input_keys: list[str] = field(default_factory=lambda: ["audio_filepath"])
    run_id: str | None = None
    parent_store_actor_name_prefix: str = "curator_global_segment_parent_store"

    def __post_init__(self) -> None:
        super().__init__()
        if not self.manifest_path:
            msg = "manifest_path is required for ManifestReader"
            raise ValueError(msg)
        if not isinstance(self.enable_global_bucketing, bool):
            msg = f"enable_global_bucketing must be a bool, got {type(self.enable_global_bucketing).__name__}"
            raise TypeError(msg)
        if self.enable_global_bucketing:
            self.owner_stage = _single_global_owner_stage(self.owner_stage)

    def decompose(self) -> list[ProcessingStage]:
        if self.enable_global_bucketing:
            return [
                _ManifestReaderGlobalBucketingStage(
                    manifest_path=self.manifest_path,
                    storage_options=self.storage_options,
                    owner_stage=self.owner_stage,
                    parent_coalescing=self.parent_coalescing,
                    duration_key=self.duration_key,
                    fallback_duration_keys=self.fallback_duration_keys,
                    sample_rate_key=self.sample_rate_key,
                    num_samples_key=self.num_samples_key,
                    target_sample_rate=self.target_sample_rate,
                    target_nchannels=self.target_nchannels,
                    waveform_dtype_bytes=self.waveform_dtype_bytes,
                    max_inference_duration_s=self.max_inference_duration_s,
                    buckets_sec=self.buckets_sec,
                    max_items_per_batch_by_bucket=self.max_items_per_batch_by_bucket,
                    max_audio_sec_per_batch=self.max_audio_sec_per_batch,
                    target_ready_batches_per_bucket=self.target_ready_batches_per_bucket,
                    annotate_plan_metadata=self.annotate_plan_metadata,
                    annotate_segment_plan=self.annotate_segment_plan,
                    segment_input_keys=self.segment_input_keys,
                    run_id=self.run_id,
                    parent_store_actor_name_prefix=self.parent_store_actor_name_prefix,
                )
            ]
        return [
            FilePartitioningStage(
                file_paths=self.manifest_path,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.storage_options,
            ),
            ManifestReaderStage(storage_options=self.storage_options),
        ]

    def build_payload_materialize_stage(
        self,
        *,
        payload_spec: Any,
        payload_config: dict[str, Any],
        pipeline_config: Any,
        run_id: str,
    ) -> ProcessingStage:
        """Build the audio payload materializer for the generic lifecycle planner.

        ``nemo_curator.pipeline.payload_lifecycle`` owns graph insertion order.
        The reader owns modality-specific materialization, so central pipeline
        code does not need to import audio reader/materializer internals.
        """

        from nemo_curator.stages.payload_lifecycle import AudioPayloadMaterializeStage

        return AudioPayloadMaterializeStage(
            name=payload_spec.materialize_stage_name,
            target_sample_rate=int(payload_config.get("target_sample_rate", 16000)),
            target_nchannels=int(payload_config.get("target_nchannels", 1)),
            audio_filepath_key=payload_spec.source_key,
            duration_key=payload_spec.duration_key,
            segment_start_key=str(payload_config.get("segment_start_key", "segment_start_s")),
            segment_duration_key=str(payload_config.get("segment_duration_key", "segment_duration_s")),
            waveform_key=payload_spec.waveform_key,
            waveform_ref_key=payload_spec.ref_key,
            sample_rate_key=payload_spec.sample_rate_key,
            num_samples_key=payload_spec.num_samples_key,
            skip_on_read_error=bool(
                payload_config.get(
                    "skip_on_read_error",
                    _config_get(pipeline_config, "audio_reader_skip_on_read_error", False),
                )
            ),
            node_memory_fraction=float(payload_config.get("node_memory_fraction", 0.80)),
            max_node_payload_bytes=payload_config.get("max_node_payload_bytes"),
            max_cluster_payload_bytes=payload_config.get("max_cluster_payload_bytes"),
            lease_ttl_s=float(payload_config.get("lease_ttl_s", 3600)),
            materialized_lease_ttl_s=float(payload_config.get("materialized_lease_ttl_s", 4 * 60 * 60)),
            admission_actor_name=str(payload_config.get("admission_actor_name", "curator_payload_admission")),
            admission_poll_interval_s=float(payload_config.get("admission_poll_interval_s", 0.25)),
            admission_wait_timeout_s=float(payload_config.get("admission_wait_timeout_s", 4 * 60 * 60)),
            run_id=run_id,
        )

    def build_payload_lifecycle_source_stage(self) -> ProcessingStage:
        """Return the physical metadata planner used by global payload runs."""
        if not self.enable_global_bucketing:
            return self
        stages = self.decompose_and_apply_with()
        if len(stages) != 1:
            msg = f"Global ManifestReader must decompose to one planner stage, got {len(stages)}"
            raise RuntimeError(msg)
        return stages[0]

    def build_payload_lifecycle_post_release_stage(
        self,
        *,
        pipeline_config: Any,
        consumers: list[ProcessingStage],
        primary_payload_spec: Any,
        run_id: str,
    ) -> ProcessingStage | None:
        """Build the optional global segment assembler for audio global bucketing."""

        if not self.enable_global_bucketing:
            return None

        assembler_cfg = _config_section(pipeline_config, "global_segment_assembler")
        scheduler_cfg = _config_section(pipeline_config, "global_audio_scheduler")
        text_keys = _as_container(assembler_cfg.get("text_keys_to_join"))
        if text_keys is None:
            text_keys = _infer_global_text_keys(consumers)
        else:
            text_keys = _normalise_string_list(text_keys, key="global_segment_assembler.text_keys_to_join")
        field_merge_strategies = _as_container(assembler_cfg.get("field_merge_strategies") or {})
        if not isinstance(field_merge_strategies, dict):
            msg = "global_segment_assembler.field_merge_strategies must be a mapping"
            raise TypeError(msg)

        payload_cfg = _config_section(pipeline_config, "payload_lifecycle")
        return GlobalSegmentAssemblerStage(
            name=str(assembler_cfg.get("name", "global_segment_assembler")),
            text_keys_to_join=text_keys,
            field_merge_strategies={str(key): str(value) for key, value in field_merge_strategies.items()},
            skip_me_key=str(
                assembler_cfg.get("skip_me_key") or _first_stage_attr(consumers, "skip_me_key", "_skip_me")
            ),
            skip_me_keys=_assembler_skip_keys(assembler_cfg, consumers),
            waveform_key=primary_payload_spec.waveform_key,
            waveform_ref_key=primary_payload_spec.ref_key,
            duration_key=primary_payload_spec.duration_key,
            num_samples_key=primary_payload_spec.num_samples_key,
            segment_start_key=str(
                assembler_cfg.get("segment_start_key") or payload_cfg.get("segment_start_key", "segment_start_s")
            ),
            segment_duration_key=str(
                assembler_cfg.get("segment_duration_key")
                or payload_cfg.get("segment_duration_key", "segment_duration_s")
            ),
            max_ready_parents_in_memory=int(assembler_cfg.get("max_ready_parents_in_memory", 4096)),
            spill_dir=assembler_cfg.get("spill_dir"),
            actor_name_prefix=str(assembler_cfg.get("actor_name_prefix", "curator_global_segment_assembler")),
            parent_store_actor_name_prefix=str(
                assembler_cfg.get(
                    "parent_store_actor_name_prefix",
                    scheduler_cfg.get("parent_store_actor_name_prefix", self.parent_store_actor_name_prefix),
                )
            ),
            run_id=run_id,
            strict_unmerged_segment_fields=bool(assembler_cfg.get("strict_unmerged_segment_fields", True)),
            unmerged_segment_field_allowlist=_normalise_optional_string_list(
                assembler_cfg.get("unmerged_segment_field_allowlist"),
                key="global_segment_assembler.unmerged_segment_field_allowlist",
            ),
            overwrite=_normalise_optional_string_list(
                assembler_cfg.get("overwrite"),
                key="global_segment_assembler.overwrite",
            ),
        )

    def get_description(self) -> str:
        parts = [f"Read JSONL manifests from {self.manifest_path}"]
        if self.enable_global_bucketing:
            parts.append("with metadata-global duration bucketing")
        elif self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        return ", ".join(parts)


@dataclass(frozen=True)
class _GlobalSegmentPlan:
    parent_index: int
    segment_idx: int
    start_sample: int
    stop_sample: int
    duration_s: float
    bucket_id: int


@dataclass(frozen=True)
class _ManifestPlanRecord:
    source_index: int
    manifest_index: int
    row_index: int
    data: dict[str, Any]
    duration_s: float
    estimated_waveform_bytes: int
    segments: tuple[_GlobalSegmentPlan, ...]
    total_segment_cost_s: float
    dominant_bucket: int
    dominant_bucket_cost_s: float

    @property
    def segment_count(self) -> int:
        return len(self.segments)


@dataclass(frozen=True)
class _PlannedManifestRecord:
    record: _ManifestPlanRecord
    plan_order: int


@dataclass(frozen=True)
class _PlannedManifestSegment:
    record: _ManifestPlanRecord
    segment: _GlobalSegmentPlan
    plan_order: int


def _safe_global_segment_actor_suffix(value: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return suffix or "default"


def _current_ray_namespace() -> str | None:
    try:
        import ray

        ctx = ray.get_runtime_context()
        namespace = getattr(ctx, "namespace", None)
        if callable(namespace):
            namespace = namespace()
        if not namespace:
            get_namespace = getattr(ctx, "get_namespace", None)
            if callable(get_namespace):
                namespace = get_namespace()
        if namespace:
            return str(namespace)
    except Exception:
        pass
    return None


def _kill_named_ray_actor(name: str, namespace: str | None = None) -> bool:
    try:
        import ray

        actor = ray.get_actor(name, namespace=namespace)
        ray.kill(actor, no_restart=True)
        return True
    except ValueError:
        return False
    except Exception as exc:
        logger.warning(f"Failed to kill global segment actor {name!r}: {exc}")
    return False


class _GlobalSegmentParentDataStore:
    """Ray actor state for original parent rows used by global segment assembly."""

    def __init__(self) -> None:
        self._parents: dict[str, dict[str, Any]] = {}

    def put_many(self, items: dict[str, dict[str, Any]]) -> int:
        for parent_id, data in items.items():
            self._parents[str(parent_id)] = copy.deepcopy(data)
        return len(items)

    def get_parent(self, parent_id: str) -> dict[str, Any] | None:
        data = self._parents.get(str(parent_id))
        return copy.deepcopy(data) if data is not None else None

    def delete_many(self, parent_ids: list[str]) -> int:
        deleted = 0
        for parent_id in parent_ids:
            if self._parents.pop(str(parent_id), None) is not None:
                deleted += 1
        return deleted

    def clear(self) -> None:
        self._parents.clear()


class _GlobalSegmentAssemblyState:
    """Ray actor state for generic global-segment parent assembly."""

    _SEGMENT_INPUT_KEYS_FIELD = "_curator_segment_input_keys"

    def __init__(
        self,
        *,
        text_keys_to_join: list[str],
        field_merge_strategies: dict[str, str] | None = None,
        skip_me_key: str,
        skip_me_keys: list[str] | None = None,
        waveform_key: str,
        waveform_ref_key: str,
        duration_key: str,
        num_samples_key: str,
        segment_start_key: str,
        segment_duration_key: str,
        max_ready_parents_in_memory: int = 4096,
        spill_dir: str | None = None,
        strict_unmerged_segment_fields: bool = True,
        unmerged_segment_field_allowlist: list[str] | None = None,
        overwrite_keys: list[str] | None = None,
        require_parent_data: bool = False,
    ) -> None:
        if max_ready_parents_in_memory < 0:
            msg = f"max_ready_parents_in_memory must be >= 0, got {max_ready_parents_in_memory}"
            raise ValueError(msg)
        self.text_keys_to_join = [str(key) for key in text_keys_to_join if str(key).strip()]
        self.field_merge_strategies = {
            str(key): str(value) for key, value in (field_merge_strategies or {}).items() if str(key).strip()
        }
        self.overwrite_keys = {str(key).strip() for key in (overwrite_keys or []) if str(key).strip()}
        for key in self.overwrite_keys:
            existing = self.field_merge_strategies.get(key)
            if existing not in (None, "overwrite"):
                msg = f"Field {key!r} cannot be configured for both overwrite and {existing!r} merge"
                raise ValueError(msg)
            self.field_merge_strategies[key] = "overwrite"
        for key in self.text_keys_to_join:
            self.field_merge_strategies.setdefault(key, "join_text")
        self.skip_me_key = skip_me_key
        self.skip_me_keys = self._normalise_skip_keys(skip_me_key, skip_me_keys)
        self.waveform_key = waveform_key
        self.waveform_ref_key = waveform_ref_key
        self.duration_key = duration_key
        self.num_samples_key = num_samples_key
        self.segment_start_key = segment_start_key
        self.segment_duration_key = segment_duration_key
        self.max_ready_parents_in_memory = int(max_ready_parents_in_memory)
        self.spill_dir_root = spill_dir
        self.strict_unmerged_segment_fields = bool(strict_unmerged_segment_fields)
        self.unmerged_segment_field_allowlist = {
            str(key).strip() for key in (unmerged_segment_field_allowlist or []) if str(key).strip()
        }
        self.require_parent_data = bool(require_parent_data)
        self._spill_dir: str | None = None
        self._parents: dict[str, dict[str, Any]] = {}
        self._ready_by_source_index: dict[int, dict[str, Any]] = {}
        self._ready_spill_paths: dict[int, str] = {}
        self._next_source_index = 0
        self._segments_seen = 0
        self._tombstone_segments_seen = 0
        self._parents_assembled = 0
        self._spilled_parents = 0
        self._spill_bytes = 0
        self._max_buffered_parents = 0
        self._max_ready_gap = 0

    def add_segment(
        self,
        *,
        parent_id: str,
        segment_idx: int,
        segment_count: int,
        data: dict[str, Any],
        metadata: dict[str, Any],
        stage_perf: list[Any],
        parent_data: dict[str, Any] | None = None,
        include_completion: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str | None]:
        if self.require_parent_data and parent_data is None:
            msg = (
                f"Global segment assembly requires original parent data for parent {parent_id!r}; "
                "refusing to assemble from segment rows because segment rows carry only configured "
                "segment input keys."
            )
            raise RuntimeError(msg)
        if segment_count <= 0:
            msg = f"segment_count must be > 0 for parent {parent_id!r}"
            raise ValueError(msg)
        if segment_idx < 0 or segment_idx >= segment_count:
            msg = (
                f"segment_idx {segment_idx} is outside expected range 0..{segment_count - 1} for parent {parent_id!r}"
            )
            raise ValueError(msg)

        self._segments_seen += 1
        skipped_segment = self._is_skipped_segment(data)
        if skipped_segment:
            self._tombstone_segments_seen += 1

        entry = self._parents.setdefault(
            parent_id,
            {
                "segment_count": segment_count,
                "source_index": self._parent_source_index(parent_id, data),
                "base_data": self._parent_base_data(data, parent_data=parent_data),
                "parent_keys": set((parent_data or data).keys()),
                "metadata": dict(metadata),
                "segments": {},
                "stage_perf": [],
            },
        )
        if int(entry["segment_count"]) != segment_count:
            msg = (
                f"Segments for parent {parent_id!r} disagree on segment_count: "
                f"{entry['segment_count']} vs {segment_count}"
            )
            raise ValueError(msg)

        segments: dict[int, dict[str, Any]] = entry["segments"]
        if segment_idx in segments:
            msg = f"Duplicate segment {segment_idx} for parent {parent_id!r}"
            raise ValueError(msg)
        self._validate_segment_output_fields(parent_id, data, set(entry["parent_keys"]), parent_data=parent_data)
        segments[segment_idx] = {
            "values": {key: copy.deepcopy(data[key]) for key in self.field_merge_strategies if key in data},
            "passthrough_values": self._passthrough_segment_values(
                data,
                set(entry["parent_keys"]),
                parent_data=parent_data,
            ),
            "skipped": skipped_segment,
            "drop_stage_ids": self._segment_drop_stage_ids(data),
        }
        entry["stage_perf"].extend(stage_perf)

        if len(segments) != segment_count:
            return ([], None) if include_completion else []

        assembled = self._assemble_parent(parent_id, entry)
        self._parents_assembled += 1
        self._parents.pop(parent_id, None)
        source_index = int(assembled["source_index"])
        if source_index in self._ready_by_source_index or source_index in self._ready_spill_paths:
            msg = f"Duplicate assembled parent source index {source_index} for parent {parent_id!r}"
            raise ValueError(msg)
        self._store_ready_parent(source_index, assembled)
        ready = self._drain_ready_in_input_order()
        return (ready, parent_id) if include_completion else ready

    def _store_ready_parent(self, source_index: int, assembled: dict[str, Any]) -> None:
        if len(self._ready_by_source_index) < self.max_ready_parents_in_memory:
            self._ready_by_source_index[source_index] = assembled
            self._update_buffer_metrics()
            return
        spill_path = self._spill_ready_parent(source_index, assembled)
        self._ready_spill_paths[source_index] = spill_path
        self._update_buffer_metrics()

    def _spill_ready_parent(self, source_index: int, assembled: dict[str, Any]) -> str:
        if self._spill_dir is None:
            if self.spill_dir_root:
                os.makedirs(self.spill_dir_root, exist_ok=True)
            self._spill_dir = tempfile.mkdtemp(
                prefix="curator_global_segment_assembler_",
                dir=self.spill_dir_root,
            )
        path = os.path.join(self._spill_dir, f"source_{source_index}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(assembled, fh, protocol=pickle.HIGHEST_PROTOCOL)
        self._spilled_parents += 1
        with contextlib.suppress(OSError):
            self._spill_bytes += int(os.path.getsize(path))
        return path

    def _drain_ready_in_input_order(self) -> list[dict[str, Any]]:
        ready: list[dict[str, Any]] = []
        while (
            self._next_source_index in self._ready_by_source_index
            or self._next_source_index in self._ready_spill_paths
        ):
            if self._next_source_index in self._ready_by_source_index:
                ready.append(self._ready_by_source_index.pop(self._next_source_index))
            else:
                ready.append(self._load_spilled_ready_parent(self._next_source_index))
            self._next_source_index += 1
        self._update_buffer_metrics()
        return ready

    def _update_buffer_metrics(self) -> None:
        buffered = len(self._parents) + len(self._ready_by_source_index) + len(self._ready_spill_paths)
        self._max_buffered_parents = max(self._max_buffered_parents, buffered)
        ready_indices = [*self._ready_by_source_index.keys(), *self._ready_spill_paths.keys()]
        if ready_indices:
            self._max_ready_gap = max(self._max_ready_gap, max(ready_indices) - self._next_source_index)

    def snapshot_metrics(self) -> dict[str, float]:
        return {
            "global_assembler_segments_seen": float(self._segments_seen),
            "global_assembler_tombstone_segments": float(self._tombstone_segments_seen),
            "global_assembler_parents_assembled": float(self._parents_assembled),
            "global_assembler_spilled_parents": float(self._spilled_parents),
            "global_assembler_spill_bytes": float(self._spill_bytes),
            "global_assembler_max_buffered_parents": float(self._max_buffered_parents),
            "global_assembler_max_ready_gap": float(self._max_ready_gap),
        }

    def _load_spilled_ready_parent(self, source_index: int) -> dict[str, Any]:
        path = self._ready_spill_paths.pop(source_index)
        with open(path, "rb") as fh:
            assembled = pickle.load(fh)  # noqa: S301 - trusted local actor spill file
        with contextlib.suppress(OSError):
            os.unlink(path)
        return assembled

    def _parent_base_data(self, data: dict[str, Any], *, parent_data: dict[str, Any] | None = None) -> dict[str, Any]:
        base = copy.deepcopy(parent_data) if parent_data is not None else dict(data)
        parent_duration = base.get("_curator_segment_parent_duration_s")
        parent_num_samples = base.get("_curator_segment_parent_num_samples")
        if parent_data is not None:
            parent_duration = data.get("_curator_segment_parent_duration_s", parent_duration)
            parent_num_samples = data.get("_curator_segment_parent_num_samples", parent_num_samples)
        for key in list(base):
            if key.startswith(("_curator_segment_", "_curator_terminal_", "_curator_payload_")):
                base.pop(key, None)
        for key in (
            self.segment_start_key,
            self.segment_duration_key,
            self.waveform_key,
            self.waveform_ref_key,
            "_curator_payload_estimated_bytes",
            "_curator_payload_bytes",
            *self.field_merge_strategies.keys(),
        ):
            base.pop(key, None)
        if parent_duration is not None:
            base[self.duration_key] = parent_duration
        if parent_num_samples is not None:
            base[self.num_samples_key] = parent_num_samples
        return base

    def _validate_segment_output_fields(
        self,
        parent_id: str,
        data: dict[str, Any],
        parent_keys: set[str],
        *,
        parent_data: dict[str, Any] | None,
    ) -> None:
        if not self.strict_unmerged_segment_fields:
            return
        unknown = sorted(
            key
            for key in data
            if self._is_forbidden_parent_collision(key, data=data, parent_keys=parent_keys, parent_data=parent_data)
        )
        if unknown:
            msg = (
                "Global segment assembly received output fields that collide with original parent fields "
                f"but were not copied to segment rows for parent {parent_id!r}: {unknown}. Add the field to "
                "global_audio_scheduler.segment_input_keys when the segment consumer is allowed to modify it, "
                "or add global_segment_assembler.field_merge_strategies/global_segment_assembler.overwrite "
                "when the segment output should replace or merge the parent value."
            )
            raise ValueError(msg)

    def _is_forbidden_parent_collision(
        self,
        key: str,
        *,
        data: dict[str, Any],
        parent_keys: set[str],
        parent_data: dict[str, Any] | None,
    ) -> bool:
        if key in self.field_merge_strategies:
            return False
        if key in self.skip_me_keys:
            return False
        if key in self.unmerged_segment_field_allowlist:
            return False
        if self._is_segment_helper_field(key):
            return False
        if parent_data is None or key not in parent_keys:
            return False
        if key in self._segment_input_keys(data):
            return False
        return not self._values_equal(data.get(key), parent_data.get(key))

    def _passthrough_segment_values(
        self,
        data: dict[str, Any],
        parent_keys: set[str],
        *,
        parent_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return unconfigured fields that can safely pass through assembly.

        Configured merge/overwrite fields are handled separately. Parent fields
        are restored from the parent store; only fields newly generated by a
        segment consumer, or parent fields intentionally sent to that consumer
        and then changed, are candidates for automatic pass-through.
        """

        segment_input_keys = self._segment_input_keys(data)
        passthrough: dict[str, Any] = {}
        for key, value in data.items():
            if key in self.field_merge_strategies or key in self.skip_me_keys:
                continue
            if key in self.unmerged_segment_field_allowlist or self._is_segment_helper_field(key):
                continue
            if parent_data is not None and key in parent_keys:
                if key not in segment_input_keys:
                    # The stored parent row already carries this value. If a
                    # later stage changed it without receiving it as input,
                    # validation raises before this point.
                    continue
                if self._values_equal(value, parent_data.get(key)):
                    continue
            passthrough[key] = copy.deepcopy(value)
        return passthrough

    def _consistent_passthrough_values(self, parent_id: str, ordered_segments: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        keys = sorted({key for segment in ordered_segments for key in segment.get("passthrough_values", {})})
        for key in keys:
            values = [
                segment["passthrough_values"][key]
                for segment in ordered_segments
                if key in segment["passthrough_values"]
            ]
            if not values:
                continue
            first = values[0]
            if all(self._values_equal(first, value) for value in values[1:]):
                result[key] = copy.deepcopy(first)
                continue
            if self.strict_unmerged_segment_fields:
                msg = (
                    "Global segment assembly received unconfigured generated field "
                    f"{key!r} with non-identical values across segments for parent {parent_id!r}. "
                    "Add the field to global_segment_assembler.field_merge_strategies, "
                    "global_segment_assembler.overwrite, or configure the field with a 'drop' merge strategy."
                )
                raise ValueError(msg)
            logger.debug(
                "Global segment assembly dropped unconfigured passthrough field {!r} with non-identical segment values",
                key,
            )
        return result

    def _is_segment_helper_field(self, key: str) -> bool:
        if key.startswith(("_curator_segment_", "_curator_global_", "_curator_terminal_", "_curator_payload_")):
            return True
        return key in {
            self.segment_start_key,
            self.segment_duration_key,
            self.duration_key,
            self.num_samples_key,
            self.waveform_key,
            self.waveform_ref_key,
            "_curator_payload_estimated_bytes",
            "_curator_payload_bytes",
            TERMINAL_DROPPED_KEY,
            "_curator_segment_dropped",
        }

    @classmethod
    def _segment_input_keys(cls, data: dict[str, Any]) -> set[str]:
        value = data.get(cls._SEGMENT_INPUT_KEYS_FIELD)
        if value is None:
            return set()
        if isinstance(value, str):
            return {value}
        try:
            return {str(item) for item in value if str(item)}
        except TypeError:
            return set()

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
            return True
        return left == right

    @staticmethod
    def _parent_source_index(parent_id: str, data: dict[str, Any]) -> int:
        value = data.get(TERMINAL_SOURCE_INDEX_KEY, data.get("_curator_segment_parent_source_index"))
        if value is not None:
            return int(value)
        try:
            return int(str(parent_id).rsplit(":", 1)[-1])
        except ValueError as exc:
            msg = f"Cannot recover source index for global segment parent {parent_id!r}"
            raise ValueError(msg) from exc

    def _assemble_parent(self, parent_id: str, entry: dict[str, Any]) -> dict[str, Any]:
        segments: dict[int, dict[str, Any]] = entry["segments"]
        ordered = [segments[idx] for idx in sorted(segments)]
        data = dict(entry["base_data"])
        for key, value in self._consistent_passthrough_values(parent_id, ordered).items():
            data[key] = value
        skipped_segments = [segment for segment in ordered if bool(segment["skipped"])]
        if skipped_segments:
            marker = self._partial_drop_marker(skipped_segments)
            for key, strategy in self.field_merge_strategies.items():
                if strategy != "drop":
                    data[key] = marker
            return {
                "data": data,
                "metadata": entry["metadata"],
                "stage_perf": self._dedupe_stage_perf(entry["stage_perf"]),
                "parent_id": parent_id,
                "source_index": int(entry["source_index"]),
            }
        for key, strategy in self.field_merge_strategies.items():
            if strategy == "drop":
                continue
            values = [segment["values"][key] for segment in ordered if key in segment["values"]]
            if not values:
                continue
            data[key] = self._merge_field_values(key, values, strategy)
        return {
            "data": data,
            "metadata": entry["metadata"],
            "stage_perf": self._dedupe_stage_perf(entry["stage_perf"]),
            "parent_id": parent_id,
            "source_index": int(entry["source_index"]),
        }

    @staticmethod
    def _partial_drop_marker(skipped_segments: list[dict[str, Any]]) -> str:
        stage_ids: list[str] = []
        for segment in skipped_segments:
            for stage_id in segment.get("drop_stage_ids", []):
                if stage_id and stage_id not in stage_ids:
                    stage_ids.append(str(stage_id))
        if not stage_ids:
            stage_ids = ["unknown_stage"]
        joined = ", ".join(stage_ids)
        return f"one or more intermediate segments dropped by {joined}"

    def _segment_drop_stage_ids(self, data: dict[str, Any]) -> list[str]:
        stage_ids: list[str] = []
        has_curator_drop_stage = False
        if bool(data.get(TERMINAL_DROPPED_KEY)):
            stage_ids.append(str(data.get(TERMINAL_DROPPED_BY_STAGE_KEY) or "unknown_stage"))
            has_curator_drop_stage = True
        if bool(data.get("_curator_segment_dropped")):
            stage_ids.append(str(data.get("_curator_segment_dropped_by_stage") or "unknown_stage"))
            has_curator_drop_stage = True
        if not has_curator_drop_stage:
            for key in sorted(self.skip_me_keys):
                if key in {TERMINAL_DROPPED_KEY, "_curator_segment_dropped"}:
                    continue
                if key in data:
                    stage_ids.append(str(key))
        deduped: list[str] = []
        for stage_id in stage_ids:
            if stage_id and stage_id not in deduped:
                deduped.append(stage_id)
        return deduped

    @staticmethod
    def _merge_field_values(key: str, values: list[Any], strategy: str) -> Any:
        if strategy in {"join", "join_text"}:
            return " ".join(str(value).strip() for value in values if str(value).strip())
        if strategy == "list":
            return values
        if strategy == "concat_list":
            merged: list[Any] = []
            for value in values:
                if isinstance(value, list):
                    merged.extend(value)
                else:
                    merged.append(value)
            return merged
        if strategy == "first":
            return values[0]
        if strategy == "last":
            return values[-1]
        if strategy == "first_non_null":
            for value in values:
                if value is not None:
                    return value
            return None
        if strategy == "last_non_null":
            for value in reversed(values):
                if value is not None:
                    return value
            return None
        if strategy == "sum":
            return sum(values)
        if strategy == "min":
            return min(values)
        if strategy == "max":
            return max(values)
        if strategy == "any":
            return any(bool(value) for value in values)
        if strategy == "all":
            return all(bool(value) for value in values)
        if strategy == "overwrite":
            first = values[0]
            if any(value != first for value in values[1:]):
                msg = f"Cannot overwrite field {key!r} with conflicting segment values"
                raise ValueError(msg)
            return first
        if strategy == "dict_merge":
            merged: dict[Any, Any] = {}
            for value in values:
                if not isinstance(value, dict):
                    msg = f"Cannot dict_merge non-dict value for field {key!r}: {type(value).__name__}"
                    raise TypeError(msg)
                merged.update(value)
            return merged
        msg = f"Unknown global segment merge strategy {strategy!r} for field {key!r}"
        raise ValueError(msg)

    @staticmethod
    def _normalise_skip_keys(skip_me_key: str, skip_me_keys: list[str] | None) -> set[str]:
        keys = [skip_me_key, *(skip_me_keys or []), TERMINAL_DROPPED_KEY, "_curator_segment_dropped"]
        return {str(key).strip() for key in keys if str(key).strip()}

    def _is_skipped_segment(self, data: dict[str, Any]) -> bool:
        for key in self.skip_me_keys:
            if key in {TERMINAL_DROPPED_KEY, "_curator_segment_dropped"}:
                if bool(data.get(key)):
                    return True
                continue
            if key in data:
                return True
        return False

    @staticmethod
    def _dedupe_stage_perf(stage_perf: list[Any]) -> list[Any]:
        deduped: list[Any] = []
        seen: set[str] = set()
        for perf in stage_perf:
            key = getattr(perf, "invocation_id", "") or f"id:{id(perf)}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(perf)
        return deduped


@dataclass
class GlobalSegmentAssemblerStage(ProcessingStage[AudioTask, AudioTask]):
    """Assemble globally planned segment rows back to parent rows.

    This is the generic inverse of ``ManifestReader(enable_global_bucketing=True)``.
    It is intentionally not ASR-specific: segment ownership comes from
    generic ``_curator_terminal_*`` metadata. Audio/global debug metadata under
    ``_curator_segment_*`` may also be present, but the terminal contract is
    what generic backends preserve when a segment is dropped. Pipeline-specific
    output fields are merged with ``field_merge_strategies``. ``text_keys_to_join``
    is a shortcut for ordered ASR-style ``join_text`` merges.
    """

    _curator_pipeline_helper_stage = True

    name: str = "GlobalSegmentAssemblerStage"
    text_keys_to_join: list[str] = field(default_factory=list)
    field_merge_strategies: dict[str, str] = field(default_factory=dict)
    skip_me_key: str = "_skip_me"
    skip_me_keys: list[str] = field(default_factory=list)
    waveform_key: str = "waveform"
    waveform_ref_key: str = "waveform_ref"
    duration_key: str = "duration"
    num_samples_key: str = "num_samples"
    segment_start_key: str = "segment_start_s"
    segment_duration_key: str = "segment_duration_s"
    max_ready_parents_in_memory: int = 4096
    spill_dir: str | None = None
    actor_name_prefix: str = "curator_global_segment_assembler"
    parent_store_actor_name_prefix: str = "curator_global_segment_parent_store"
    run_id: str | None = None
    strict_unmerged_segment_fields: bool = True
    unmerged_segment_field_allowlist: list[str] = field(default_factory=list)
    overwrite: list[str] = field(default_factory=list)
    _actor: Any = field(init=False, default=None, repr=False)
    _parent_store_actor: Any = field(init=False, default=None, repr=False)
    _parent_data_cache: dict[str, dict[str, Any]] = field(init=False, default_factory=dict, repr=False)
    _last_actor_metrics: dict[str, float] = field(init=False, default_factory=dict, repr=False)
    _curator_consumes_segment_rows: bool = field(init=False, default=True, repr=False)

    def __post_init__(self) -> None:
        self.text_keys_to_join = [str(key) for key in self.text_keys_to_join if str(key).strip()]
        self.field_merge_strategies = {
            str(key): str(value) for key, value in self.field_merge_strategies.items() if str(key).strip()
        }
        self.max_ready_parents_in_memory = int(self.max_ready_parents_in_memory)
        if self.max_ready_parents_in_memory < 0:
            msg = f"max_ready_parents_in_memory must be >= 0, got {self.max_ready_parents_in_memory}"
            raise ValueError(msg)
        for key in self.text_keys_to_join:
            self.field_merge_strategies.setdefault(key, "join_text")
        self.skip_me_keys = sorted(
            _GlobalSegmentAssemblyState._normalise_skip_keys(self.skip_me_key, self.skip_me_keys)
        )
        self.unmerged_segment_field_allowlist = [
            str(key).strip() for key in self.unmerged_segment_field_allowlist if str(key).strip()
        ]
        self.overwrite = [str(key).strip() for key in self.overwrite if str(key).strip()]
        for key in self.overwrite:
            existing = self.field_merge_strategies.get(key)
            if existing not in (None, "overwrite"):
                msg = f"Field {key!r} cannot be configured for both overwrite and {existing!r} merge"
                raise ValueError(msg)
            self.field_merge_strategies[key] = "overwrite"
        self.run_id = str(self.run_id or uuid.uuid4().hex)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], list(self.field_merge_strategies)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        # Create the Ray actor lazily so bucket-off runs that never produce
        # segment rows do not pay for an unused assembly actor.
        return None

    def num_workers(self) -> int | None:
        return 1

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {}

    def process(self, task: AudioTask) -> AudioTask | None:
        parent_id_value = task.data.get(TERMINAL_GROUP_ID_KEY, task.data.get("_curator_segment_parent_id"))
        if parent_id_value is None:
            return task
        self._ensure_actor()
        parent_id = str(parent_id_value)
        segment_idx = int(task.data.get(TERMINAL_INDEX_KEY, task.data.get("_curator_segment_idx", 0)))
        segment_count = int(task.data.get(TERMINAL_COUNT_KEY, task.data.get("_curator_segment_count", 1)))
        parent_data = self._parent_data(parent_id)
        assembled_items, completed_parent_id = self._ray_get(
            self._actor.add_segment.remote(
                parent_id=parent_id,
                segment_idx=segment_idx,
                segment_count=segment_count,
                data=dict(task.data),
                metadata=dict(task._metadata),
                stage_perf=list(task._stage_perf),
                parent_data=parent_data,
                include_completion=True,
            )
        )
        if completed_parent_id is not None:
            self._release_parent_data([completed_parent_id])
        self._log_actor_metric_deltas()
        if not assembled_items:
            return None
        if len(assembled_items) != 1:
            msg = (
                "GlobalSegmentAssemblerStage.process() cannot emit multiple assembled parents; "
                "call process_batch() for segment assembly."
            )
            raise RuntimeError(msg)
        assembled = assembled_items[0]
        output = AudioTask(
            dataset_name=task.dataset_name,
            data=assembled["data"],
            _metadata=assembled["metadata"],
            _stage_perf=assembled["stage_perf"],
        )
        output.task_id = f"assembled_{_safe_global_segment_actor_suffix(str(assembled['parent_id']))}"
        return output

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        outputs: list[AudioTask] = []
        completed_parent_ids: list[str] = []
        for task in tasks:
            parent_id_value = task.data.get(TERMINAL_GROUP_ID_KEY, task.data.get("_curator_segment_parent_id"))
            if parent_id_value is None:
                outputs.append(task)
                continue
            self._ensure_actor()
            parent_id = str(parent_id_value)
            segment_idx = int(task.data.get(TERMINAL_INDEX_KEY, task.data.get("_curator_segment_idx", 0)))
            segment_count = int(task.data.get(TERMINAL_COUNT_KEY, task.data.get("_curator_segment_count", 1)))
            parent_data = self._parent_data(parent_id)
            assembled_items, completed_parent_id = self._ray_get(
                self._actor.add_segment.remote(
                    parent_id=parent_id,
                    segment_idx=segment_idx,
                    segment_count=segment_count,
                    data=dict(task.data),
                    metadata=dict(task._metadata),
                    stage_perf=list(task._stage_perf),
                    parent_data=parent_data,
                    include_completion=True,
                )
            )
            if completed_parent_id is not None:
                completed_parent_ids.append(completed_parent_id)
            self._log_actor_metric_deltas()
            for assembled in assembled_items:
                output = AudioTask(
                    dataset_name=task.dataset_name,
                    data=assembled["data"],
                    _metadata=assembled["metadata"],
                    _stage_perf=assembled["stage_perf"],
                )
                output.task_id = f"assembled_{_safe_global_segment_actor_suffix(str(assembled['parent_id']))}"
                outputs.append(output)
        self._release_parent_data(completed_parent_ids)
        return outputs

    def _log_actor_metric_deltas(self) -> None:
        if self._actor is None:
            return
        snapshot = self._ray_get(self._actor.snapshot_metrics.remote())
        deltas: dict[str, float] = {}
        for key, value in snapshot.items():
            current = float(value)
            previous = float(self._last_actor_metrics.get(key, 0.0))
            delta = current - previous
            if delta > 0:
                deltas[key] = delta
            self._last_actor_metrics[key] = current
        if deltas:
            self._log_metrics(deltas)

    def _ensure_actor(self) -> None:
        if self._actor is not None:
            return
        import ray

        namespace = _current_ray_namespace()
        actor_name = f"{self.actor_name_prefix}_{_safe_global_segment_actor_suffix(str(self.run_id))}"
        try:
            self._actor = ray.get_actor(actor_name, namespace=namespace)
            return
        except ValueError:
            pass

        RemoteState = ray.remote(_GlobalSegmentAssemblyState)
        try:
            self._actor = RemoteState.options(name=actor_name, lifetime="detached", namespace=namespace).remote(
                text_keys_to_join=self.text_keys_to_join,
                field_merge_strategies=self.field_merge_strategies,
                skip_me_key=self.skip_me_key,
                skip_me_keys=self.skip_me_keys,
                waveform_key=self.waveform_key,
                waveform_ref_key=self.waveform_ref_key,
                duration_key=self.duration_key,
                num_samples_key=self.num_samples_key,
                segment_start_key=self.segment_start_key,
                segment_duration_key=self.segment_duration_key,
                max_ready_parents_in_memory=self.max_ready_parents_in_memory,
                spill_dir=self.spill_dir,
                strict_unmerged_segment_fields=self.strict_unmerged_segment_fields,
                unmerged_segment_field_allowlist=self.unmerged_segment_field_allowlist,
                overwrite_keys=self.overwrite,
                require_parent_data=True,
            )
            logger.info("Created global segment assembler actor {}", actor_name)
        except ValueError:
            self._actor = ray.get_actor(actor_name, namespace=namespace)

    def _parent_data(self, parent_id: str) -> dict[str, Any] | None:
        if parent_id in self._parent_data_cache:
            return copy.deepcopy(self._parent_data_cache[parent_id])
        self._ensure_parent_store_actor()
        if self._parent_store_actor is None:
            msg = (
                f"Global segment parent data store is unavailable for parent {parent_id!r}; "
                "cannot safely restore original parent fields after global bucketing."
            )
            raise RuntimeError(msg)
        parent_data = self._ray_get(self._parent_store_actor.get_parent.remote(parent_id))
        if parent_data is None:
            msg = (
                f"Global segment parent data for parent {parent_id!r} was not found in the parent store; "
                "cannot safely restore original parent fields after global bucketing."
            )
            raise RuntimeError(msg)
        self._parent_data_cache[parent_id] = copy.deepcopy(parent_data)
        return parent_data

    def _ensure_parent_store_actor(self) -> None:
        if self._parent_store_actor is not None:
            return
        import ray

        namespace = _current_ray_namespace()
        actor_name = f"{self.parent_store_actor_name_prefix}_{_safe_global_segment_actor_suffix(str(self.run_id))}"
        try:
            self._parent_store_actor = ray.get_actor(actor_name, namespace=namespace)
        except ValueError:
            self._parent_store_actor = None

    def _release_parent_data(self, parent_ids: list[str]) -> None:
        if not parent_ids:
            return
        unique_parent_ids = list(dict.fromkeys(str(parent_id) for parent_id in parent_ids))
        for parent_id in unique_parent_ids:
            self._parent_data_cache.pop(parent_id, None)
        if self._parent_store_actor is not None:
            self._ray_get(self._parent_store_actor.delete_many.remote(unique_parent_ids))

    def cleanup_run_resources(self) -> None:
        namespace = _current_ray_namespace()
        actor_name = f"{self.actor_name_prefix}_{_safe_global_segment_actor_suffix(str(self.run_id))}"
        if _kill_named_ray_actor(actor_name, namespace):
            self._actor = None
        store_name = f"{self.parent_store_actor_name_prefix}_{_safe_global_segment_actor_suffix(str(self.run_id))}"
        _kill_named_ray_actor(store_name, namespace)
        self._parent_store_actor = None
        self._parent_data_cache.clear()

    @staticmethod
    def _ray_get(obj: Any) -> Any:
        import ray

        return ray.get(obj)


@dataclass
class _ManifestReaderGlobalBucketingStage(ProcessingStage[EmptyTask, AudioTask]):
    """Read manifests and emit segment rows from a full-manifest bucket plan.

    The planner sees all manifest rows and planned segments, but it never loads
    waveform. Runtime rows are ordinary ``AudioTask`` segment tasks carrying
    ``segment_start_s`` and ``segment_duration_s``. Downstream audio reading
    decodes only that local-file segment, then a later assembler stitches segment
    ASR outputs back to one parent manifest row.
    """

    manifest_path: str | list[str]
    name: str = "manifest_reader_global_bucketing"
    storage_options: dict[str, Any] | None = None
    owner_stage: str | None = None
    parent_coalescing: bool = True
    duration_key: str = "duration"
    fallback_duration_keys: list[str] = field(default_factory=lambda: ["actual_duration", "duration_sec"])
    sample_rate_key: str = "sample_rate"
    num_samples_key: str = "num_samples"
    target_sample_rate: int = 16000
    target_nchannels: int = 1
    waveform_dtype_bytes: int = 4
    max_inference_duration_s: float = 120.0
    buckets_sec: list[float] = field(default_factory=lambda: [0.0, 30.0, 60.0, 120.0])
    max_items_per_batch_by_bucket: list[int] | None = None
    max_audio_sec_per_batch: float | None = None
    target_ready_batches_per_bucket: int = 4
    annotate_plan_metadata: bool = True
    annotate_segment_plan: bool = False
    segment_input_keys: list[str] = field(default_factory=lambda: ["audio_filepath"])
    run_id: str | None = None
    parent_store_actor_name_prefix: str = "curator_global_segment_parent_store"
    _parent_store_actor: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.manifest_path:
            msg = "manifest_path is required for ManifestReader(enable_global_bucketing=True)"
            raise ValueError(msg)
        self._validate_audio_planning()
        self._validate_buckets()
        self._validate_planning_limits()
        self.segment_input_keys = self._normalise_segment_input_keys(self.segment_input_keys)
        self.run_id = str(self.run_id or uuid.uuid4().hex)

    def _validate_audio_planning(self) -> None:
        self.max_inference_duration_s = resolve_max_model_input_duration(
            max_duration_s=self.max_inference_duration_s,
            owner="ManifestReader(enable_global_bucketing=True)",
        )
        if self.target_sample_rate <= 0:
            msg = f"target_sample_rate must be > 0, got {self.target_sample_rate}"
            raise ValueError(msg)
        if self.target_nchannels <= 0:
            msg = f"target_nchannels must be > 0, got {self.target_nchannels}"
            raise ValueError(msg)
        if self.waveform_dtype_bytes <= 0:
            msg = f"waveform_dtype_bytes must be > 0, got {self.waveform_dtype_bytes}"
            raise ValueError(msg)

    def _validate_buckets(self) -> None:
        self._validate_bucket_edges()
        self._validate_bucket_caps()
        self._validate_max_audio_sec_per_batch()

    def _validate_bucket_edges(self) -> None:
        if not self.buckets_sec:
            msg = "buckets_sec must contain at least one edge"
            raise ValueError(msg)
        for edge in self.buckets_sec:
            if isinstance(edge, bool) or not isinstance(edge, Real):
                msg = f"every buckets_sec entry must be numeric, got {type(edge).__name__}"
                raise TypeError(msg)
        if float(self.buckets_sec[0]) != 0.0:
            msg = f"buckets_sec must start at 0.0, got {self.buckets_sec[0]}"
            raise ValueError(msg)
        for i in range(len(self.buckets_sec) - 1):
            if float(self.buckets_sec[i + 1]) <= float(self.buckets_sec[i]):
                msg = (
                    f"buckets_sec must be strictly increasing; got {self.buckets_sec[i]} -> {self.buckets_sec[i + 1]}"
                )
                raise ValueError(msg)

    def _validate_bucket_caps(self) -> None:
        if self.max_items_per_batch_by_bucket is not None:
            if len(self.max_items_per_batch_by_bucket) != len(self.buckets_sec):
                msg = "max_items_per_batch_by_bucket length must match buckets_sec length"
                raise ValueError(msg)
            for cap in self.max_items_per_batch_by_bucket:
                if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
                    msg = f"max_items_per_batch_by_bucket entries must be positive ints, got {cap!r}"
                    raise ValueError(msg)

    def _validate_max_audio_sec_per_batch(self) -> None:
        if self.max_audio_sec_per_batch is not None:
            if isinstance(self.max_audio_sec_per_batch, bool) or not isinstance(self.max_audio_sec_per_batch, Real):
                msg = "max_audio_sec_per_batch must be numeric or None"
                raise TypeError(msg)
            if self.max_audio_sec_per_batch <= 0:
                msg = f"max_audio_sec_per_batch must be > 0 or None, got {self.max_audio_sec_per_batch}"
                raise ValueError(msg)

    def _validate_planning_limits(self) -> None:
        value = self.target_ready_batches_per_bucket
        if isinstance(value, bool) or not isinstance(value, Real):
            msg = "target_ready_batches_per_bucket must be numeric"
            raise TypeError(msg)
        if value <= 0:
            msg = f"target_ready_batches_per_bucket must be > 0, got {value}"
            raise ValueError(msg)

    def process(self, task: EmptyTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        records = self._read_manifest_records()
        self._store_parent_records(records)
        planned = self._plan_records(records)
        results = [
            self._segment_to_task(task, output_index, planned_segment)
            for output_index, planned_segment in enumerate(planned)
        ]

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "manifests_read": float(len(self._manifest_paths())),
                "entries_read": float(len(records)),
                "entries_emitted": float(len(results)),
                "global_manifest_bucketing_enabled": 1.0,
                "global_manifest_segments": float(sum(record.segment_count for record in records)),
                "global_manifest_planned_rows": float(len(results)),
                "global_manifest_parent_rows": float(len(records)),
                "global_manifest_segment_rows": float(len(results)),
                "audio_duration_s": float(sum(record.duration_s for record in records)),
                "estimated_waveform_bytes": float(
                    sum(
                        max(
                            1,
                            int(
                                segment.duration_s
                                * self.target_sample_rate
                                * self.target_nchannels
                                * self.waveform_dtype_bytes
                            ),
                        )
                        for planned_segment in planned
                        for segment in [planned_segment.segment]
                    )
                ),
            }
        )
        logger.info(
            "ManifestReader global bucketing: loaded {} parent rows from {} manifest(s); emitted {} segment row(s)",
            len(records),
            len(self._manifest_paths()),
            len(results),
        )
        return results

    def _manifest_paths(self) -> list[str]:
        if isinstance(self.manifest_path, str):
            return [self.manifest_path]
        return [str(path) for path in self.manifest_path]

    def _read_manifest_records(self) -> list[_ManifestPlanRecord]:
        records: list[_ManifestPlanRecord] = []
        source_index = 0
        for manifest_index, manifest in enumerate(self._manifest_paths()):
            fs, resolved = url_to_fs(manifest, **(self.storage_options or {}))
            manifest_count = 0
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for row_index, line in enumerate(f):
                    if not line.strip():
                        continue
                    data = json.loads(line.strip())
                    records.append(
                        self._build_record(
                            source_index=source_index,
                            manifest_index=manifest_index,
                            row_index=row_index,
                            data=data,
                        )
                    )
                    source_index += 1
                    manifest_count += 1
            logger.info("ManifestReader global bucketing: loaded {} entries from {}", manifest_count, manifest)
        return records

    def _store_parent_records(self, records: list[_ManifestPlanRecord]) -> None:
        parent_data = {self._parent_id(record): copy.deepcopy(record.data) for record in records}
        if not parent_data:
            return
        try:
            import ray

            if not ray.is_initialized():
                msg = (
                    "Global bucketing requires Ray to store original parent rows for later segment assembly. "
                    "Ray is not initialized, so parent fields cannot be safely restored."
                )
                raise RuntimeError(msg)
            namespace = _current_ray_namespace()
            actor_name = f"{self.parent_store_actor_name_prefix}_{_safe_global_segment_actor_suffix(str(self.run_id))}"
            try:
                actor = ray.get_actor(actor_name, namespace=namespace)
            except ValueError:
                RemoteStore = ray.remote(_GlobalSegmentParentDataStore)
                actor = RemoteStore.options(name=actor_name, lifetime="detached", namespace=namespace).remote()
            self._parent_store_actor = actor
            ray.get(actor.put_many.remote(parent_data))
            logger.info("Stored {} global segment parent rows in actor {}", len(parent_data), actor_name)
        except Exception as exc:
            msg = (
                "Global segment parent store unavailable; refusing to emit segment rows because the assembler "
                "would not be able to restore original parent fields."
            )
            raise RuntimeError(msg) from exc

    def _build_record(
        self,
        *,
        source_index: int,
        manifest_index: int,
        row_index: int,
        data: dict[str, Any],
    ) -> _ManifestPlanRecord:
        duration_s = self._duration_from_row(data)
        segments = tuple(self._planned_segments(source_index, data, duration_s))
        bucket_costs: dict[int, float] = {}
        bucket_counts: dict[int, int] = {}
        for segment in segments:
            bucket_costs[segment.bucket_id] = bucket_costs.get(segment.bucket_id, 0.0) + segment.duration_s
            bucket_counts[segment.bucket_id] = bucket_counts.get(segment.bucket_id, 0) + 1
        dominant_bucket = max(
            bucket_counts,
            key=lambda bucket: (bucket_counts[bucket], bucket_costs[bucket], bucket),
        )
        return _ManifestPlanRecord(
            source_index=source_index,
            manifest_index=manifest_index,
            row_index=row_index,
            data=data,
            duration_s=duration_s,
            estimated_waveform_bytes=self._estimate_waveform_bytes(data, duration_s),
            segments=segments,
            total_segment_cost_s=sum(segment.duration_s for segment in segments),
            dominant_bucket=dominant_bucket,
            dominant_bucket_cost_s=bucket_costs[dominant_bucket],
        )

    def _plan_records(self, records: list[_ManifestPlanRecord]) -> list[_PlannedManifestSegment]:
        if not records:
            return []
        ready_batches = self._global_ready_batches(records)
        by_parent = {record.source_index: record for record in records}
        ordered: list[tuple[_ManifestPlanRecord, _GlobalSegmentPlan]] = []
        assigned: set[tuple[int, int]] = set()
        for _, segment_batch, _ in ready_batches:
            for segment in segment_batch:
                key = (segment.parent_index, segment.segment_idx)
                if key in assigned:
                    continue
                ordered.append((by_parent[segment.parent_index], segment))
                assigned.add(key)
        for record in records:
            for segment in record.segments:
                key = (record.source_index, segment.segment_idx)
                if key in assigned:
                    continue
                ordered.append((record, segment))
                assigned.add(key)
        return [
            _PlannedManifestSegment(record=record, segment=segment, plan_order=plan_order)
            for plan_order, (record, segment) in enumerate(ordered)
        ]

    def _global_ready_batches(
        self,
        records: list[_ManifestPlanRecord],
    ) -> list[tuple[list[int], list[_GlobalSegmentPlan], float]]:
        from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy

        segment_items = [segment for record in records for segment in record.segments]
        if not segment_items:
            return []
        caps = self.max_items_per_batch_by_bucket or [
            max(1, int(self.target_ready_batches_per_bucket)) for _ in self.buckets_sec
        ]
        policy = BatchPolicy(
            enabled=True,
            strategy="duration_bucketed",
            buckets_sec=[float(edge) for edge in self.buckets_sec],
            max_items_per_batch_by_bucket=list(caps),
            max_audio_sec_per_batch=self.max_audio_sec_per_batch,
            prebatching_window_size=None,
            flush_interval_ms=0,
        )
        return policy.bucketize_with_costs(segment_items, cost_fn=lambda segment: segment.duration_s)

    def _segment_to_task(self, task: EmptyTask, output_index: int, planned: _PlannedManifestSegment) -> AudioTask:
        record = planned.record
        segment = planned.segment
        sample_rate = int(self._numeric_value(record.data.get(self.sample_rate_key)) or self.target_sample_rate)
        parent_num_samples = self._num_samples_for_planning(record.data, record.duration_s, sample_rate)
        segment_num_samples = max(0, int(segment.stop_sample - segment.start_sample))
        segment_start_s = float(segment.start_sample) / float(sample_rate)
        parent_id = self._parent_id(record)
        data = self._segment_input_data(record.data)
        data.update(
            {
                "segment_start_s": segment_start_s,
                "segment_duration_s": segment.duration_s,
                self.duration_key: segment.duration_s,
                self.num_samples_key: segment_num_samples,
                _GlobalSegmentAssemblyState._SEGMENT_INPUT_KEYS_FIELD: tuple(self.segment_input_keys),
                TERMINAL_GROUP_ID_KEY: parent_id,
                TERMINAL_INDEX_KEY: segment.segment_idx,
                TERMINAL_COUNT_KEY: record.segment_count,
                TERMINAL_SOURCE_INDEX_KEY: record.source_index,
                "_curator_segment_parent_id": parent_id,
                "_curator_segment_idx": segment.segment_idx,
                "_curator_segment_count": record.segment_count,
                "_curator_segment_bucket": segment.bucket_id,
                "_curator_segment_parent_duration_s": record.duration_s,
                "_curator_segment_parent_num_samples": parent_num_samples,
                "_curator_segment_parent_source_index": record.source_index,
                "_curator_segment_parent_manifest_index": record.manifest_index,
                "_curator_segment_parent_row_index": record.row_index,
            }
        )
        metadata = dict(task._metadata)
        if self.annotate_plan_metadata:
            metadata.update(
                {
                    "_curator_global_owner_stage": self.owner_stage,
                    "_curator_global_plan_source_index": record.source_index,
                    "_curator_global_plan_manifest_index": record.manifest_index,
                    "_curator_global_plan_row_index": record.row_index,
                    "_curator_global_plan_order": planned.plan_order,
                    "_curator_global_plan_bucket": segment.bucket_id,
                    "_curator_global_plan_segments": record.segment_count,
                    "_curator_global_plan_duration_s": segment.duration_s,
                    "_curator_global_parent_duration_s": record.duration_s,
                    "_curator_payload_estimated_bytes": segment_num_samples
                    * self.target_nchannels
                    * self.waveform_dtype_bytes,
                }
            )
            if self.annotate_segment_plan:
                metadata["_curator_global_plan_segment_boundaries"] = [
                    {
                        "segment_idx": planned_segment.segment_idx,
                        "start_sample": planned_segment.start_sample,
                        "stop_sample": planned_segment.stop_sample,
                        "duration_s": planned_segment.duration_s,
                        "bucket_id": planned_segment.bucket_id,
                    }
                    for planned_segment in record.segments
                ]
        audio_task = AudioTask(
            dataset_name=task.dataset_name,
            data=data,
            _metadata=metadata,
            _stage_perf=list(task._stage_perf),
        )
        audio_task._set_task_id(task.task_id, output_index)
        return audio_task

    @staticmethod
    def _parent_id(record: _ManifestPlanRecord) -> str:
        return f"{record.manifest_index}:{record.row_index}:{record.source_index}"

    def _segment_input_data(self, parent_data: dict[str, Any]) -> dict[str, Any]:
        return {key: copy.deepcopy(parent_data[key]) for key in self.segment_input_keys if key in parent_data}

    @staticmethod
    def _normalise_segment_input_keys(keys: list[str] | tuple[str, ...] | None) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for raw in keys or ["audio_filepath"]:
            key = str(raw).strip()
            if key and key not in seen:
                seen.add(key)
                result.append(key)
        if not result:
            msg = "segment_input_keys must contain at least one key so segment materialization can find audio input"
            raise ValueError(msg)
        return result

    def cleanup_run_resources(self) -> None:
        namespace = _current_ray_namespace()
        store_name = f"{self.parent_store_actor_name_prefix}_{_safe_global_segment_actor_suffix(str(self.run_id))}"
        _kill_named_ray_actor(store_name, namespace)
        self._parent_store_actor = None

    def _duration_from_row(self, data: dict[str, Any]) -> float:
        for key in [self.duration_key, *self.fallback_duration_keys]:
            if key in data:
                value = self._numeric_value(data.get(key))
                if value is not None and value > 0:
                    return float(value)
                if value is not None:
                    msg = (
                        "ManifestReader global bucketing requires a positive "
                        f"{key!r} value before audio payload materialization; got {data.get(key)!r}"
                    )
                    raise ValueError(msg)
        msg = (
            "ManifestReader global bucketing requires a positive duration estimate "
            f"in one of {[self.duration_key, *self.fallback_duration_keys]!r}"
        )
        raise ValueError(msg)

    def _planned_segments(
        self,
        parent_index: int,
        data: dict[str, Any],
        duration_s: float,
    ) -> list[_GlobalSegmentPlan]:
        sample_rate = int(self._numeric_value(data.get(self.sample_rate_key)) or self.target_sample_rate)
        num_samples = self._num_samples_for_planning(data, duration_s, sample_rate)
        if num_samples <= 0:
            return [
                _GlobalSegmentPlan(
                    parent_index=parent_index,
                    segment_idx=0,
                    start_sample=0,
                    stop_sample=0,
                    duration_s=0.0,
                    bucket_id=0,
                )
            ]

        segments = plan_audio_segments(
            num_samples=num_samples,
            sample_rate=sample_rate,
            max_duration_s=self.max_inference_duration_s,
            owner="ManifestReader(enable_global_bucketing=True)",
        )

        planned_segments: list[_GlobalSegmentPlan] = []
        for segment in segments:
            planned_segments.append(
                _GlobalSegmentPlan(
                    parent_index=parent_index,
                    segment_idx=segment.index,
                    start_sample=segment.start_sample,
                    stop_sample=segment.stop_sample,
                    duration_s=segment.duration_s,
                    bucket_id=self._bucket_for(segment.duration_s),
                )
            )
        return planned_segments

    def _num_samples_for_planning(self, data: dict[str, Any], duration_s: float, sample_rate: int) -> int:
        value = self._numeric_value(data.get(self.num_samples_key))
        if value is not None and value > 0:
            return int(value)
        return duration_to_num_samples(duration_s, sample_rate)

    def _estimate_waveform_bytes(self, data: dict[str, Any], duration_s: float) -> int:
        sample_rate = int(self._numeric_value(data.get(self.sample_rate_key)) or self.target_sample_rate)
        num_samples = self._num_samples_for_planning(data, duration_s, sample_rate)
        return int(num_samples * self.target_nchannels * self.waveform_dtype_bytes)

    @staticmethod
    def _numeric_value(value: object) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _bucket_for(self, cost: float) -> int:
        for i in range(len(self.buckets_sec) - 1, -1, -1):
            if cost >= float(self.buckets_sec[i]):
                return i
        return 0

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {}


@dataclass
class ManifestWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Append AudioTasks to a JSONL manifest file.

    The output file is truncated once in ``setup()`` (called on the driver)
    so repeated pipeline runs produce a clean output.  ``setup_on_node()``
    only creates the parent directory -- it never truncates, so multi-node
    deployments do not erase each other's data.

    The stage is pinned to one worker/actor for all supported backends so
    append writes to ``output_path`` are serialized. In-memory waveform tensors
    can be omitted through explicit serialization policy.

    Supports local and cloud paths via fsspec.

    Args:
        output_path: Destination JSONL path (local or cloud).
        write_perf_stats: If True, aggregate attached stage perf and refresh
            ``perf_summary.json`` next to the output manifest after each batch
            write, with teardown as a final backstop.
        drop_manifest_keys: Explicit task data keys to omit from JSONL output.
        drop_array_like_values: If True, omit tensor/array-like task data.
        perf_summary_path: Optional override for perf summary output path.
    """

    output_path: str
    name: str = "manifest_writer"
    write_perf_stats: bool = False
    duration_key: str = "duration"
    drop_manifest_keys: tuple[str, ...] = ()
    drop_array_like_values: bool = False
    perf_summary_path: str | None = None
    _writer_metrics: AudioManifestWriterMetrics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for ManifestWriterStage"
            raise ValueError(msg)
        self._writer_metrics = AudioManifestWriterMetrics(
            stage_name=self.name,
            duration_key=self.duration_key,
            write_perf_stats=self.write_perf_stats,
        )

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Truncate the output file once on the driver before processing starts."""
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)
        with self._fs.open(self._path, "w", encoding="utf-8"):
            pass
        if self.write_perf_stats:
            self._writer_metrics.reset_wall_timer()
        logger.info(f"ManifestWriterStage: writing to {self.output_path}")

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Ensure parent directory exists on each node (no truncation)."""
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)
        if self.write_perf_stats:
            self._writer_metrics.reset_wall_timer()

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)
        lines = manifest_lines(
            tasks,
            self.drop_manifest_keys,
            drop_array_like_values=self.drop_array_like_values,
        )
        if self.write_perf_stats:
            self._writer_metrics.record_invocation(len(tasks))
            write_t0 = time.perf_counter()
        with self._fs.open(self._path, "a", encoding="utf-8") as f:
            f.writelines(lines)
        if self.write_perf_stats:
            self._writer_metrics.add_manifest_write_time(time.perf_counter() - write_t0)
            for task in tasks:
                self._writer_metrics.record_task(task)
            self._write_perf_summary()
        copied_tasks = []
        for task in tasks:
            copied_task = AudioTask(
                dataset_name=task.dataset_name,
                data=task.data,
                _metadata=task._metadata,
                _stage_perf=list(task._stage_perf),
            )
            copied_tasks.append(copied_task)
        return copied_tasks

    def _resolved_perf_summary_path(self) -> str:
        if self.perf_summary_path:
            return self.perf_summary_path
        parent = self.output_path.rsplit("/", 1)[0] if "/" in self.output_path else ""
        return f"{parent}/perf_summary.json" if parent else "perf_summary.json"

    def _write_perf_summary(self) -> None:
        summary_path = self._resolved_perf_summary_path()
        fs, resolved = url_to_fs(summary_path)
        parent_dir = "/".join(resolved.split("/")[:-1])
        if parent_dir:
            fs.makedirs(parent_dir, exist_ok=True)
        summary = self._writer_metrics.build_perf_summary()
        write_t0 = time.perf_counter()
        with fs.open(resolved, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self._writer_metrics.add_perf_write_time(time.perf_counter() - write_t0)
        logger.info(f"Wrote perf_summary.json: {summary_path}")

    def record_external_stage_perf(self, perf_stats: Any) -> None:
        """Merge an externally collected stage summary into the persisted perf JSON.

        Executors can observe run-level data after the terminal writer has
        already flushed its normal summary.  This method performs a narrow
        read/merge/write of that single external stage so the writer's existing
        counters are not recomputed or overwritten by the driver-side stage
        instance.
        """
        if not self.write_perf_stats:
            return
        stage_summary = self._writer_metrics.build_external_stage_summary(perf_stats)
        if not stage_summary:
            return
        summary_path = self._resolved_perf_summary_path()
        fs, resolved = url_to_fs(summary_path)
        parent_dir = "/".join(resolved.split("/")[:-1])
        if parent_dir:
            fs.makedirs(parent_dir, exist_ok=True)
        summary: dict[str, Any]
        if fs.exists(resolved):
            try:
                with fs.open(resolved, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception as exc:
                logger.warning("Could not read existing perf_summary.json at {}: {}", summary_path, exc)
                summary = {}
        else:
            summary = {}
        stages = summary.setdefault("stages", {})
        if isinstance(stages, dict):
            stages[perf_stats.stage_name] = stage_summary
        else:
            summary["stages"] = {perf_stats.stage_name: stage_summary}
        write_t0 = time.perf_counter()
        with fs.open(resolved, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self._writer_metrics.add_perf_write_time(time.perf_counter() - write_t0)
        logger.info("Merged external perf stage {} into {}", perf_stats.stage_name, summary_path)

    def teardown(self) -> None:
        if self.write_perf_stats and (
            self._writer_metrics.items_processed > 0 or self._writer_metrics.total_utterances > 0
        ):
            self._write_perf_summary()
        elif self.write_perf_stats:
            logger.info("Skipping perf_summary.json write because no tasks were processed")

    def num_workers(self) -> int | None:
        return 1


def load_audio_file(audio_path: str, mono: bool = True) -> tuple[torch.Tensor, int]:
    """Load audio file and return waveform tensor (channels, samples) and sample rate."""
    data, sample_rate = soundfile.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data)
    waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform.T
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def ensure_waveform_2d(waveform: Any) -> torch.Tensor:
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
