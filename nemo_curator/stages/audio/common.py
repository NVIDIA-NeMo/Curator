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

import json
import math
import os
import time
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
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask


def get_audio_duration(audio_filepath: str) -> float:
    """Get the duration of the audio file in seconds."""
    try:
        info = soundfile.info(audio_filepath)
        return info.frames / info.samplerate
    except Exception as e:  # noqa: BLE001
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

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        paths = task.data
        results: list[AudioTask] = []
        count = 0
        for manifest in paths:
            fs, resolved = url_to_fs(manifest)
            manifest_count = 0
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(
                            AudioTask(
                                task_id=f"{task.task_id}_{count}",
                                dataset_name=task.dataset_name,
                                data=json.loads(line.strip()),
                                _metadata=task._metadata,
                                _stage_perf=list(task._stage_perf),
                            )
                        )
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

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}


@dataclass
class ManifestReader(CompositeStage[_EmptyTask, AudioTask]):
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
    """

    manifest_path: str | list[str]
    name: str = "manifest_reader"
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".jsonl", ".json"])
    storage_options: dict[str, Any] | None = None

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
            ManifestReaderStage(),
        ]

    def get_description(self) -> str:
        parts = [f"Read JSONL manifests from {self.manifest_path}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        return ", ".join(parts)


@dataclass(frozen=True)
class _VirtualChunkPlan:
    parent_index: int
    chunk_idx: int
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
    chunks: tuple[_VirtualChunkPlan, ...]
    total_chunk_cost_s: float
    dominant_bucket: int
    dominant_bucket_cost_s: float

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True)
class _PlannedManifestRecord:
    record: _ManifestPlanRecord
    window_id: int
    window_order: int


@dataclass
class GlobalBucketedManifestReader(ProcessingStage[_EmptyTask, AudioTask]):
    """Read manifests and optionally order rows by a full-manifest bucket plan.

    This implements metadata-global parent-coalesced bucketing for audio
    pipelines. The planner sees all manifest rows and virtual chunks, but it
    never loads waveform. Runtime rows remain ordinary ``AudioTask`` parents, so
    downstream resampling streams each source once, keeps waveform in memory for
    the active backend window, and lets the configured owner GPU stage run its
    live chunk bucket queues under normal Ray Data/Xenna backpressure.
    """

    manifest_path: str | list[str]
    name: str = "global_bucketed_manifest_reader"
    enabled: bool = False
    owner_stage: str | None = None
    planning_scope: str = "full_manifest"
    planning_unit: str = "parent"
    gpu_unit: str = "virtual_chunk"
    parent_coalescing: bool = True
    duration_key: str = "duration"
    fallback_duration_keys: list[str] = field(default_factory=lambda: ["actual_duration", "duration_sec"])
    sample_rate_key: str = "sample_rate"
    num_samples_key: str = "num_samples"
    target_sample_rate: int = 16000
    target_nchannels: int = 1
    waveform_dtype_bytes: int = 4
    chunking_enabled: bool = True
    ideal_inference_segment_s: float = 120.0
    max_inference_duration_s: float | None = None
    buckets_sec: list[float] = field(default_factory=lambda: [0.0, 30.0, 60.0, 120.0])
    max_items_per_batch_by_bucket: list[int] | None = None
    max_audio_sec_per_batch: float | None = None
    max_waveform_bytes: int | str | None = None
    max_audio_seconds: float | None = None
    max_parent_rows: int | None = None
    target_ready_batches_per_bucket: int = 4
    annotate_plan_metadata: bool = True
    annotate_chunk_plan: bool = False

    def __post_init__(self) -> None:
        if not self.manifest_path:
            msg = "manifest_path is required for GlobalBucketedManifestReader"
            raise ValueError(msg)
        if not isinstance(self.enabled, bool):
            msg = f"GlobalBucketedManifestReader.enabled must be a bool, got {type(self.enabled).__name__}"
            raise TypeError(msg)
        if not isinstance(self.chunking_enabled, bool):
            msg = (
                "GlobalBucketedManifestReader.chunking_enabled must be a bool, "
                f"got {type(self.chunking_enabled).__name__}"
            )
            raise TypeError(msg)
        if self.planning_scope != "full_manifest":
            msg = f"planning_scope={self.planning_scope!r} is unsupported; use 'full_manifest'"
            raise ValueError(msg)
        if self.planning_unit != "parent":
            msg = f"planning_unit={self.planning_unit!r} is unsupported; use 'parent'"
            raise ValueError(msg)
        if self.gpu_unit != "virtual_chunk":
            msg = f"gpu_unit={self.gpu_unit!r} is unsupported; use 'virtual_chunk'"
            raise ValueError(msg)
        if not self.parent_coalescing:
            msg = "GlobalBucketedManifestReader requires parent_coalescing=true to avoid repeated audio I/O"
            raise ValueError(msg)
        if self.ideal_inference_segment_s <= 0:
            msg = f"ideal_inference_segment_s must be > 0, got {self.ideal_inference_segment_s}"
            raise ValueError(msg)
        if self.max_inference_duration_s is not None and self.max_inference_duration_s <= 0:
            msg = f"max_inference_duration_s must be > 0 or None, got {self.max_inference_duration_s}"
            raise ValueError(msg)
        if self.target_sample_rate <= 0:
            msg = f"target_sample_rate must be > 0, got {self.target_sample_rate}"
            raise ValueError(msg)
        if self.target_nchannels <= 0:
            msg = f"target_nchannels must be > 0, got {self.target_nchannels}"
            raise ValueError(msg)
        if self.waveform_dtype_bytes <= 0:
            msg = f"waveform_dtype_bytes must be > 0, got {self.waveform_dtype_bytes}"
            raise ValueError(msg)
        self._validate_buckets()
        self._validate_window_limits()

    def _validate_buckets(self) -> None:
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
                msg = f"buckets_sec must be strictly increasing; got {self.buckets_sec[i]} -> {self.buckets_sec[i + 1]}"
                raise ValueError(msg)
        if self.max_items_per_batch_by_bucket is not None:
            if len(self.max_items_per_batch_by_bucket) != len(self.buckets_sec):
                msg = "max_items_per_batch_by_bucket length must match buckets_sec length"
                raise ValueError(msg)
            for cap in self.max_items_per_batch_by_bucket:
                if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
                    msg = f"max_items_per_batch_by_bucket entries must be positive ints, got {cap!r}"
                    raise ValueError(msg)
        if self.max_audio_sec_per_batch is not None:
            if isinstance(self.max_audio_sec_per_batch, bool) or not isinstance(self.max_audio_sec_per_batch, Real):
                msg = "max_audio_sec_per_batch must be numeric or None"
                raise TypeError(msg)
            if self.max_audio_sec_per_batch <= 0:
                msg = f"max_audio_sec_per_batch must be > 0 or None, got {self.max_audio_sec_per_batch}"
                raise ValueError(msg)

    def _validate_window_limits(self) -> None:
        self._max_waveform_bytes_int = self._parse_byte_limit(self.max_waveform_bytes)
        for name, value in (
            ("max_audio_seconds", self.max_audio_seconds),
            ("max_parent_rows", self.max_parent_rows),
            ("target_ready_batches_per_bucket", self.target_ready_batches_per_bucket),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                msg = f"{name} must be numeric or None"
                raise TypeError(msg)
            if value <= 0:
                msg = f"{name} must be > 0 or None, got {value}"
                raise ValueError(msg)

    def process(self, task: _EmptyTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        records = self._read_manifest_records()
        planned = self._plan_records(records) if self.enabled else [
            _PlannedManifestRecord(record=record, window_id=0, window_order=record.source_index)
            for record in records
        ]
        results = [
            self._record_to_task(task, output_index, planned_record)
            for output_index, planned_record in enumerate(planned)
        ]

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "manifests_read": float(len(self._manifest_paths())),
                "entries_read": float(len(records)),
                "entries_emitted": float(len(results)),
                "global_manifest_bucketing_enabled": float(self.enabled),
                "global_manifest_virtual_chunks": float(sum(record.chunk_count for record in records)),
                "global_manifest_windows": float(len({planned_record.window_id for planned_record in planned})),
                "audio_duration_s": float(sum(record.duration_s for record in records)),
                "estimated_waveform_bytes": float(sum(record.estimated_waveform_bytes for record in records)),
            }
        )
        logger.info(
            "GlobalBucketedManifestReader: loaded {} rows from {} manifest(s); global planning {} with {} window(s)",
            len(results),
            len(self._manifest_paths()),
            "enabled" if self.enabled else "disabled",
            len({planned_record.window_id for planned_record in planned}),
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
            fs, resolved = url_to_fs(manifest)
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
            logger.info("GlobalBucketedManifestReader: loaded {} entries from {}", manifest_count, manifest)
        return records

    def _build_record(
        self,
        *,
        source_index: int,
        manifest_index: int,
        row_index: int,
        data: dict[str, Any],
    ) -> _ManifestPlanRecord:
        duration_s = self._duration_from_row(data)
        chunks = tuple(self._virtual_chunks(source_index, data, duration_s))
        bucket_costs: dict[int, float] = {}
        bucket_counts: dict[int, int] = {}
        for chunk in chunks:
            bucket_costs[chunk.bucket_id] = bucket_costs.get(chunk.bucket_id, 0.0) + chunk.duration_s
            bucket_counts[chunk.bucket_id] = bucket_counts.get(chunk.bucket_id, 0) + 1
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
            chunks=chunks,
            total_chunk_cost_s=sum(chunk.duration_s for chunk in chunks),
            dominant_bucket=dominant_bucket,
            dominant_bucket_cost_s=bucket_costs[dominant_bucket],
        )

    def _plan_records(self, records: list[_ManifestPlanRecord]) -> list[_PlannedManifestRecord]:
        if not records:
            return []
        ready_batches = self._global_ready_batches(records)
        windows = self._coalesce_ready_batches_into_windows(records, ready_batches)
        return [
            _PlannedManifestRecord(record=record, window_id=window_id, window_order=window_order)
            for window_id, window in enumerate(windows)
            for window_order, record in enumerate(window)
        ]

    def _global_ready_batches(
        self,
        records: list[_ManifestPlanRecord],
    ) -> list[tuple[list[int], list[_VirtualChunkPlan], float]]:
        from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy

        chunk_items = [chunk for record in records for chunk in record.chunks]
        if not chunk_items:
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
        return policy.bucketize_with_costs(chunk_items, cost_fn=lambda chunk: chunk.duration_s)

    def _coalesce_ready_batches_into_windows(
        self,
        records: list[_ManifestPlanRecord],
        ready_batches: list[tuple[list[int], list[_VirtualChunkPlan], float]],
    ) -> list[list[_ManifestPlanRecord]]:
        by_parent = {record.source_index: record for record in records}
        assigned: set[int] = set()
        windows: list[list[_ManifestPlanRecord]] = []
        current: list[_ManifestPlanRecord] = []
        current_audio_s = 0.0
        current_waveform_bytes = 0

        def flush_current() -> None:
            nonlocal current, current_audio_s, current_waveform_bytes
            if current:
                windows.append(current)
                current = []
                current_audio_s = 0.0
                current_waveform_bytes = 0

        for _, chunk_batch, _ in ready_batches:
            for chunk in chunk_batch:
                if chunk.parent_index in assigned:
                    continue
                record = by_parent[chunk.parent_index]
                if self._would_exceed_window(
                    current,
                    current_audio_s,
                    current_waveform_bytes,
                    record,
                ):
                    flush_current()
                current.append(record)
                current_audio_s += record.duration_s
                current_waveform_bytes += record.estimated_waveform_bytes
                assigned.add(record.source_index)

        for record in records:
            if record.source_index in assigned:
                continue
            if self._would_exceed_window(current, current_audio_s, current_waveform_bytes, record):
                flush_current()
            current.append(record)
            current_audio_s += record.duration_s
            current_waveform_bytes += record.estimated_waveform_bytes
            assigned.add(record.source_index)
        flush_current()
        return windows

    def _would_exceed_window(
        self,
        current: list[_ManifestPlanRecord],
        current_audio_s: float,
        current_waveform_bytes: int,
        record: _ManifestPlanRecord,
    ) -> bool:
        if not current:
            return False
        if self.max_parent_rows is not None and len(current) + 1 > int(self.max_parent_rows):
            return True
        if self.max_audio_seconds is not None and current_audio_s + record.duration_s > float(self.max_audio_seconds):
            return True
        if self._max_waveform_bytes_int is not None:
            return current_waveform_bytes + record.estimated_waveform_bytes > self._max_waveform_bytes_int
        return False

    def _record_to_task(self, task: _EmptyTask, output_index: int, planned: _PlannedManifestRecord) -> AudioTask:
        record = planned.record
        metadata = dict(task._metadata)
        if self.annotate_plan_metadata:
            metadata.update(
                {
                    "_curator_global_owner_stage": self.owner_stage,
                    "_curator_global_plan_source_index": record.source_index,
                    "_curator_global_plan_manifest_index": record.manifest_index,
                    "_curator_global_plan_row_index": record.row_index,
                    "_curator_global_plan_window_id": planned.window_id,
                    "_curator_global_plan_window_order": planned.window_order,
                    "_curator_global_plan_bucket": record.dominant_bucket,
                    "_curator_global_plan_chunks": record.chunk_count,
                    "_curator_global_plan_duration_s": record.duration_s,
                    "_curator_global_plan_estimated_waveform_bytes": record.estimated_waveform_bytes,
                }
            )
            if self.annotate_chunk_plan:
                metadata["_curator_global_plan_chunk_boundaries"] = [
                    {
                        "chunk_idx": chunk.chunk_idx,
                        "start_sample": chunk.start_sample,
                        "stop_sample": chunk.stop_sample,
                        "duration_s": chunk.duration_s,
                        "bucket_id": chunk.bucket_id,
                    }
                    for chunk in record.chunks
                ]
        return AudioTask(
            task_id=f"{task.task_id}_{output_index}",
            dataset_name=task.dataset_name,
            data=record.data,
            _metadata=metadata,
            _stage_perf=list(task._stage_perf),
        )

    def _duration_from_row(self, data: dict[str, Any]) -> float:
        for key in [self.duration_key, *self.fallback_duration_keys]:
            if key in data:
                value = self._numeric_value(data.get(key))
                if value is not None:
                    return max(0.0, float(value))
        return 0.0

    def _virtual_chunks(
        self,
        parent_index: int,
        data: dict[str, Any],
        duration_s: float,
    ) -> list[_VirtualChunkPlan]:
        sample_rate = int(self._numeric_value(data.get(self.sample_rate_key)) or self.target_sample_rate)
        num_samples = self._num_samples_for_planning(data, duration_s, sample_rate)
        if num_samples <= 0:
            return [
                _VirtualChunkPlan(
                    parent_index=parent_index,
                    chunk_idx=0,
                    start_sample=0,
                    stop_sample=0,
                    duration_s=0.0,
                    bucket_id=0,
                )
            ]

        if self.chunking_enabled:
            segment_s = float(self.max_inference_duration_s or self.ideal_inference_segment_s)
            max_samples = max(1, int(segment_s * sample_rate))
        else:
            max_samples = num_samples

        chunks: list[_VirtualChunkPlan] = []
        for chunk_idx, start in enumerate(range(0, num_samples, max_samples)):
            stop = min(start + max_samples, num_samples)
            chunk_s = float(stop - start) / float(sample_rate)
            chunks.append(
                _VirtualChunkPlan(
                    parent_index=parent_index,
                    chunk_idx=chunk_idx,
                    start_sample=start,
                    stop_sample=stop,
                    duration_s=chunk_s,
                    bucket_id=self._bucket_for(chunk_s),
                )
            )
        return chunks

    def _num_samples_for_planning(self, data: dict[str, Any], duration_s: float, sample_rate: int) -> int:
        value = self._numeric_value(data.get(self.num_samples_key))
        if value is not None and value > 0:
            return int(value)
        return int(math.ceil(max(duration_s, 0.0) * float(sample_rate)))

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

    @staticmethod
    def _parse_byte_limit(value: int | str | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            msg = "max_waveform_bytes must be an int, byte string, or None"
            raise TypeError(msg)
        if isinstance(value, int):
            if value <= 0:
                msg = f"max_waveform_bytes must be > 0 or None, got {value}"
                raise ValueError(msg)
            return value
        if not isinstance(value, str):
            msg = f"max_waveform_bytes must be an int, byte string, or None, got {type(value).__name__}"
            raise TypeError(msg)
        text = value.strip()
        if not text:
            return None
        units = {
            "b": 1,
            "kb": 1000,
            "mb": 1000**2,
            "gb": 1000**3,
            "tb": 1000**4,
            "kib": 1024,
            "mib": 1024**2,
            "gib": 1024**3,
            "tib": 1024**4,
        }
        number = text
        multiplier = 1
        for suffix, factor in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
            if text.lower().endswith(suffix):
                number = text[: -len(suffix)].strip()
                multiplier = factor
                break
        try:
            parsed = float(number)
        except ValueError as exc:
            msg = f"Invalid max_waveform_bytes value: {value!r}"
            raise ValueError(msg) from exc
        if parsed <= 0:
            msg = f"max_waveform_bytes must be > 0 or None, got {value!r}"
            raise ValueError(msg)
        return int(parsed * multiplier)

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}


@dataclass
class ManifestWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Append AudioTasks to a JSONL manifest file.

    The output file is truncated once in ``setup()`` (called on the driver)
    so repeated pipeline runs produce a clean output.  ``setup_on_node()``
    only creates the parent directory -- it never truncates, so multi-node
    deployments do not erase each other's data.

    The stage is pinned to one worker/actor for all supported backends so
    append writes to ``output_path`` are serialized. In-memory waveform tensors
    and array-like values are omitted from JSON output by default.

    Supports local and cloud paths via fsspec.

    Args:
        output_path: Destination JSONL path (local or cloud).
        write_perf_stats: If True, aggregate attached stage perf and refresh
            ``perf_summary.json`` next to the output manifest after each batch
            write, with teardown as a final backstop.
        drop_manifest_keys: Explicit task data keys to omit from JSONL output.
        perf_summary_path: Optional override for perf summary output path.
    """

    output_path: str
    name: str = "manifest_writer"
    write_perf_stats: bool = True
    duration_key: str = "duration"
    drop_manifest_keys: tuple[str, ...] = ("waveform",)
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
        lines = manifest_lines(tasks, self.drop_manifest_keys)
        self._writer_metrics.record_invocation(len(tasks))
        write_t0 = time.perf_counter()
        with self._fs.open(self._path, "a", encoding="utf-8") as f:
            f.writelines(lines)
        self._writer_metrics.add_manifest_write_time(time.perf_counter() - write_t0)
        for task in tasks:
            self._writer_metrics.record_task(task)
        if self.write_perf_stats:
            self._write_perf_summary()
        return [
            AudioTask(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                data=task.data,
                _metadata=task._metadata,
                _stage_perf=list(task._stage_perf),
            )
            for task in tasks
        ]

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

    def teardown(self) -> None:
        if self.write_perf_stats and (
            self._writer_metrics.items_processed > 0 or self._writer_metrics.total_utterances > 0
        ):
            self._write_perf_summary()
        elif self.write_perf_stats:
            logger.info("Skipping perf_summary.json write because no tasks were processed")

    def num_workers(self) -> int | None:
        return 1

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}


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
