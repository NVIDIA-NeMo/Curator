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
from operator import eq, ge, gt, le, lt, ne
from typing import Any

import soundfile
import torch
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.io.manifest_writer_utils import AudioManifestWriterMetrics
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask
from nemo_curator.utils.performance_utils import StagePerfStats


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
    duration_key: str = "duration"

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        paths = task.data
        results: list[AudioTask] = []
        count = 0
        for manifest in paths:
            fs, resolved = url_to_fs(manifest)
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(
                            AudioTask(
                                dataset_name=task.dataset_name,
                                data=json.loads(line.strip()),
                                _metadata=task._metadata,
                                _stage_perf=list(task._stage_perf),
                            )
                        )
                        count += 1
            logger.info(f"ManifestReaderStage: loaded {count} entries from {manifest}")
        duration_values: list[float] = []
        for item in results:
            if self.duration_key not in item.data:
                continue
            raw_duration = item.data[self.duration_key]
            if isinstance(raw_duration, bool):
                continue
            try:
                duration_s = float(raw_duration)
            except (TypeError, ValueError):
                continue
            if math.isfinite(duration_s) and duration_s >= 0:
                duration_values.append(duration_s)
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "manifests_read": len(paths),
                "entries_read": len(results),
                "pipeline_input_rows": len(results),
                "pipeline_input_audio_s": sum(duration_values),
                "pipeline_input_duration_rows": len(duration_values),
            }
        )
        return results

    def num_workers(self) -> int | None:
        return 1


@dataclass
class ManifestReader(CompositeStage[EmptyTask, AudioTask]):
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


@dataclass
class ManifestWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Append a single AudioTask to a JSONL manifest file.

    The output file is truncated once in ``setup()`` (called on the driver)
    so repeated pipeline runs produce a clean output.  ``setup_on_node()``
    only creates the parent directory -- it never truncates, so multi-node
    deployments do not erase each other's data.

    .. note::
       Because all nodes append to the same path, callers in multi-node
       setups should either use a shared filesystem or provide a
       node-unique ``output_path``.

    Supports local and cloud paths via fsspec.

    Args:
        output_path: Destination JSONL path (local or cloud).
        write_perf_stats: Write one terminal ``perf_summary.json`` for each
            successful pipeline run. Disabled by default.
        duration_key: Output manifest field containing audio seconds.
        perf_summary_path: Optional explicit summary path. Defaults to
            ``perf_summary.json`` beside ``output_path``.
        perf_run_id: Optional caller-owned run identifier. When blank inside a
            ``Pipeline``, Curator generates one for the run.
        perf_executor: Optional executor label. When blank inside a ``Pipeline``,
            Curator uses the concrete executor class name.
        perf_pipeline_metadata: Optional JSON-serializable metadata. Values are
            written as supplied and are not redacted.
    """

    output_path: str
    name: str = "manifest_writer"
    write_perf_stats: bool = False
    duration_key: str = "duration"
    perf_summary_path: str | None = None
    perf_run_id: str = ""
    perf_executor: str = ""
    perf_pipeline_metadata: dict[str, Any] | None = None
    _writer_metrics: AudioManifestWriterMetrics = field(init=False, repr=False)
    _external_perf_stats: list[StagePerfStats] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for ManifestWriterStage"
            raise ValueError(msg)
        self._reset_writer_metrics()

    def _reset_writer_metrics(self) -> None:
        self._writer_metrics = AudioManifestWriterMetrics(
            stage_name=self.name,
            duration_key=self.duration_key,
            write_perf_stats=self.write_perf_stats,
        )
        self._external_perf_stats = []

    def _prepare_output_path(self) -> None:
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)
        with self._fs.open(self._path, "w", encoding="utf-8"):
            pass

    def _remove_existing_perf_summary(self) -> None:
        if not self.write_perf_stats:
            return
        perf_fs, perf_path = url_to_fs(self._resolved_perf_summary_path())
        try:
            if perf_fs.exists(perf_path):
                perf_fs.rm(perf_path)
        except OSError as exc:
            logger.warning("Could not clear previous performance summary {}: {}", perf_path, exc)

    def prepare_performance_summary(self) -> None:
        """Prepare driver-owned output paths and clean run-scoped state."""
        self._reset_writer_metrics()
        self._prepare_output_path()
        self._remove_existing_perf_summary()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize one writer worker with fresh run-scoped metrics."""
        self._reset_writer_metrics()
        self._prepare_output_path()
        if not self._curator_run_id:
            self._remove_existing_perf_summary()
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

    def process(self, task: AudioTask) -> AudioTask:
        if self.write_perf_stats:
            self._writer_metrics.record_invocation(1)
        write_t0 = time.perf_counter()
        with self._fs.open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(task.data, ensure_ascii=False) + "\n")
        if self.write_perf_stats:
            manifest_write_time_s = time.perf_counter() - write_t0
            self._writer_metrics.add_manifest_write_time(manifest_write_time_s)
            self._writer_metrics.record_task(task)
            raw_duration = task.data.get(self.duration_key)
            if isinstance(raw_duration, bool):
                output_audio_s = 0.0
                output_duration_rows = 0.0
            else:
                try:
                    output_audio_s = float(raw_duration)
                except (TypeError, ValueError):
                    output_audio_s = 0.0
                    output_duration_rows = 0.0
                else:
                    output_duration_rows = float(math.isfinite(output_audio_s) and output_audio_s >= 0)
                    output_audio_s = output_audio_s if output_duration_rows else 0.0
            invocation_metrics = getattr(self, "_custom_metrics", None) or {}
            self._log_metrics(
                {
                    "manifest_write_time_s": invocation_metrics.get("manifest_write_time_s", 0.0)
                    + manifest_write_time_s,
                    "writer_process_calls": invocation_metrics.get("writer_process_calls", 0.0) + 1.0,
                    "writer_invocation_count": invocation_metrics.get("writer_invocation_count", 0.0) + 1.0,
                    "writer_items_processed": invocation_metrics.get("writer_items_processed", 0.0) + 1.0,
                    "pipeline_output_rows": invocation_metrics.get("pipeline_output_rows", 0.0) + 1.0,
                    "pipeline_output_audio_s": invocation_metrics.get("pipeline_output_audio_s", 0.0) + output_audio_s,
                    "pipeline_output_duration_rows": invocation_metrics.get("pipeline_output_duration_rows", 0.0)
                    + output_duration_rows,
                }
            )
        return AudioTask(
            dataset_name=task.dataset_name,
            data=task.data,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )

    def _resolved_perf_summary_path(self) -> str:
        if self.perf_summary_path:
            return self.perf_summary_path
        parent, separator, _filename = self.output_path.rpartition("/")
        return f"{parent}{separator}perf_summary.json" if separator else "perf_summary.json"

    def _resolved_perf_context(self) -> tuple[str, str, dict[str, Any]]:
        pipeline_metadata = dict(self._curator_pipeline_metadata or {})
        pipeline_metadata.update(self.perf_pipeline_metadata or {})
        return (
            self.perf_run_id or self._curator_run_id,
            self.perf_executor or self._curator_executor,
            pipeline_metadata,
        )

    def _write_perf_summary(
        self,
        *,
        wall_time_s: float | None = None,
        status: str = "completed",
        preserve_existing_stages: bool = True,
    ) -> None:
        summary_path = self._resolved_perf_summary_path()
        perf_fs, perf_path = url_to_fs(summary_path)
        parent_dir = "/".join(perf_path.split("/")[:-1])
        if parent_dir:
            perf_fs.makedirs(parent_dir, exist_ok=True)
        run_id, executor, pipeline_metadata = self._resolved_perf_context()
        summary = self._writer_metrics.build_perf_summary(
            stage_id=self._curator_stage_id,
            wall_time_s=wall_time_s,
            run_id=run_id,
            executor=executor,
            pipeline_metadata=pipeline_metadata,
        )
        if preserve_existing_stages:
            try:
                with perf_fs.open(perf_path, encoding="utf-8") as f:
                    existing = json.load(f)
                existing_stages = existing.get("stages", {})
                if isinstance(existing_stages, dict):
                    stages = summary.setdefault("stages", {})
                    for stage_key, stage_summary in existing_stages.items():
                        stages.setdefault(stage_key, stage_summary)
            except (FileNotFoundError, OSError, ValueError, TypeError):
                pass
        summary["status"] = status
        write_t0 = time.perf_counter()
        with perf_fs.open(perf_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self._writer_metrics.add_perf_write_time(time.perf_counter() - write_t0)

    def record_external_stage_perf(self, perf_stats: StagePerfStats) -> bool:
        """Accept one executor-owned record for driver finalization."""
        if not self.write_perf_stats:
            return False
        self._external_perf_stats.append(perf_stats)
        return True

    def record_external_stage_perfs(self, perf_stats: list[StagePerfStats]) -> bool:
        """Accept an authoritative invocation set from an optional backend collector."""
        if not self.write_perf_stats:
            return False
        self._external_perf_stats.extend(perf_stats)
        return True

    def teardown(self) -> None:
        # A planned pipeline writes once from ``finalize_performance_summary``
        # on the driver. Preserve direct stage usage outside Pipeline.
        if self.write_perf_stats and not self._curator_run_id:
            self._writer_metrics.record_stage_perf(self._external_perf_stats)
            self._write_perf_summary()

    def finalize_performance_summary(
        self,
        tasks: list[AudioTask],
        *,
        external_perf_stats: list[StagePerfStats],
        wall_time_s: float,
    ) -> None:
        """Write the authoritative driver-owned summary exactly once."""
        if not self.write_perf_stats:
            return
        final_metrics = AudioManifestWriterMetrics(
            stage_name=self.name,
            duration_key=self.duration_key,
            write_perf_stats=True,
        )
        for task in tasks:
            final_metrics.record_invocation(1)
            final_metrics.record_task(task)
        final_metrics.record_stage_perf([*self._external_perf_stats, *external_perf_stats])
        self._writer_metrics = final_metrics
        self._write_perf_summary(wall_time_s=wall_time_s)
        self._external_perf_stats = []

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
