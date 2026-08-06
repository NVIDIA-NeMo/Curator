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
import os
import posixpath
import time
from dataclasses import dataclass, field
from operator import eq, ge, gt, le, lt, ne
from typing import TYPE_CHECKING, Any

import soundfile
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask
from nemo_curator.utils.file_utils import write_json_file_streaming_array

if TYPE_CHECKING:
    import fsspec

    from nemo_curator.utils.performance_utils import StagePerfStats


def _normalized_filesystem_path(fs: "fsspec.AbstractFileSystem", path: str) -> str:
    if isinstance(fs, LocalFileSystem):
        return os.path.realpath(os.path.abspath(path))
    return posixpath.normpath(path)


def _same_filesystem_location(
    left_fs: "fsspec.AbstractFileSystem",
    left_path: str,
    right_fs: "fsspec.AbstractFileSystem",
    right_path: str,
) -> bool:
    """Return whether two resolved fsspec locations identify the same destination."""
    same_filesystem = left_fs is right_fs or (
        type(left_fs) is type(right_fs)
        and left_fs.protocol == right_fs.protocol
        and left_fs.storage_options == right_fs.storage_options
    )
    return same_filesystem and _normalized_filesystem_path(left_fs, left_path) == _normalized_filesystem_path(
        right_fs, right_path
    )


def _same_filesystem_path(left_url: str, right_url: str) -> bool:
    """Return whether two URLs identify the same normalized fsspec destination."""
    left_fs, left_path = url_to_fs(left_url)
    right_fs, right_path = url_to_fs(right_url)
    return _same_filesystem_location(left_fs, left_path, right_fs, right_path)


def _append_slurm_shard_suffix(path: str, shard_index: int, total_shards: int) -> str:
    """Derive a deterministic per-shard filename without changing its directory."""
    parent, filename = posixpath.split(path)
    stem, suffix = posixpath.splitext(filename)
    sharded_filename = f"{stem}.shard-{shard_index:05d}-of-{total_shards:05d}{suffix}"
    return posixpath.join(parent, sharded_filename) if parent else sharded_filename


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
        performance_report_path: Optional JSON destination for all raw
            pipeline invocation metrics. Supports local and cloud paths through
            fsspec.
    """

    output_path: str
    name: str = "manifest_writer"
    performance_report_path: str | None = None

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for ManifestWriterStage"
            raise ValueError(msg)
        if self.performance_report_path is not None and _same_filesystem_path(
            self.output_path, self.performance_report_path
        ):
            msg = "performance_report_path must not resolve to the manifest output_path"
            raise ValueError(msg)

    def requests_performance_records(self) -> bool:
        """A configured raw report requires complete invocation collection."""
        return self.performance_report_path is not None

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Truncate the output file once on the driver before processing starts."""
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)
        with self._fs.open(self._path, "w", encoding="utf-8"):
            pass
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
        with self._fs.open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(task.data, ensure_ascii=False) + "\n")
        return AudioTask(
            dataset_name=task.dataset_name,
            data=task.data,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )

    def finalize_performance_report(
        self,
        *,
        performance_records: list["StagePerfStats"],
        wall_time_s: float,
        report_context: dict[str, Any],
    ) -> None:
        """Write all driver-collected invocation metrics through the existing writer."""
        self._write_raw_performance_report(
            performance_records=performance_records,
            wall_time_s=wall_time_s,
            report_context=report_context,
        )

    def _write_raw_performance_report(
        self,
        *,
        performance_records: list["StagePerfStats"],
        wall_time_s: float,
        report_context: dict[str, Any],
    ) -> None:
        """Persist the raw report so specialized writers can reuse the contract."""
        if self.performance_report_path is None:
            return
        report_fs, report_path = url_to_fs(self.performance_report_path)
        slurm_array = report_context["slurm_array"]
        shard_index = slurm_array["shard_index"] if slurm_array is not None else None
        total_shards = slurm_array["total_shards"] if slurm_array is not None else None
        if shard_index is not None and total_shards is not None:
            report_path = _append_slurm_shard_suffix(report_path, shard_index, total_shards)
        output_fs, output_path = url_to_fs(self.output_path)
        if _same_filesystem_location(output_fs, output_path, report_fs, report_path):
            msg = "effective performance report path must not resolve to the manifest output_path"
            raise ValueError(msg)
        iter_dicts = getattr(performance_records, "iter_dicts", None)
        record_dicts = (
            iter_dicts() if callable(iter_dicts) else (record.to_extended_dict() for record in performance_records)
        )
        write_json_file_streaming_array(
            report_path,
            {
                "schema_version": 1,
                **report_context,
                "wall_time_s": wall_time_s,
                "record_count": len(performance_records),
            },
            array_key="records",
            items=record_dicts,
            fs=report_fs,
        )
        logger.info(f"ManifestWriterStage: wrote performance report to {report_path}")

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
