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
# ruff: noqa: ANN401

import json
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
from nemo_curator.stages.audio.io.manifest_writer_utils import AudioManifestWriterMetrics, manifest_lines
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    get = getattr(config, "get", None)
    if callable(get):
        return get(key, default)
    return default


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
            ManifestReaderStage(storage_options=self.storage_options),
        ]

    def get_description(self) -> str:
        parts = [f"Read JSONL manifests from {self.manifest_path}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        return ", ".join(parts)

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
            admission_actor_name=str(payload_config.get("admission_actor_name", "curator_payload_admission")),
            admission_poll_interval_s=float(payload_config.get("admission_poll_interval_s", 0.25)),
            admission_wait_timeout_s=float(payload_config.get("admission_wait_timeout_s", 4 * 60 * 60)),
            run_id=run_id,
        )


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

    def record_external_stage_perf(self, perf_stats: Any) -> bool:
        """Merge an externally collected stage summary into the persisted perf JSON."""
        if not self.write_perf_stats:
            return False
        stage_summary = self._writer_metrics.build_external_stage_summary(perf_stats)
        if not stage_summary:
            return False
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
            except Exception as exc:  # noqa: BLE001
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
        return True

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
