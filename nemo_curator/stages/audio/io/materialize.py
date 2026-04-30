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

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask
from nemo_curator.tasks.audio_task import build_audio_sample_key, ensure_checkpoint_shard_id
from nemo_curator.utils.remote_io import basename_from_path


def _path_suffix(path: str, fallback: str = ".bin") -> str:
    suffix = Path(basename_from_path(path)).suffix
    return suffix if suffix else fallback


@dataclass
class BaseAudioMaterializeStage(ProcessingStage[AudioTask, AudioTask]):
    audio_filepath_key: str = "audio_filepath"
    manifest_audio_filepath_key: str = "_manifest_audio_filepath"
    temporary_audio_key: str = "_temporary_audio_path"
    materialization_mode_key: str = "_materialization_mode"
    materialized_field_name_key: str = "_materialized_audio_field"
    offset_key: str = "offset"
    duration_key: str = "duration"
    temp_dir: str | None = None
    materialization_dir: str | None = None
    segment_if_offset_present: bool = True
    name: str = "base_materialize_audio"

    def process(self, task: AudioTask) -> AudioTask:
        msg = f"{self.__class__.__name__} only supports process_batch"
        raise NotImplementedError(msg)

    def _materialize_tasks_from_bytes(
        self,
        tasks: list[AudioTask],
        raw_audio: bytes,
        source_name: str,
        *,
        reference_field: str,
        output_field: str,
    ) -> None:
        field_paths = (reference_field, output_field)
        segment_tasks: list[AudioTask] = []
        member_tasks_only: list[AudioTask] = []
        for task in tasks:
            if self._should_segment(task, source_name, reference_field=field_paths[0]):
                segment_tasks.append(task)
            else:
                member_tasks_only.append(task)

        decoded_audio: tuple[Any, int] | None = None
        if segment_tasks:
            decoded_audio = soundfile.read(io.BytesIO(raw_audio), dtype="float32")

        for task in member_tasks_only:
            self._materialize_task(
                task,
                raw_audio,
                source_name,
                field_paths=field_paths,
            )
        for task in segment_tasks:
            self._materialize_task(
                task,
                raw_audio,
                source_name,
                field_paths=field_paths,
                decoded_audio=decoded_audio,
            )

    def _materialize_task(
        self,
        task: AudioTask,
        raw_audio: bytes,
        source_name: str,
        *,
        field_paths: tuple[str, str],
        decoded_audio: tuple[Any, int] | None = None,
    ) -> None:
        reference_field, output_field = field_paths
        should_segment = self._should_segment(task, source_name, reference_field=reference_field)
        output_path, is_temporary = self._create_output_path(
            task,
            output_field=output_field,
            suffix=".wav" if should_segment else _path_suffix(source_name),
        )
        if should_segment:
            self._write_segmented_audio(task, raw_audio, output_path, decoded_audio=decoded_audio)
            materialization_mode = "segment"
        else:
            output_path.write_bytes(raw_audio)
            materialization_mode = "member"

        if output_field == self.audio_filepath_key:
            task.data.setdefault(self.manifest_audio_filepath_key, task.data.get(self.audio_filepath_key))
        task.data[output_field] = output_path.as_posix()
        task.data[self.materialized_field_name_key] = output_field
        if is_temporary:
            task.data[self.temporary_audio_key] = output_path.as_posix()
        else:
            task.data.pop(self.temporary_audio_key, None)
        task.data[self.materialization_mode_key] = materialization_mode

    def _should_segment(self, task: AudioTask, source_name: str, *, reference_field: str) -> bool:
        if not self.segment_if_offset_present:
            return False
        # Prefer the original manifest path when available so retries do not compare
        # against a previously materialized temp path. If neither field exists, the
        # empty-string fallback intentionally makes the comparison conservative.
        original_path = str(
            task.data.get(self.manifest_audio_filepath_key, task.data.get(reference_field, "")) or ""
        )
        offset = float(task.data.get(self.offset_key, 0.0) or 0.0)
        return source_name != original_path or offset > 0.0 or task.data.get(self.duration_key) is not None

    def _create_temp_path(self, *, suffix: str) -> Path:
        target_dir = Path(self.temp_dir) if self.temp_dir is not None else Path(tempfile.gettempdir())
        target_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="nemo_curator_materialized_audio_", suffix=suffix, dir=target_dir)
        os.close(fd)
        return Path(path)

    def _get_sample_key(self, task: AudioTask) -> str:
        if task.sample_key:
            return task.sample_key
        task.sample_key = build_audio_sample_key(task.data, dataset_name=task.dataset_name)
        return task.sample_key

    def _create_output_path(self, task: AudioTask, *, output_field: str, suffix: str) -> tuple[Path, bool]:
        if self.materialization_dir is None:
            return self._create_temp_path(suffix=suffix), True

        sample_basis = self._get_sample_key(task)
        if output_field != self.audio_filepath_key:
            sample_basis = f"{sample_basis}:{output_field}"
        sample_hash = hashlib.sha256(sample_basis.encode("utf-8")).hexdigest()
        shard_id = ensure_checkpoint_shard_id(task)
        target_dir = Path(self.materialization_dir) / shard_id / sample_hash[:2]
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{sample_hash}{suffix}", False

    def _write_segmented_audio(
        self,
        task: AudioTask,
        raw_audio: bytes,
        output_path: Path,
        *,
        decoded_audio: tuple[Any, int] | None = None,
    ) -> None:
        offset = float(task.data.get(self.offset_key, 0.0) or 0.0)
        duration = task.data.get(self.duration_key)
        if duration is not None and float(duration) <= 0.0:
            msg = f"Duration must be greater than 0 for segmented audio, got {duration!r}"
            raise RuntimeError(msg)
        if decoded_audio is None:
            waveform, sample_rate = soundfile.read(io.BytesIO(raw_audio), dtype="float32")
        else:
            waveform, sample_rate = decoded_audio
        start = max(round(offset * sample_rate), 0)
        member_name = str(task.data.get(self.audio_filepath_key, "unknown"))
        if start >= waveform.shape[0]:
            msg = f"Offset {offset}s exceeds audio length for source '{member_name}'"
            raise RuntimeError(msg)
        end = waveform.shape[0]
        if duration is not None:
            end = min(start + round(float(duration) * sample_rate), waveform.shape[0])
        soundfile.write(output_path.as_posix(), waveform[start:end], sample_rate)


@dataclass
class CleanupTemporaryAudioStage(ProcessingStage[AudioTask, AudioTask]):
    temporary_audio_key: str = "_temporary_audio_path"
    audio_filepath_key: str = "audio_filepath"
    manifest_audio_filepath_key: str = "_manifest_audio_filepath"
    materialization_mode_key: str = "_materialization_mode"
    materialized_field_name_key: str = "_materialized_audio_field"
    restore_manifest_audio_filepath: bool = True
    drop_temporary_metadata: bool = True
    drop_materialized_field: bool = True
    ignore_missing: bool = True
    name: str = "cleanup_temporary_audio"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def process(self, task: AudioTask) -> AudioTask:
        temp_path = task.data.get(self.temporary_audio_key)
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                if not self.ignore_missing:
                    raise

        materialized_field_name = task.data.get(self.materialized_field_name_key)
        if (
            self.restore_manifest_audio_filepath
            and self.manifest_audio_filepath_key in task.data
            and materialized_field_name in {None, self.audio_filepath_key}
        ):
            task.data[self.audio_filepath_key] = task.data[self.manifest_audio_filepath_key]

        if self.drop_temporary_metadata:
            if (
                self.drop_materialized_field
                and isinstance(materialized_field_name, str)
                and materialized_field_name != self.audio_filepath_key
            ):
                task.data.pop(materialized_field_name, None)
            task.data.pop(self.temporary_audio_key, None)
            task.data.pop(self.materialization_mode_key, None)
            task.data.pop(self.manifest_audio_filepath_key, None)
            task.data.pop(self.materialized_field_name_key, None)

        return task
