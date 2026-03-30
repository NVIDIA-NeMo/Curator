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

"""
Audio segment concatenation stage.

Concatenates multiple AudioTask segments into one combined waveform per
original source file. Tasks arriving in process_batch are grouped by
their ``original_file`` key, sorted by ``segment_num`` within each group
(gaps from filtered-out segments are fine -- order is preserved), and
concatenated into one AudioTask per source file.

Stores segment-to-original mappings in task._metadata so downstream
stages (TimestampMapperStage) can resolve final positions back to
the original file.

Uses canonical waveform + sample_rate format only (no pydub).

Example:
    from nemo_curator.stages.audio.preprocessing import SegmentConcatenationStage

    stage = SegmentConcatenationStage(silence_duration_sec=0.5)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

from nemo_curator.stages.audio.common import ensure_waveform_2d


@dataclass
class SegmentMapping:
    """Mapping from concatenated position to original file position."""
    original_file: str
    original_start_ms: int
    original_end_ms: int
    concat_start_ms: int
    concat_end_ms: int
    segment_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_file": self.original_file,
            "original_start_ms": self.original_start_ms,
            "original_end_ms": self.original_end_ms,
            "concat_start_ms": self.concat_start_ms,
            "concat_end_ms": self.concat_end_ms,
            "segment_index": self.segment_index,
        }


@dataclass
class SegmentConcatenationStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Concatenate AudioTask segments into a single combined waveform per source file.

    Uses process_batch to receive list[AudioTask] (segments potentially from
    multiple files, with possible gaps from filtered-out segments). Groups
    by ``original_file``, sorts by ``segment_num`` within each group, and
    produces one output AudioTask per source file.

    Segment-to-original mappings are stored in task._metadata["segment_mappings"]
    for downstream timestamp resolution.

    Args:
        silence_duration_sec: Duration of silence between segments (seconds)
        batch_size: Must be >= the max number of segments any single file
            can produce (VAD on a 1-hour file at 2s segments ~ 1800).
            All segments from one file must arrive in the same
            process_batch call for correct concatenation.
    """

    silence_duration_sec: float = 0.5

    name: str = "SegmentConcatenation"
    batch_size: int = 10000
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["waveform", "sample_rate", "num_segments", "total_duration_sec", "original_file"]

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Concatenate segments stored in ``task.data["segments"]`` (nested mode).

        Falls back to raising ``NotImplementedError`` when the task does not
        carry a ``"segments"`` key — callers should use ``process_batch``
        for the legacy fan-out path.
        """
        segments = task.data.get("segments")
        if segments is None:
            msg = "SegmentConcatenationStage.process() requires task.data['segments'] (nested mode)"
            raise NotImplementedError(msg)

        if not segments:
            return []

        seg_tasks = [
            AudioTask(data=seg, task_id=task.task_id, dataset_name=task.dataset_name)
            for seg in segments
        ]
        seg_tasks.sort(key=self._sort_key)

        original_file = segments[0].get("original_file", "unknown")
        combined = self._concatenate_group(original_file, seg_tasks)
        if combined is None:
            return []
        return combined

    @staticmethod
    def _sort_key(task: AudioTask) -> tuple[int, int, int]:
        """Sort key: (segment_num, start_ms, 0). Uses multiple fallbacks."""
        d = task.data
        seg = d.get("segment_num")
        start = d.get("start_ms")
        if seg is not None:
            return (int(seg), int(start) if start is not None else 0, 0)
        if start is not None:
            return (0, int(start), 0)
        return (0, 0, 0)

    @staticmethod
    def _group_key(task: AudioTask) -> str:
        """Resolve group key from task data, falling back through known keys."""
        d = task.data
        return d.get("original_file") or d.get("audio_filepath") or task.task_id or "unknown"

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """
        Group tasks by original source file, sort by segment order, concatenate.

        Handles edge cases:
        - Single task: passes through without concatenation overhead.
        - No segment_num: falls back to start_ms for ordering, then insertion order.
        - No original_file: falls back to audio_filepath, then task_id.
        - Filtered-out segments: gaps are fine, remaining segments keep their order.

        Returns one AudioTask per source file group.
        """
        if not tasks:
            return []

        groups: dict[str, list[AudioTask]] = defaultdict(list)
        for task in tasks:
            groups[self._group_key(task)].append(task)

        results: list[AudioTask] = []
        for original_file, group_tasks in groups.items():
            group_tasks.sort(key=self._sort_key)

            combined_task = self._concatenate_group(original_file, group_tasks)
            if combined_task is not None:
                results.append(combined_task)

        return results

    def _concatenate_group(self, original_file: str, tasks: list[AudioTask]) -> AudioTask | None:
        """Concatenate a group of tasks from the same original file."""
        parts: list[torch.Tensor] = []
        mappings: list[dict[str, Any]] = []
        current_pos_ms = 0
        sample_rate: int | None = None
        num_channels: int | None = None
        silence_duration_ms = int(self.silence_duration_sec * 1000)

        for task in tasks:
            item = task.data
            waveform = item.get("waveform")
            sr = item.get("sample_rate")
            if waveform is None:
                continue
            if sr is None:
                seg_id = item.get("segment_num", "?")
                logger.error(f"[SegmentConcat] Skipping segment {seg_id}: sample_rate key is missing.")
                continue
            if sr <= 0:
                seg_id = item.get("segment_num", "?")
                logger.warning(f"[SegmentConcat] Skipping segment {seg_id}: invalid sample_rate={sr}")
                continue

            waveform = ensure_waveform_2d(waveform)
            if parts and sr != sample_rate:
                logger.warning(
                    f"[SegmentConcat] Sample rate mismatch: "
                    f"expected {sample_rate}Hz, got {sr}Hz. Output audio may be corrupted."
                )
            sample_rate = sr
            silence_samples = int(silence_duration_ms * sample_rate / 1000)

            cur_channels = waveform.shape[0]
            if num_channels is None:
                num_channels = cur_channels
            elif cur_channels != num_channels:
                logger.warning(
                    f"[SegmentConcat] Channel count mismatch: "
                    f"expected {num_channels}, got {cur_channels}. Skipping segment."
                )
                continue

            num_samples = waveform.shape[-1]
            segment_duration_ms = int(1000 * num_samples / sample_rate)

            orig_start = item.get("start_ms", 0)
            orig_end = item.get("end_ms", 0)
            if orig_end <= orig_start:
                orig_end = orig_start + segment_duration_ms

            seg_num = item.get("segment_num", len(mappings))
            mapping = SegmentMapping(
                original_file=original_file,
                original_start_ms=orig_start,
                original_end_ms=orig_end,
                concat_start_ms=current_pos_ms,
                concat_end_ms=current_pos_ms + segment_duration_ms,
                segment_index=seg_num,
            )
            mappings.append(mapping.to_dict())

            parts.append(waveform)
            current_pos_ms += segment_duration_ms

            parts.append(torch.zeros(num_channels, silence_samples, dtype=waveform.dtype, device=waveform.device))
            current_pos_ms += silence_duration_ms

        if not parts:
            return None

        combined = torch.cat(parts[:-1], dim=-1)
        current_pos_ms -= silence_duration_ms
        total_duration_sec = current_pos_ms / 1000.0

        output_data = {
            "waveform": combined,
            "sample_rate": sample_rate,
            "original_file": original_file,
            "num_segments": len(mappings),
            "total_duration_sec": total_duration_sec,
        }

        logger.info(
            f"[SegmentConcat] {original_file}: "
            f"{len(mappings)} segments -> {total_duration_sec:.2f}s combined"
        )

        result_task = AudioTask(
            data=output_data,
            task_id=tasks[0].task_id,
            dataset_name=tasks[0].dataset_name,
        )
        result_task._metadata = {"segment_mappings": mappings}

        return result_task
