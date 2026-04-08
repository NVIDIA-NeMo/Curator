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

Concatenates VAD segments stored in ``task.data["segments"]`` (nested mode)
into one combined waveform per source file.  Segments are sorted by
``segment_num`` (gaps from filtered-out segments are fine — order is
preserved) and concatenated with configurable silence between them.

Stores segment-to-original mappings in ``task._metadata`` so downstream
stages (TimestampMapperStage) can resolve final positions back to
the original file.

Uses canonical waveform + sample_rate format only (no pydub).

Example:
    from nemo_curator.stages.audio.preprocessing import SegmentConcatenationStage

    stage = SegmentConcatenationStage(silence_duration_sec=0.5)
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from nemo_curator.stages.audio.common import ensure_waveform_2d
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


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
    Concatenate nested VAD segments into a single combined waveform.

    Expects each incoming ``AudioTask`` to carry a
    ``task.data["segments"]`` list (one file = one task, produced by
    ``VADSegmentationStage(nested=True)``).  Segments are sorted by
    ``segment_num``, concatenated with silence gaps, and the result
    is a single ``AudioTask`` with the combined waveform and
    segment-to-original mappings in ``task._metadata["segment_mappings"]``.

    Args:
        silence_duration_sec: Duration of silence inserted between
            consecutive segments (seconds).
    """

    silence_duration_sec: float = 0.5

    name: str = "SegmentConcatenation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["waveform", "sample_rate", "num_segments", "total_duration_sec", "original_file"]

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Concatenate segments from ``task.data["segments"]``."""
        segments = task.data.get("segments")
        if segments is None:
            msg = "SegmentConcatenationStage requires task.data['segments'] (nested VAD mode)"
            raise ValueError(msg)

        if not segments:
            return []

        segments_sorted = sorted(segments, key=self._seg_sort_key)
        original_file = segments_sorted[0].get("original_file", "unknown")

        combined = self._concatenate(original_file, segments_sorted, task.task_id, task.dataset_name)
        if combined is None:
            return []
        return combined

    @staticmethod
    def _seg_sort_key(seg: dict[str, Any]) -> tuple[int, int, int]:
        """Sort key for segment dicts: (segment_num, start_ms, 0)."""
        seg_num = seg.get("segment_num")
        start = seg.get("start_ms")
        if seg_num is not None:
            return (int(seg_num), int(start) if start is not None else 0, 0)
        if start is not None:
            return (0, int(start), 0)
        return (0, 0, 0)

    @staticmethod
    def _validate_segment(seg: dict[str, Any]) -> tuple[torch.Tensor, int] | None:
        """Validate and return (waveform, sample_rate) or None if invalid."""
        waveform = seg.get("waveform")
        sr = seg.get("sample_rate")
        if waveform is None:
            return None
        seg_id = seg.get("segment_num", "?")
        if sr is None:
            logger.error(f"[SegmentConcat] Skipping segment {seg_id}: sample_rate key is missing.")
            return None
        if sr <= 0:
            logger.warning(f"[SegmentConcat] Skipping segment {seg_id}: invalid sample_rate={sr}")
            return None
        return ensure_waveform_2d(waveform), sr

    def _concatenate(
        self,
        original_file: str,
        segments: list[dict[str, Any]],
        task_id: str,
        dataset_name: str,
    ) -> AudioTask | None:
        """Concatenate a list of segment dicts from the same source file."""
        parts: list[torch.Tensor] = []
        mappings: list[dict[str, Any]] = []
        current_pos_ms = 0
        sample_rate: int | None = None
        num_channels: int | None = None
        silence_duration_ms = int(self.silence_duration_sec * 1000)

        for seg in segments:
            validated = self._validate_segment(seg)
            if validated is None:
                continue
            waveform, sr = validated

            if parts and sr != sample_rate:
                logger.warning(
                    f"[SegmentConcat] Sample rate mismatch: "
                    f"expected {sample_rate}Hz, got {sr}Hz. Skipping segment."
                )
                continue
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

            orig_start = seg.get("start_ms", 0)
            orig_end = seg.get("end_ms", 0)
            if orig_end <= orig_start:
                orig_end = orig_start + segment_duration_ms

            seg_num = seg.get("segment_num", len(mappings))
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

        logger.info(f"[SegmentConcat] {original_file}: {len(mappings)} segments -> {total_duration_sec:.2f}s combined")

        result_task = AudioTask(
            data=output_data,
            task_id=task_id,
            dataset_name=dataset_name,
        )
        result_task._metadata = {"segment_mappings": mappings}

        return result_task
