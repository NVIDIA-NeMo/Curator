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

from nemo_curator.stages.audio._agent_ready import AgentReady, IOSpec, StageContract
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
class SegmentConcatenationStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
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
    segments_key: str = "segments"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    original_file_key: str = "original_file"
    num_segments_key: str = "num_segments"
    total_duration_sec_key: str = "total_duration_sec"

    name: str = "SegmentConcatenation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.waveform_key,
            self.sample_rate_key,
            self.num_segments_key,
            self.total_duration_sec_key,
            self.original_file_key,
        ]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.segments_key]),
            writes=IOSpec(
                data_keys=[
                    self.waveform_key,
                    self.sample_rate_key,
                    self.original_file_key,
                    self.num_segments_key,
                    self.total_duration_sec_key,
                ],
                produces=["tensor"],
            ),
            metadata_writes=["segment_mappings"],
            cardinality="N:1",
            iteration_key=self.segments_key,
        )

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Concatenate segments from ``task.data["segments"]``."""
        segments = task.data.get(self.segments_key)
        if segments is None:
            msg = f"SegmentConcatenationStage requires task.data[{self.segments_key!r}] (nested VAD mode)"
            raise ValueError(msg)

        if not segments:
            return []

        segments_sorted = sorted(segments, key=self._seg_sort_key)
        original_file = segments_sorted[0].get("original_file", "unknown")

        combined = self._concatenate(original_file, segments_sorted, task)
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

    def _validate_segment(self, seg: dict[str, Any]) -> tuple[torch.Tensor, int] | None:
        """Validate and return (waveform, sample_rate) or None if invalid."""
        waveform = seg.get(self.waveform_key)
        sr = seg.get(self.sample_rate_key)
        if waveform is None:
            logger.warning(
                f"[SegmentConcat] Skipping segment {seg.get('segment_num', '?')}: no "
                f"{self.waveform_key!r} (was VAD run with keep_segment_waveform_in_task=False?)"
            )
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
        parent_task: AudioTask,
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
                    f"[SegmentConcat] Sample rate mismatch: expected {sample_rate}Hz, got {sr}Hz. Skipping segment."
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
            self.waveform_key: combined,
            self.sample_rate_key: sample_rate,
            self.original_file_key: original_file,
            self.num_segments_key: len(mappings),
            self.total_duration_sec_key: total_duration_sec,
        }

        logger.info(f"[SegmentConcat] {original_file}: {len(mappings)} segments -> {total_duration_sec:.2f}s combined")

        return AudioTask(
            data=output_data,
            dataset_name=parent_task.dataset_name,
            _metadata={**(parent_task._metadata or {}), "segment_mappings": mappings},
            _stage_perf=list(parent_task._stage_perf),
        )
