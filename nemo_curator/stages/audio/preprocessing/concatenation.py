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

Concatenates multiple segment items within a single AudioBatch into one
combined waveform. Stores segment-to-original mappings in task._metadata
so downstream stages (TimestampMapperStage) can resolve final positions
back to the original file.

Uses canonical waveform + sample_rate format only (no pydub).

Example:
    from nemo_curator.stages.audio.preprocessing import SegmentConcatenationStage

    stage = SegmentConcatenationStage(silence_duration_sec=0.5)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from ..configs import SegmentConcatenationConfig


@dataclass
class SegmentMapping:
    """Mapping from concatenated position to original file position."""
    original_file: str
    original_start_ms: int
    original_end_ms: int
    concat_start_ms: int
    concat_end_ms: int
    segment_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_file': self.original_file,
            'original_start_ms': self.original_start_ms,
            'original_end_ms': self.original_end_ms,
            'concat_start_ms': self.concat_start_ms,
            'concat_end_ms': self.concat_end_ms,
            'segment_index': self.segment_index,
        }


@dataclass
class SegmentConcatenationStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Concatenate segment items within an AudioBatch into a single waveform.

    Takes AudioBatch(items=M) and returns AudioBatch(items=1) with a single
    combined waveform. Segment-to-original mappings are stored in
    ``task._metadata["segment_mappings"]`` for downstream timestamp resolution.

    Args:
        config: SegmentConcatenationConfig object (overrides other params if provided)
        silence_duration_sec: Duration of silence between segments (seconds)

    Example:
        stage = SegmentConcatenationStage(silence_duration_sec=0.5)
    """

    config: Optional[SegmentConcatenationConfig] = None
    silence_duration_sec: float = 0.5

    name: str = "SegmentConcatenation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()
        if self.config is not None:
            self.silence_duration_sec = self.config.silence_duration_sec

    def inputs(self) -> Tuple[List[str], List[str]]:
        return ["data"], []

    def outputs(self) -> Tuple[List[str], List[str]]:
        return [], ["waveform", "sample_rate", "num_segments", "total_duration_sec"]

    def process(self, task: AudioBatch) -> Optional[AudioBatch]:
        """
        Concatenate all items in the task into a single waveform item.

        Segment mappings are stored in ``task._metadata["segment_mappings"]``
        so that TimestampMapperStage can translate positions back to the
        original file.
        """
        items = task.data
        if not items:
            return AudioBatch(
                data=[], task_id=task.task_id, dataset_name=task.dataset_name,
                _metadata=task._metadata, _stage_perf=list(task._stage_perf),
            )

        parts: List[torch.Tensor] = []
        mappings: List[Dict[str, Any]] = []
        current_pos_ms = 0
        sample_rate: Optional[int] = None
        num_channels: Optional[int] = None
        silence_duration_ms = int(self.silence_duration_sec * 1000)

        for idx, item in enumerate(items):
            waveform = item.get('waveform')
            sr = item.get('sample_rate')
            if waveform is None:
                continue
            if sr is None:
                logger.error(f"[SegmentConcat] Skipping segment {idx}: 'sample_rate' key is missing. "
                             "Please set 'sample_rate' in the item dict.")
                continue
            if sr <= 0:
                logger.warning(f"[SegmentConcat] Skipping segment {idx}: invalid sample_rate={sr}")
                continue
            if not torch.is_tensor(waveform):
                waveform = torch.as_tensor(waveform, dtype=torch.float32)
            if parts and sr != sample_rate:
                logger.warning(
                    f"[SegmentConcat] Sample rate mismatch at segment {idx}: "
                    f"expected {sample_rate}Hz, got {sr}Hz. Output audio may be corrupted."
                )
            sample_rate = sr
            silence_samples = int(silence_duration_ms * sample_rate / 1000)

            # Normalize to 2D (channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            cur_channels = waveform.shape[0]
            if num_channels is None:
                num_channels = cur_channels
            elif cur_channels != num_channels:
                logger.warning(
                    f"[SegmentConcat] Channel count mismatch at segment {idx}: "
                    f"expected {num_channels}, got {cur_channels}. Skipping segment."
                )
                continue

            num_samples = waveform.shape[-1]
            segment_duration_ms = int(1000 * num_samples / sample_rate)

            orig_start = item.get('start_ms', 0)
            orig_end = item.get('end_ms', 0)
            if orig_end <= orig_start:
                orig_end = orig_start + segment_duration_ms

            mapping = SegmentMapping(
                original_file=item.get('original_file', item.get('audio_filepath', 'unknown')),
                original_start_ms=orig_start,
                original_end_ms=orig_end,
                concat_start_ms=current_pos_ms,
                concat_end_ms=current_pos_ms + segment_duration_ms,
                segment_index=idx,
            )
            mappings.append(mapping.to_dict())

            parts.append(waveform)
            current_pos_ms += segment_duration_ms

            parts.append(torch.zeros(num_channels, silence_samples, dtype=waveform.dtype, device=waveform.device))
            current_pos_ms += silence_duration_ms

        if not parts:
            return AudioBatch(
                data=[], task_id=task.task_id, dataset_name=task.dataset_name,
                _metadata=task._metadata, _stage_perf=list(task._stage_perf),
            )

        # Remove trailing silence
        combined = torch.cat(parts[:-1], dim=-1)
        current_pos_ms -= silence_duration_ms
        total_duration_sec = current_pos_ms / 1000.0

        original_file = items[0].get('original_file', items[0].get('audio_filepath', 'unknown'))

        output_data = {
            'waveform': combined,
            'sample_rate': sample_rate,
            'original_file': original_file,
            'num_segments': len(mappings),
            'total_duration_sec': total_duration_sec,
        }

        metadata = dict(task._metadata) if task._metadata else {}
        metadata['segment_mappings'] = mappings

        logger.info(f"[SegmentConcat] {len(mappings)} segments -> {total_duration_sec:.2f}s combined")

        return AudioBatch(
            data=[output_data],
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            _metadata=metadata,
            _stage_perf=list(task._stage_perf),
        )
