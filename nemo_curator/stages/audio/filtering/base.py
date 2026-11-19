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

"""Shared mode/action handling and audio resolution for the audio quality filters.

``UTMOSFilterStage``, ``SIGMOSFilterStage`` and ``BandFilterStage`` score one
mono clip per item. This base owns everything around the score: validating the
mode/action/key configuration, choosing task or nested-segment scope, the
filter-vs-annotate survivor policy, and resolving the clip's mono audio.
Subclasses implement ``_process_single``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch
from loguru import logger

from nemo_curator.stages.audio._agent._residency import (
    normalize_audio_waveform,
    resolve_audio,
    validate_audio_key_configuration,
    validate_input_residency,
)
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.stages.audio._agent._residency import FileAudioHydration

_VALID_MODES = {"task", "segments", "auto"}
_VALID_ACTIONS = {"filter", "annotate"}


class AudioQualityFilterStage:
    """Mixin for dataclass filter stages that declare ``mode``, ``action``, ``input_residency`` and the audio keys."""

    FILE_AUDIO_HYDRATION: ClassVar[FileAudioHydration] = "auto_partial"
    INFER_SAMPLE_RATE_FROM_FILE: ClassVar[bool] = False

    def _validate_quality_filter_config(self, output_keys: dict[str, str]) -> None:
        if self.mode not in _VALID_MODES:
            msg = f"mode must be one of {_VALID_MODES!r}, got {self.mode!r}"
            raise ValueError(msg)
        if self.action not in _VALID_ACTIONS:
            msg = f"action must be one of {_VALID_ACTIONS!r}, got {self.action!r}"
            raise ValueError(msg)
        validate_input_residency(self.input_residency, stage_name=self.name)
        validate_audio_key_configuration(
            self.name,
            input_keys={
                "audio_filepath_key": self.audio_filepath_key,
                "waveform_key": self.waveform_key,
                "sample_rate_key": self.sample_rate_key,
                "segments_key": self.segments_key,
            },
            output_keys=output_keys,
        )

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Score ``task`` (or each nested segment) and filter or annotate it.

        Segment scope applies when ``mode="segments"``, or when ``mode="auto"``
        and ``task.data`` contains ``segments_key``. With ``action="filter"``
        only passing items survive; with ``action="annotate"`` every item is
        kept, including items that fail or cannot be scored.
        """
        use_segments = self.mode == "segments" or (self.mode == "auto" and self.segments_key in task.data)
        if not use_segments:
            return self._process_single(task) or (task if self.action == "annotate" else [])
        segments = task.data.get(self.segments_key)
        if segments is None:
            segments = []
        elif not isinstance(segments, list):
            logger.error(f"Expected {self.segments_key!r} to be a list, got {type(segments).__name__}")
            return task if self.action == "annotate" else []
        survivors = []
        for seg in segments:
            if not isinstance(seg, dict):
                logger.error(f"Expected each {self.segments_key!r} item to be a mapping")
                if self.action == "annotate":
                    survivors.append(seg)
                continue
            temp = AudioTask(data=seg)
            if self._process_single(temp) is not None or self.action == "annotate":
                survivors.append(temp.data)
        task.data[self.segments_key] = survivors
        return task if survivors or self.action == "annotate" else []

    @abstractmethod
    def _process_single(self, task: AudioTask) -> AudioTask | None:
        """Score one item; return it when it passes, ``None`` when it fails or cannot be scored."""

    def _resolve_mono_audio(self, task: AudioTask) -> tuple[torch.Tensor, int] | None:
        """Return the item's mono ``(1, N)`` float waveform and sample rate, or ``None`` (logged)."""
        try:
            resolved = resolve_audio(
                task.data,
                residency=self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
                mono=True,
                infer_sample_rate_from_file=self.INFER_SAMPLE_RATE_FROM_FILE,
                file_audio_hydration=self.FILE_AUDIO_HYDRATION,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{task.task_id}] Failed to load audio: {e}")
            return None
        if resolved is None:
            logger.warning(f"[{task.task_id}] No usable {self.input_residency!r} audio input found")
            return None
        waveform, sample_rate = resolved
        try:
            if torch.is_tensor(waveform) and not waveform.is_floating_point():
                waveform = waveform.to(dtype=torch.float32)
            return normalize_audio_waveform(waveform, stage_name=self.name, mono=True), int(sample_rate)
        except (RuntimeError, TypeError, ValueError) as e:
            logger.error(f"[{task.task_id}] Failed to normalize resident audio: {e}")
            return None
