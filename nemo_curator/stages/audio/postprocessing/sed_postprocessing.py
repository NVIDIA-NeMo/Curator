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

"""SED Postprocessing stage: label audio entries with detected sound events.

Reads the per-frame probability matrix from task data (in-memory, preferred)
or from an NPZ sidecar file (fallback), then detects events for each
sound class group (speech, music, noise, etc.) using thresholding.
The detected events are added as labels to the task data — no filtering
or segmentation is performed.

Requires: ``pip install numpy`` (scipy optional for smoothing)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class SEDPostprocessingStage(ProcessingStage[AudioTask, AudioTask]):
    """Label audio entries with detected sound events from SED framewise output.

    Reads framewise data from task data (in-memory ``_sed_framewise`` key,
    preferred) or falls back to reading from NPZ file (``npz_filepath`` key).
    For each sound class group in ``SUPERCLASS_GROUPS`` (speech, music, noise,
    etc.), aggregates probabilities, converts to events via thresholding, and
    stores all detected events as ``sed_events`` on the output task.

    This is a **labeling** stage — the task passes through with event labels
    attached, no filtering or fan-out is performed.

    Args:
        agg_mode: Aggregation mode for class groups. Default ``"noisy_or"``.
        threshold: Probability threshold for event detection. Default 0.5.
        min_duration_sec: Minimum event duration. Default 0.3.
        smoothing_window_sec: Median filter window in seconds (0 = disabled).
        hysteresis_low: Low threshold for hysteresis (None = simple threshold).
        hysteresis_high: High threshold for hysteresis (None = simple threshold).
        merge_gap_sec: Merge events with gaps smaller than this (0 = disabled).
        framewise_key: Key in task data for in-memory framewise array. Default ``"_sed_framewise"``.
        npz_filepath_key: Key in task data for NPZ path (fallback). Default ``"npz_filepath"``.
        events_key: Key for output events list. Default ``"sed_events"``.
    """

    agg_mode: str = "noisy_or"
    threshold: float = 0.5
    min_duration_sec: float = 0.3
    smoothing_window_sec: float = 0.0
    hysteresis_low: float | None = None
    hysteresis_high: float | None = None
    merge_gap_sec: float = 0.0
    framewise_key: str = "_sed_framewise"
    npz_filepath_key: str = "npz_filepath"
    events_key: str = "sed_events"

    name: str = "SEDPostprocessing"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.events_key]

    def process(self, task: AudioTask) -> AudioTask:
        output_data = dict(task.data)
        output_data[self.events_key] = self._detect_all_events(output_data)
        output_data.pop(self.framewise_key, None)
        return AudioTask(
            task_id=f"{task.task_id}_sed_post",
            dataset_name=task.dataset_name,
            filepath_key=task.filepath_key,
            data=output_data,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def _detect_all_events(self, data: dict) -> list[dict]:
        """Detect events for all SUPERCLASS_GROUPS and return a merged, sorted list."""
        import numpy as np

        from nemo_curator.stages.audio.postprocessing.sed_utils import (
            SUPERCLASS_GROUPS,
            aggregate_speech_probs,
            framewise_to_events,
        )

        framewise_raw = data.get(self.framewise_key)
        if framewise_raw is not None:
            framewise = np.asarray(framewise_raw, dtype=np.float32)
            fps = float(data.get("sed_fps", self._default_fps()))
            valid_frames = int(data.get("sed_valid_frames", framewise.shape[0]))
        else:
            npz_path = str(data.get(self.npz_filepath_key, ""))
            if not npz_path or not os.path.exists(npz_path):
                logger.warning("No in-memory framewise data or NPZ file found; skipping.")
                return []
            with np.load(npz_path) as npz:
                framewise = npz["framewise"].astype(np.float32)
                fps = float(npz["fps"])
                valid_frames = int(npz.get("valid_frames", framewise.shape[0]))

        framewise = framewise[:valid_frames]
        smoothing_frames = int(self.smoothing_window_sec * fps) if self.smoothing_window_sec > 0 else 0

        all_events: list[dict] = []
        for label, class_indices in SUPERCLASS_GROUPS.items():
            probs = aggregate_speech_probs(framewise, class_indices, mode=self.agg_mode)
            events = framewise_to_events(
                probs=probs,
                fps=fps,
                threshold=self.threshold,
                min_duration_sec=self.min_duration_sec,
                smoothing_window_frames=smoothing_frames,
                hysteresis_low=self.hysteresis_low,
                hysteresis_high=self.hysteresis_high,
                merge_gap_sec=self.merge_gap_sec,
            )
            for evt in events:
                evt["label"] = label
            all_events.extend(events)

        all_events.sort(key=lambda e: e["start_time"])
        return all_events

    @staticmethod
    def _default_fps() -> float:
        return 16000.0 / 320