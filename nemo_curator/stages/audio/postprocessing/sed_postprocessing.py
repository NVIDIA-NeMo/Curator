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

"""Postprocess SED frame probabilities into timestamped sound events."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class SEDPostprocessingStage(ProcessingStage[AudioTask, AudioTask]):
    """Label audio entries with detected sound events from framewise SED output."""

    name: str = "SEDPostprocessing"
    agg_mode: str = "noisy_or"
    threshold: float = 0.5
    min_duration_sec: float = 0.3
    smoothing_window_sec: float = 0.0
    hysteresis_low: float | None = None
    hysteresis_high: float | None = None
    merge_gap_sec: float = 0.0
    emit_subcategories: bool = False
    framewise_key: str = "_sed_framewise"
    npz_filepath_key: str = "npz_filepath"
    events_key: str = "sed_events"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.events_key]

    def process(self, task: AudioTask) -> AudioTask:
        process_t0 = time.perf_counter()
        task.data[self.events_key] = self._detect_all_events(task.data)
        task.data.pop(self.framewise_key, None)
        self._log_metrics({
            "utterances_input": 1.0,
            "events_detected": float(len(task.data[self.events_key])),
            "process_time_s": time.perf_counter() - process_t0,
        })
        return task

    def _detect_all_events(self, data: dict) -> list[dict]:
        import numpy as np

        from nemo_curator.stages.audio.postprocessing.sed_utils import (
            AUDIOSET_CLASS_NAMES,
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
                logger.warning("No in-memory SED framewise data or NPZ file found; skipping")
                return []
            with np.load(npz_path) as npz:
                framewise = npz["framewise"].astype(np.float32)
                fps = float(npz["fps"])
                valid_frames = int(npz.get("valid_frames", framewise.shape[0]))

        if framewise.ndim != 2 or framewise.shape[0] == 0:
            return []

        framewise = framewise[:valid_frames]
        smoothing_frames = int(self.smoothing_window_sec * fps) if self.smoothing_window_sec > 0 else 0
        common_kwargs = {
            "fps": fps,
            "threshold": self.threshold,
            "min_duration_sec": self.min_duration_sec,
            "smoothing_window_frames": smoothing_frames,
            "hysteresis_low": self.hysteresis_low,
            "hysteresis_high": self.hysteresis_high,
            "merge_gap_sec": self.merge_gap_sec,
        }

        all_events: list[dict] = []
        if self.emit_subcategories:
            for superclass, class_indices in SUPERCLASS_GROUPS.items():
                for idx in class_indices:
                    if idx >= framewise.shape[1]:
                        continue
                    events = framewise_to_events(probs=framewise[:, idx], **common_kwargs)
                    subcategory_name = AUDIOSET_CLASS_NAMES.get(idx, f"class_{idx}")
                    for event in events:
                        event["label"] = subcategory_name
                        event["superclass"] = superclass
                    all_events.extend(events)
        else:
            for label, class_indices in SUPERCLASS_GROUPS.items():
                probs = aggregate_speech_probs(framewise, class_indices, mode=self.agg_mode)
                events = framewise_to_events(probs=probs, **common_kwargs)
                for event in events:
                    event["label"] = label
                all_events.extend(events)

        all_events.sort(key=lambda event: event["start_time"])
        return all_events

    @staticmethod
    def _default_fps() -> float:
        return 16000.0 / 320.0
