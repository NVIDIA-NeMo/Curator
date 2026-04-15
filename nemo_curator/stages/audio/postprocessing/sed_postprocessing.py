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

"""SED Postprocessing stage: extract clean-speech events from SED NPZ output.

Reads the per-frame probability matrix saved by ``SEDInferenceStage``,
aggregates speech-class probabilities (noisy-or by default), applies
threshold/hysteresis/smoothing/merging to detect speech events, and
optionally classifies each candidate through a GBT model.

Requires: ``pip install numpy`` (scipy optional for smoothing)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.audio.postprocessing.sed_utils import (
    SPEECH_CLASS_INDICES,
    aggregate_speech_probs,
    framewise_to_events,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class SEDPostprocessingStage(ProcessingStage[AudioTask, AudioTask]):
    """Extract clean-speech timestamps from SED framewise NPZ output.

    Reads ``npz_filepath`` from the input task (produced by ``SEDInferenceStage``),
    aggregates speech probabilities across AudioSet speech classes, converts the
    probability curve to events using thresholding, and stores the detected events
    as ``predicted_events`` on the output task.

    Args:
        speech_agg_mode: Aggregation mode for speech classes. Default ``"noisy_or"``.
        speech_threshold: Simple threshold for event detection. Default 0.5.
        min_duration_sec: Minimum event duration. Default 0.3.
        smoothing_window_sec: Median filter window in seconds (0 = disabled).
        hysteresis_low: Low threshold for hysteresis (None = simple threshold).
        hysteresis_high: High threshold for hysteresis (None = simple threshold).
        merge_gap_sec: Merge events with gaps smaller than this (0 = disabled).
        classifier_model_path: Optional path to a GBT ``.joblib`` classifier.
        classifier_threshold: Minimum classifier probability to keep an event.
        npz_filepath_key: Key in task data for NPZ path. Default ``"npz_filepath"``.
        events_key: Key for output events list. Default ``"predicted_events"``.
    """

    speech_agg_mode: str = "noisy_or"
    speech_threshold: float = 0.5
    min_duration_sec: float = 0.3
    smoothing_window_sec: float = 0.0
    hysteresis_low: float | None = None
    hysteresis_high: float | None = None
    merge_gap_sec: float = 0.0
    classifier_model_path: str = ""
    classifier_threshold: float = 0.5
    npz_filepath_key: str = "npz_filepath"
    events_key: str = "predicted_events"

    name: str = "SEDPostprocessing"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def setup(self, _worker_metadata: Any = None) -> None:
        self._classifier = None
        if self.classifier_model_path:
            try:
                import joblib

                self._classifier = joblib.load(self.classifier_model_path)
                logger.info(f"Loaded GBT classifier from {self.classifier_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load classifier: {e}")

    def teardown(self) -> None:
        self._classifier = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.npz_filepath_key, self.events_key]

    def process(self, task):
        import numpy as np
        import pandas as pd

        from nemo_curator.tasks import DocumentBatch

        if isinstance(task, DocumentBatch):
            df = task.to_pandas() if hasattr(task, "to_pandas") else task.data
            results = []
            for _, row in df.iterrows():
                r = row.to_dict()
                npz_path = str(r.get(self.npz_filepath_key, ""))
                if npz_path and os.path.exists(npz_path):
                    r[self.events_key] = self._detect_events(npz_path)
                else:
                    r[self.events_key] = []
                results.append(r)
            return DocumentBatch(data=pd.DataFrame(results), dataset_name=task.dataset_name, task_id=task.task_id)

        # AudioTask path
        npz_path = task.data.get(self.npz_filepath_key, "")
        if not npz_path or not os.path.exists(npz_path):
            msg = f"Missing or nonexistent {self.npz_filepath_key}: {npz_path!r}"
            raise ValueError(msg)

        output_data = dict(task.data)
        output_data[self.events_key] = self._detect_events(npz_path)
        return AudioTask(
            task_id=f"{task.task_id}_sed_post",
            dataset_name=task.dataset_name,
            filepath_key=task.filepath_key,
            data=output_data,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def _detect_events(self, npz_path: str) -> list[dict]:
        import numpy as np

        with np.load(npz_path) as data:
            framewise = data["framewise"].astype(np.float32)
            fps = float(data["fps"])
            valid_frames = int(data.get("valid_frames", framewise.shape[0]))

        framewise = framewise[:valid_frames]
        speech_probs = aggregate_speech_probs(framewise, SPEECH_CLASS_INDICES, mode=self.speech_agg_mode)

        smoothing_frames = int(self.smoothing_window_sec * fps) if self.smoothing_window_sec > 0 else 0
        events = framewise_to_events(
            probs=speech_probs,
            fps=fps,
            threshold=self.speech_threshold,
            min_duration_sec=self.min_duration_sec,
            smoothing_window_frames=smoothing_frames,
            hysteresis_low=self.hysteresis_low,
            hysteresis_high=self.hysteresis_high,
            merge_gap_sec=self.merge_gap_sec,
        )
        for evt in events:
            evt["label"] = "speech"
        return events
