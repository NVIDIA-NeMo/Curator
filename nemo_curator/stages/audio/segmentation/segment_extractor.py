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

"""Segment extractor: fan-out AudioTasks from SED-detected speech events.

Takes an AudioTask with ``predicted_events`` (from SEDPostprocessingStage)
and produces one AudioTask per event, each carrying the event's start/end
timestamps. Follows the same fan-out pattern as ``VADSegmentationStage``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class SegmentExtractorStage(ProcessingStage[AudioTask, AudioTask]):
    """Fan-out: one AudioTask per detected speech event.

    Reads ``events_key`` from the input task (list of event dicts with
    ``start_time`` and ``end_time``), and emits one output AudioTask per
    event with the segment timestamps attached.

    Args:
        events_key: Key in task data for the events list. Default ``"predicted_events"``.
        filepath_key: Key for audio file path. Default ``"audio_filepath"``.
    """

    events_key: str = "predicted_events"
    filepath_key: str = "audio_filepath"

    name: str = "SegmentExtractor"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.filepath_key, "segment_start", "segment_end", "segment_idx"]

    def ray_stage_spec(self) -> dict[str, Any]:
        from nemo_curator.backends.experimental.ray_data.utils import RayStageSpecKeys

        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, task: AudioTask) -> list[AudioTask]:
        events = task.data.get(self.events_key, [])
        if not events:
            return []

        output_tasks: list[AudioTask] = []
        for idx, event in enumerate(events):
            seg_data = dict(task.data)
            seg_data["segment_start"] = event["start_time"]
            seg_data["segment_end"] = event["end_time"]
            seg_data["segment_idx"] = idx
            seg_data["segment_confidence"] = event.get("mean_confidence", 0.0)

            seg_task = AudioTask(
                task_id=f"{task.task_id}_seg_{idx}",
                dataset_name=task.dataset_name,
                filepath_key=task.filepath_key or self.filepath_key,
                data=seg_data,
            )
            output_tasks.append(seg_task)

        return output_tasks
