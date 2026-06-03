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

"""Fan out one audio task into one task per SED-detected segment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class SegmentExtractorStage(ProcessingStage[AudioTask, AudioTask]):
    """Fan out an ``AudioTask`` into one output task per SED event.

    This is the in-pipeline fan-out counterpart to
    ``io.extract_segments.SegmentExtractionStage``. It does not write audio
    clips. It copies the input task data and attaches segment timestamps from
    ``events_key`` to each emitted task.
    """

    name: str = "SegmentExtractor"
    events_key: str = "sed_events"
    filepath_key: str = "audio_filepath"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.events_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key, "segment_start", "segment_end", "segment_idx", "segment_confidence"]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, task: AudioTask) -> list[AudioTask]:
        events = task.data.get(self.events_key, [])
        if not events:
            self._log_metrics({"utterances_input": 1.0, "segments_emitted": 0.0})
            return []

        output_tasks: list[AudioTask] = []
        for idx, event in enumerate(events):
            seg_data = dict(task.data)
            seg_data["segment_start"] = event["start_time"]
            seg_data["segment_end"] = event["end_time"]
            seg_data["segment_idx"] = idx
            seg_data["segment_confidence"] = event.get("mean_confidence", 0.0)

            output_tasks.append(
                AudioTask(
                    task_id=f"{task.task_id}_seg_{idx}",
                    dataset_name=task.dataset_name,
                    filepath_key=task.filepath_key or self.filepath_key,
                    data=seg_data,
                    _metadata=dict(task._metadata),
                    _stage_perf=list(task._stage_perf),
                )
            )

        self._log_metrics({
            "utterances_input": 1.0,
            "segments_emitted": float(len(output_tasks)),
        })
        return output_tasks
