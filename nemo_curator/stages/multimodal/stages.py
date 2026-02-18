# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import MultiBatchTask


@dataclass
class BasicMultimodalFilterStage(ProcessingStage[MultiBatchTask, MultiBatchTask]):
    """Minimal validation/filter stage for multimodal rows."""

    drop_invalid_rows: bool = True
    name: str = "basic_multimodal_filter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: MultiBatchTask) -> MultiBatchTask:
        df = task.to_pandas().copy()
        if df.empty:
            return task

        if self.drop_invalid_rows:
            allowed = {"text", "image", "metadata"}
            df = df[df["modality"].isin(allowed)]
            # Keep metadata rows at sentinel position -1; content rows should be non-negative.
            valid_pos = (df["modality"] == "metadata") & (df["position"] == -1)
            valid_pos = valid_pos | ((df["modality"] != "metadata") & (df["position"] >= 0))
            df = df[valid_pos]

        return MultiBatchTask(
            task_id=f"{task.task_id}_{self.name}",
            dataset_name=task.dataset_name,
            data=df.reset_index(drop=True),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
