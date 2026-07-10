# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic stages for returning atomic dispatch batches to ordinary task rows."""

from dataclasses import dataclass
from typing import Any

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DispatchBatchTask, Task


@dataclass
class DispatchBatchUnpackStage(ProcessingStage[DispatchBatchTask, Task[Any]]):
    """Fan one processed dispatch envelope back out into its child tasks."""

    _curator_pipeline_helper_stage = True

    name: str = "dispatch_batch_unpack"

    def __post_init__(self) -> None:
        self.batch_size = 1

    def process(self, task: DispatchBatchTask) -> list[Task[Any]]:
        if not isinstance(task, DispatchBatchTask):
            msg = f"{type(self).__name__} expected DispatchBatchTask, got {type(task).__name__}"
            raise TypeError(msg)
        return task.flattened_items()

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {}
