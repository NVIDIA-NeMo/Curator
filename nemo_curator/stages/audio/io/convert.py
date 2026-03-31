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

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, DocumentBatch


class AudioToDocumentStage(ProcessingStage[AudioTask, DocumentBatch]):
    """Convert AudioTask entries into DocumentBatch DataFrames.

    Overrides ``process_batch`` to aggregate an entire batch of
    ``AudioTask`` objects into a single multi-row ``DocumentBatch``,
    avoiding the overhead of many single-row DataFrames.  Set
    ``batch_size`` to control how many audio entries land in each
    DataFrame (default 64).
    """

    name = "AudioToDocumentStage"
    batch_size: int = 64

    def process(self, task: AudioTask) -> DocumentBatch:
        msg = "AudioToDocumentStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[DocumentBatch]:
        if len(tasks) == 0:
            return []
        df = pd.DataFrame([t.data for t in tasks])
        perf = []
        for t in tasks:
            perf.extend(t._stage_perf)
        return [
            DocumentBatch(
                data=df,
                task_id=",".join(t.task_id for t in tasks),
                dataset_name=",".join(dict.fromkeys(t.dataset_name for t in tasks)),
                _stage_perf=perf,
            )
        ]
