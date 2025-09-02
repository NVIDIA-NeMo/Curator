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

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch, DocumentBatch


class AudioToDocumentStage(ProcessingStage[AudioBatch, DocumentBatch]):
    """
    Stage to conver DocumentObject to DocumentBatch

    """

    def process(self, task: AudioBatch) -> list[DocumentBatch]:
        return [
            DocumentBatch(
                data=pd.DataFrame(task.data),
                task_id=task.task_id,
                dataset_name=task.dataset_name,
            )
        ]
