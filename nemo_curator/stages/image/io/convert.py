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

from dataclasses import dataclass, field

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, ImageBatch


@dataclass
class ConvertImageBatchToDocumentBatchStage(ProcessingStage[ImageBatch, DocumentBatch]):
    """
    Convert image batch to DocumentBatch

    Args:
        fields: list of fields of ImageObject to convert to DocumentBatch
    """
    fields: list[str] = field(default_factory=list)
    _name: str = "convert_image_batch_to_document_batch"

    def process(self, task: ImageBatch) -> DocumentBatch:
        """
        Convert image batch to DocumentBatch
        """
        data = {}
        if self.fields:
            for field in self.fields:
                data[field] = [getattr(image_obj, field, None) for image_obj in task.data]
        else:
            # Default to image_id if no fields specified
            data["image_id"] = [image_obj.image_id for image_obj in task.data]
        df = pd.DataFrame(data)

        return DocumentBatch(
            task_id=f"{task.task_id}_{self.name}",
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
