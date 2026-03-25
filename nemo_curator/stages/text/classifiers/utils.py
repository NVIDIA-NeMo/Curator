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

import numpy as np

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_FIELD, SEQ_ORDER_FIELD, TOKEN_LENGTH_FIELD
from nemo_curator.tasks import DocumentBatch

DEBERTA_TOKENIZER_PADDING_SIDE = "right"


class SortByLengthStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Stage that sorts the input data by the length of the input tokens.
    """

    def __init__(self):
        self.name = "sort_by_length_stage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [ATTENTION_MASK_FIELD]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [ATTENTION_MASK_FIELD, SEQ_ORDER_FIELD]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if SEQ_ORDER_FIELD in batch.data.columns:
            return batch

        output = batch.data.copy()

        # Add column to preserve original order
        output[SEQ_ORDER_FIELD] = np.arange(len(output))
        output[TOKEN_LENGTH_FIELD] = output[ATTENTION_MASK_FIELD].map(sum)
        output = output.sort_values(by=TOKEN_LENGTH_FIELD, kind="stable", ignore_index=True).drop(
            columns=[TOKEN_LENGTH_FIELD]
        )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=output,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
