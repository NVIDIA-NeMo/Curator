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

from nemo_curator.stages.text.classifiers.utils import SortByLengthStage
from nemo_curator.stages.text.models.utils import SEQ_ORDER_FIELD
from nemo_curator.tasks import DocumentBatch


class TestSortByLengthStage:
    def test_process(self):
        batch = DocumentBatch(
            task_id="test",
            dataset_name="test",
            data=pd.DataFrame({"attention_mask": [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]}),
        )
        stage = SortByLengthStage()
        result = stage.process(batch)
        assert SEQ_ORDER_FIELD in result.data.columns
        assert result.data[SEQ_ORDER_FIELD].tolist() == [2, 0, 1]

    def test_process_no_op(self):
        batch = DocumentBatch(
            task_id="test",
            dataset_name="test",
            data=pd.DataFrame(
                # Set the SEQ_ORDER_FIELD to a random order
                {"attention_mask": [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], SEQ_ORDER_FIELD: [1, 2, 0]}
            ),
        )
        stage = SortByLengthStage()
        result = stage.process(batch)
        # Check that the data is not modified
        assert result.data.equals(batch.data)
