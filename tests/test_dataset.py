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

from nemo_curator.datasets import DocumentDataset


def test_to_from_pandas() -> None:
    original_df = pd.DataFrame({"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]})
    dataset = DocumentDataset.from_pandas(original_df)
    converted_df = dataset.to_pandas()
    pd.testing.assert_frame_equal(original_df, converted_df)
