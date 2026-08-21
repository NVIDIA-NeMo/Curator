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
import pyarrow as pa
import pytest

from nemo_curator.tasks import DocumentBatch


@pytest.mark.parametrize("string_type", [pa.string(), pa.large_string()])
def test_to_pandas_preserves_arrow_string_storage(string_type: pa.DataType) -> None:
    """Converting an Arrow batch should not materialize strings as Python objects."""
    table = pa.table(
        {
            "text": pa.array(["first", None, "third"], type=string_type),
            "score": pa.array([1, 2, 3], type=pa.int64()),
        }
    )

    dataframe = DocumentBatch(dataset_name="test", data=table).to_pandas()

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe["text"].dtype.storage == "pyarrow"
    assert dataframe["text"].tolist()[::2] == ["first", "third"]
    assert pd.isna(dataframe["text"].iloc[1])
    assert dataframe["score"].dtype == "int64"
