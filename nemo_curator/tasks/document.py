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
import pyarrow as pa
from loguru import logger

from .tasks import Task


@dataclass
class DocumentBatch(Task[pa.Table | pd.DataFrame]):
    """
    Task for processing batches of text documents.
    Documents are stored as a dataframe (PyArrow Table or Pandas DataFrame).

    """

    data: pa.Table | pd.DataFrame = field(default_factory=pa.Table)

    def to_pyarrow(self) -> pa.Table:
        """Convert data to PyArrow table."""
        if isinstance(self.data, pa.Table):
            return self.data
        elif isinstance(self.data, pd.DataFrame):
            return pa.Table.from_pandas(self.data)
        else:
            msg = f"Cannot convert {type(self.data)} to PyArrow table"
            raise TypeError(msg)

    def to_pandas(self) -> pd.DataFrame:
        """Convert data to Pandas DataFrame."""
        if isinstance(self.data, pd.DataFrame):
            return self.data
        elif isinstance(self.data, pa.Table):
            return self.data.to_pandas()
        else:
            msg = f"Cannot convert {type(self.data)} to Pandas DataFrame"
            raise TypeError(msg)

    @property
    def num_items(self) -> int:
        """Get the number of documents in this batch."""
        return len(self.data)

    def get_columns(self) -> list[str]:
        """Get column names from the data."""
        if isinstance(self.data, pd.DataFrame):
            return list(self.data.columns)
        elif isinstance(self.data, pa.Table):
            return self.data.column_names
        else:
            msg = f"Unsupported data type: {type(self.data)}"
            raise TypeError(msg)

    def validate(self) -> bool:
        """Validate the task data."""

        if self.num_items <= 0:
            logger.warning(f"Task {self.task_id} has no items")
            return False

        if not self.get_columns():
            logger.warning(f"Task {self.task_id} could not find any columns in the data")
            return False

        return True
