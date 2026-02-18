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

import json
from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa
from loguru import logger

from .tasks import Task

MULTIMODAL_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("content_type", pa.string(), nullable=True),
        pa.field("text_content", pa.string(), nullable=True),
        pa.field("binary_content", pa.large_binary(), nullable=True),
        pa.field("metadata_source", pa.string(), nullable=True),
        pa.field("metadata_json", pa.string(), nullable=True),
        pa.field("materialize_error", pa.string(), nullable=True),
    ]
)


@dataclass
class MultiBatchTask(Task[pa.Table | pd.DataFrame]):
    """Task carrying row-wise multimodal records."""

    data: pa.Table | pd.DataFrame = field(default_factory=lambda: pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA))

    def to_pyarrow(self) -> pa.Table:
        if isinstance(self.data, pa.Table):
            return self.data
        if isinstance(self.data, pd.DataFrame):
            return pa.Table.from_pandas(self.data, preserve_index=False)
        msg = f"Cannot convert {type(self.data)} to PyArrow table"
        raise TypeError(msg)

    def to_pandas(self) -> pd.DataFrame:
        if isinstance(self.data, pd.DataFrame):
            return self.data
        if isinstance(self.data, pa.Table):
            # Strict mode: preserve Arrow-backed nullable/native types in pandas.
            return self.data.to_pandas(types_mapper=pd.ArrowDtype)
        msg = f"Cannot convert {type(self.data)} to Pandas DataFrame"
        raise TypeError(msg)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def get_columns(self) -> list[str]:
        if isinstance(self.data, pd.DataFrame):
            return list(self.data.columns)
        if isinstance(self.data, pa.Table):
            return self.data.column_names
        msg = f"Unsupported data type: {type(self.data)}"
        raise TypeError(msg)

    def validate(self) -> bool:
        if self.num_items <= 0:
            logger.warning(f"Task {self.task_id} has no items")
            return False
        required = {"sample_id", "position", "modality"}
        columns = set(self.get_columns())
        missing = sorted(required - columns)
        if missing:
            logger.warning(f"Task {self.task_id} missing required columns: {missing}")
            return False
        return True

    @staticmethod
    def build_metadata_source(
        source_id: str | None,
        source_shard: str | None,
        content_path: str | None,
        content_key: str | None,
    ) -> str:
        return json.dumps(
            {
                "source_id": source_id,
                "source_shard": source_shard,
                "content_path": content_path,
                "content_key": content_key,
            },
            ensure_ascii=True,
        )

    @staticmethod
    def parse_metadata_source(source_value: str | None) -> dict[str, str | None]:
        """Parse one metadata_source JSON string into a source locator dict."""
        if source_value is None or pd.isna(source_value):
            return {
                "source_id": None,
                "source_shard": None,
                "content_path": None,
                "content_key": None,
            }
        if source_value == "":
            return {
                "source_id": None,
                "source_shard": None,
                "content_path": None,
                "content_key": None,
            }
        parsed = json.loads(source_value)
        if not isinstance(parsed, dict):
            msg = "metadata_source must decode to a JSON object"
            raise TypeError(msg)
        return {
            "source_id": parsed.get("source_id"),
            "source_shard": parsed.get("source_shard"),
            "content_path": parsed.get("content_path"),
            "content_key": parsed.get("content_key"),
        }

    def with_parsed_source_columns(self, prefix: str = "_src_") -> pd.DataFrame:
        """Return a pandas view with parsed metadata source columns added.

        Added columns:
        - {prefix}source_id
        - {prefix}source_shard
        - {prefix}content_path
        - {prefix}content_key
        """
        df = self.to_pandas().copy()
        parsed = [self.parse_metadata_source(value) for value in df["metadata_source"].tolist()]
        parsed_df = pd.DataFrame.from_records(
            parsed,
            columns=["source_id", "source_shard", "content_path", "content_key"],
        )
        for col in parsed_df.columns:
            df[f"{prefix}{col}"] = parsed_df[col].to_numpy(copy=False)
        return df
