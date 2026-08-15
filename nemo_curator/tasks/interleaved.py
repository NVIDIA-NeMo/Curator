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

"""Interleaved task type and schema for row-wise interleaved multimodal records.

Schema columns fall into two categories:

**Reserved columns** (``RESERVED_COLUMNS``) -- managed by pipeline stages:

    ==================  =============  ===========  ===============================================
    Column              Type           Category     Description
    ==================  =============  ===========  ===============================================
    ``sample_id``       string (req)   Identity     Unique document/sample identifier
    ``position``        int32 (req)    Identity     Position within sample (-1 for metadata rows)
    ``modality``        string (req)   Identity     Row modality -- built-in values are ``text``,
                                                   ``image``, and ``metadata``; extensible to
                                                   ``audio``, ``table``, ``generated_image``, etc.
    ``content_type``    string         Content      MIME type (e.g. ``text/plain``, ``image/jpeg``)
    ``text_content``    string         Content      Text payload for text rows
    ``binary_content``  large_binary   Content      Image bytes (populated by materialization)
    ``source_ref``      FILE           Internal     Parquet FILE-compatible reference
    ``source_member``   string         Internal     Archive member metadata adjacent to FILE
    ``source_frame_index`` int32        Internal     Multi-frame index metadata adjacent to FILE
    ``materialize_error`` string       Internal     Error message if materialization failed
    ==================  =============  ===========  ===============================================

**User columns** (passthrough) -- extra fields from source data added via the
``fields`` parameter on the reader. These flow through the pipeline untouched.
"""

from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from nemo_curator.utils.storage_utils import FILE_REFERENCE_TYPE

from .tasks import Task

INTERLEAVED_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("content_type", pa.string(), nullable=True),
        pa.field("text_content", pa.string(), nullable=True),
        pa.field("binary_content", pa.large_binary(), nullable=True),
        pa.field("source_ref", FILE_REFERENCE_TYPE, nullable=True),
        pa.field("source_member", pa.string(), nullable=True),
        pa.field("source_frame_index", pa.int32(), nullable=True),
        pa.field("materialize_error", pa.string(), nullable=True),
    ]
)

RESERVED_COLUMNS: frozenset[str] = frozenset(INTERLEAVED_SCHEMA.names)


@dataclass
class InterleavedBatch(Task[pa.Table | pd.DataFrame]):
    """Task carrying row-wise multimodal records.

    See module docstring for the full schema reference (reserved vs user columns).
    """

    REQUIRED_COLUMNS: frozenset[str] = frozenset(
        name for name, f in zip(INTERLEAVED_SCHEMA.names, INTERLEAVED_SCHEMA, strict=True) if not f.nullable
    )

    data: pa.Table | pd.DataFrame = field(default_factory=lambda: pa.Table.from_pylist([], schema=INTERLEAVED_SCHEMA))

    # -- conversion --

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
            return self.data.to_pandas(types_mapper=pd.ArrowDtype)
        msg = f"Cannot convert {type(self.data)} to Pandas DataFrame"
        raise TypeError(msg)

    # -- introspection --

    @property
    def num_items(self) -> int:
        """Number of unique samples (distinct ``sample_id`` values)."""
        if isinstance(self.data, pa.Table):
            return pc.count_distinct(self.data.column("sample_id")).as_py()
        return int(self.data["sample_id"].nunique())

    def count(self, *, modality: str | None = None) -> int:
        """Return row count, optionally filtered by modality.

        Examples::

            task.count()                    # total rows
            task.count(modality="image")    # image rows only
            task.count(modality="text")     # text rows only
        """
        if modality is None:
            return len(self.data)
        if isinstance(self.data, pa.Table):
            return pc.sum(pc.equal(self.data.column("modality"), modality)).as_py()
        return int((self.data["modality"] == modality).sum())

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
        columns = set(self.get_columns())
        missing = sorted(self.REQUIRED_COLUMNS - columns)
        if missing:
            logger.warning(f"Task {self.task_id} missing required columns: {missing}")
            return False
        return True

    # -- mutation (not yet implemented) --

    def add_rows(
        self,
        rows: pa.Table | pd.DataFrame | list[dict],
        sample_id: str | None = None,
        auto_position: bool = True,
    ) -> "InterleavedBatch":
        """Add rows to this task.

        Args:
            rows: New rows to append. Must contain required columns unless
                overridden by *sample_id* / *auto_position*.
            sample_id: If provided, assign this ``sample_id`` to all new rows.
            auto_position: If ``True``, auto-assign ``position`` values
                continuing from the existing maximum per sample.
        """
        raise NotImplementedError

    def delete_rows(self, mask: pd.Series) -> "InterleavedBatch":
        """Delete rows where *mask* is ``True``.

        Args:
            mask: Boolean Series aligned to the data. ``True`` marks a row
                for deletion.
        """
        raise NotImplementedError

    # -- source_ref helpers --

    @staticmethod
    def build_source_ref(
        uri: str | None,
        offset: int | None = None,
        size: int | None = None,
    ) -> dict[str, object] | None:
        """Build a Parquet FILE-compatible external reference."""
        if (offset is not None and size is None) or any(value is not None and value < 0 for value in (offset, size)):
            msg = "source_ref offset and size must be non-negative, with size set for offset"
            raise ValueError(msg)
        return (
            {
                "uri": uri,
                "offset": offset,
                "size": size,
                "content_type": None,
                "checksum": None,
                "inline": None,
            }
            if uri
            else None
        )

    def with_parsed_source_ref_columns(self, prefix: str = "_src_") -> pd.DataFrame:
        """Return a DataFrame with FILE locator and adjacent source columns expanded."""
        df = self.to_pandas().copy()
        parsed = pd.DataFrame.from_records(
            (value if isinstance(value, dict) else {} for value in df["source_ref"]),
            columns=["uri", "offset", "size"],
        )
        for col in parsed:
            df[f"{prefix}{col}"] = parsed[col].to_numpy(copy=False)
        df[f"{prefix}member"] = df.get("source_member")
        df[f"{prefix}frame_index"] = df.get("source_frame_index")
        return df
