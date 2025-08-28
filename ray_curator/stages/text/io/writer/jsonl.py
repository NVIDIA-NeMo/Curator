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
from typing import Any

from nemo_curator.tasks import DocumentBatch

from .base import BaseWriter


@dataclass
class JsonlWriter(BaseWriter):
    """Writer that writes a DocumentBatch to a JSONL file."""

    # Additional kwargs for pandas.DataFrame.to_json
    file_extension: str = "jsonl"
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    _name: str = "jsonl_writer"

    def write_data(self, task: DocumentBatch, file_path: str) -> None:
        """Write data to JSONL file using pandas DataFrame.to_json."""
        df = task.to_pandas()  # Convert to pandas DataFrame if needed
        # Filter fields if specified
        if self.fields is not None:
            df = df[self.fields]

        # Build kwargs for to_json with explicit options
        write_kwargs = {
            "lines": True,
            "orient": "records",
        }

        # Add any additional kwargs, allowing them to override defaults
        write_kwargs.update(self.write_kwargs)
        df.to_json(file_path, **write_kwargs)
