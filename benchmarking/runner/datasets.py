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

from __future__ import annotations


class DatasetResolver:
    def __init__(self, data: list[dict]) -> None:
        """
        Constructor for a DatasetResolver which accepts a list of dataset dictionaries.
        Dataset dictionaries are likely created from reading one or more YAML files.
        Example input:
        [ { "name": "my_dataset",
            "formats": [
              {"type": "parquet", "path": "/path/to/my_dataset.parquet"},
              {"type": "jsonl", "path": "/path/to/my_dataset.jsonl"},
            ]
           },
           ...]
        """
        self._map = {}

        # Check for duplicate dataset names before proceeding
        names = [d["name"] for d in data] if len(data) else []
        if len(names) != len(set(names)):
            duplicates = {name for name in names if names.count(name) > 1}
            msg = f"Duplicate dataset name(s) found: {', '.join(duplicates)}"
            raise ValueError(msg)

        for dataset in data:
            formats = dataset["formats"]
            if not isinstance(formats, list):
                msg = "formats must be a list"
                raise TypeError(msg)
            format_map = {}
            for fmt in formats:
                format_map[fmt["type"]] = fmt["path"]
            self._map[dataset["name"]] = format_map

    def resolve(self, dataset_name: str, file_format: str) -> str:
        if dataset_name not in self._map:
            msg = f"Unknown dataset: {dataset_name}"
            raise KeyError(msg)
        formats = self._map[dataset_name]
        if file_format not in formats:
            msg = f"Unknown format '{file_format}' for dataset '{dataset_name}'"
            raise KeyError(msg)
        return formats[file_format]
