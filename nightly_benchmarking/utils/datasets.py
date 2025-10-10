from __future__ import annotations

import json


class DatasetResolver:
    def __init__(self, dataset_paths_json: str) -> None:
        with open(dataset_paths_json) as f:
            self._map = json.load(f)

    def resolve(self, dataset_name: str, file_format: str) -> str:
        if dataset_name not in self._map:
            msg = f"Unknown dataset: {dataset_name}"
            raise KeyError(msg)
        formats = self._map[dataset_name]
        if file_format not in formats:
            msg = f"Unknown format '{file_format}' for dataset '{dataset_name}'"
            raise KeyError(msg)
        return formats[file_format]
