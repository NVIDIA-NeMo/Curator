from __future__ import annotations


class DatasetResolver:
    def __init__(self) -> None:
        self._map = {}

    @classmethod
    def create_from_dicts(cls, data: list[dict]) -> DatasetResolver:
        """
        Factory method to create a DatasetResolver from a list of dataset dictionaries.
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
        # Check for duplicate dataset names before proceeding
        names = [d["name"] for d in data]
        if len(names) != len(set(names)):
            duplicates = set([name for name in names if names.count(name) > 1])
            raise ValueError(f"Duplicate dataset name(s) found: {', '.join(duplicates)}")

        instance = cls()
        for dataset in data:
            formats = dataset["formats"]
            assert isinstance(formats, list), "formats must be a list"
            format_map = {}
            for fmt in formats:
                format_map[fmt["type"]] = fmt["path"]
            instance._map[dataset["name"]] = format_map
        return instance


    def resolve(self, dataset_name: str, file_format: str) -> str:
        if dataset_name not in self._map:
            msg = f"Unknown dataset: {dataset_name}"
            raise KeyError(msg)
        formats = self._map[dataset_name]
        if file_format not in formats:
            msg = f"Unknown format '{file_format}' for dataset '{dataset_name}'"
            raise KeyError(msg)
        return formats[file_format]
