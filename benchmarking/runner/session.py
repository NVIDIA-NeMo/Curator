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

# ruff: noqa: ERA001

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Use pytest's expression eval code to support "-k" style matching.
# TODO: This adds a dependency on a pytest internal module.
#       Consider vendoring the pytest code or implementing a custom expression evaluator.
from _pytest.mark import Expression
from loguru import logger

if TYPE_CHECKING:
    from runner.sinks.sink import Sink
from runner.datasets import DatasetResolver
from runner.entry import Entry
from runner.path_resolver import PathResolver
from runner.utils import get_total_memory_bytes


@dataclass(kw_only=True)
class Session:
    results_path: Path
    entries: list[Entry] = field(default_factory=list)
    sinks: list[Sink] = field(default_factory=list)
    default_timeout_s: int = 7200
    # object store size is either a value in bytes (int), a fraction of total system memory (float), or None or the
    # value "default" (string) both representing the default object store size as used by "ray start".
    object_store_size: int | float | str | None = 0.5
    # Whether to delete the entry's scratch directory after completion by default
    delete_scratch: bool = True
    path_resolver: PathResolver = None
    dataset_resolver: DatasetResolver = None

    def __post_init__(self) -> None:
        """Post-initialization checks and updates for dataclass."""
        names = [entry.name for entry in self.entries]
        if len(names) != len(set(names)):
            duplicates = {name for name in names if names.count(name) > 1}
            msg = f"Duplicate entry name(s) found: {', '.join(duplicates)}"
            raise ValueError(msg)

        # Process object_store_size by converting values representing fractions of system memory to bytes.
        if isinstance(self.object_store_size, float):
            self.object_store_size = int(get_total_memory_bytes() * self.object_store_size)

        # Update delete_scratch for each entry that has not been set to the session-level delete_scratch setting
        for entry in self.entries:
            if entry.delete_scratch is None:
                entry.delete_scratch = self.delete_scratch

        # Update timeout_s for each entry that has not been set to the session-level default_timeout_s
        for entry in self.entries:
            if entry.timeout_s is None:
                entry.timeout_s = self.default_timeout_s

        # Update object store size for each entry that has not been set.
        for entry in self.entries:
            if entry.object_store_size is None:
                entry.object_store_size = self.object_store_size

    @classmethod
    def assert_valid_config_dict(cls, data: dict) -> None:
        """Assert that the configuration contains the minimum required config values."""
        required_fields = ["results_path", "datasets_path", "model_weights_path", "entries"]
        missing_fields = [k for k in required_fields if k not in data]
        if missing_fields:
            msg = f"Invalid configuration: missing required fields: {missing_fields}"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict, entry_filter_expr: str | None = None) -> Session:
        """
        Factory method to create a Session from a dictionary.

        The dictionary is typically created from reading one or more YAML files.
        This method resolves environment variables and converts the list of
        entry dicts to Entry objects, and returns a new Session
        object.
        """
        cls.assert_valid_config_dict(data)
        path_resolver = PathResolver(data)
        dataset_resolver = DatasetResolver(data.get("datasets", []))

        # Filter out data not needed for a Session object.
        sess_field_names = {f.name for f in fields(cls)}
        sess_data = {k: v for k, v in data.items() if k in sess_field_names}
        sinks = cls.create_sinks_from_dict(sess_data.get("sinks", []))

        entries = [Entry.from_dict(e) for e in sess_data["entries"]]

        # Filter entries based on the expression, if provided.
        # Example: expr "foo and not foobar" will include all entries
        # with "foo" in the name but not "foobar".
        if entry_filter_expr is not None:
            filtered_entries = []
            expr = Expression.compile(entry_filter_expr)
            for entry in entries:

                def matcher(subname_in_expr: str, entry: Entry = entry) -> bool:
                    return subname_in_expr in entry.name.lower()

                if expr.evaluate(matcher):
                    filtered_entries.append(entry)
            entries = filtered_entries

        sess_data["results_path"] = path_resolver.resolve("results_path")
        sess_data["entries"] = entries
        sess_data["sinks"] = sinks
        sess_data["path_resolver"] = path_resolver
        sess_data["dataset_resolver"] = dataset_resolver

        return cls(**sess_data)

    @classmethod
    def create_sinks_from_dict(cls, sink_configs: list[dict]) -> list[Sink]:
        """Load sinks from the list of sink configuration dictionaries."""
        sinks = []
        for sink_config in sink_configs:
            sink_name = sink_config["name"]
            if sink_name == "mlflow":
                from runner.sinks.mlflow_sink import MlflowSink

                sinks.append(MlflowSink(sink_config=sink_config))
            elif sink_name == "slack":
                from runner.sinks.slack_sink import SlackSink

                sinks.append(SlackSink(sink_config=sink_config))
            elif sink_name == "gdrive":
                from runner.sinks.gdrive_sink import GdriveSink

                sinks.append(GdriveSink(sink_config=sink_config))
            else:
                logger.warning(f"Unknown sink: {sink_name}, skipping")
        return sinks
