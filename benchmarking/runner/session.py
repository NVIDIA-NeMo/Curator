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

from loguru import logger

if TYPE_CHECKING:
    from runner.sinks.sink import Sink
from runner.datasets import DatasetResolver
from runner.entry import Entry
from runner.path_resolver import PathResolver
from runner.utils import get_total_memory_bytes


@dataclass(frozen=True, kw_only=True)
class Session:
    results_path: Path
    entries: list[Entry] = field(default_factory=list)
    sinks: list[Sink] = field(default_factory=list)
    default_timeout_s: int = 7200
    # Set object store memory to 50% of total system memory by default
    default_object_store_size_bytes: int = int(get_total_memory_bytes() * 0.5)
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

        # Update delete_scratch for each entry that has not been set to the session-level delete_scratch setting
        for entry in self.entries:
            if entry.delete_scratch is None:
                entry.delete_scratch = self.delete_scratch

        # Update timeout_s for each entry that has not been set to the session-level default_timeout_s
        for entry in self.entries:
            if entry.timeout_s is None:
                entry.timeout_s = self.default_timeout_s

        # Update object store size for each entry that has not been set to the session-level default_object_store_size setting
        for entry in self.entries:
            if entry.object_store_size_bytes is None:
                entry.object_store_size_bytes = self.default_object_store_size_bytes

    @classmethod
    def assert_valid_config_dict(cls, data: dict) -> None:
        """Assert that the configuration contains the minimum required config values."""
        required_fields = ["results_path", "datasets_path", "entries"]
        missing_fields = [k for k in required_fields if k not in data]
        if missing_fields:
            msg = f"Invalid configuration: missing required fields: {missing_fields}"
            raise ValueError(msg)

    @classmethod
    def create_from_dict(cls, data: dict) -> Session:
        """
        Factory method to create a Session from a dictionary.

        The dictionary is typically created from reading one or more YAML files.
        This method resolves environment variables and converts the list of
        entry dicts to Entry objects, and returns a new Session
        object.
        """
        path_resolver = PathResolver(data)
        dataset_resolver = DatasetResolver(data.get("datasets", []))

        # Filter out data not needed for a Session object.
        mc_field_names = {f.name for f in fields(cls)}
        mc_data = {k: v for k, v in data.items() if k in mc_field_names}
        sinks = cls.create_sinks_from_dict(mc_data.get("sinks", []))
        # Load entries only if enabled (enabled by default)
        # TODO: should entries be created unconditionally and have an "enabled" field instead?
        entries = [Entry(**e) for e in mc_data["entries"] if e.get("enabled", True)]

        mc_data["results_path"] = path_resolver.resolve("results_path")
        mc_data["entries"] = entries
        mc_data["sinks"] = sinks
        mc_data["path_resolver"] = path_resolver
        mc_data["dataset_resolver"] = dataset_resolver

        return cls(**mc_data)

    @classmethod
    def create_sinks_from_dict(cls, sink_configs: list[dict]) -> list[Sink]:
        """Load sinks from the list of sink configuration dictionaries."""
        sinks = []
        for sink_config in sink_configs:
            sink_name = sink_config["name"]
            sink_enabled = sink_config.get("enabled", True)
            if not sink_enabled:
                logger.warning(f"Sink {sink_name} is not enabled, skipping")
                continue
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
