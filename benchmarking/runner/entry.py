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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runner.datasets import DatasetResolver
    from runner.path_resolver import PathResolver


@dataclass
class Entry:
    name: str
    script: str | None = None
    args: str | None = None
    script_base_dir: Path = Path(__file__).parent.parent / "scripts"
    timeout_s: int | None = None
    sink_data: list[dict[str, Any]] | dict[str, Any] = field(default_factory=dict)
    requirements: list[dict[str, Any]] | dict[str, Any] = field(default_factory=dict)
    ray: dict[str, Any] = field(default_factory=dict)  # supports only single node: num_cpus,num_gpus,object_store_gb
    # If set, overrides the session-level default_object_store_size setting for this entry
    object_store_size_bytes: int | None = None
    # If set, overrides the session-level delete_scratch setting for this entry
    delete_scratch: bool | None = None
    enabled: bool = True

    def __post_init__(self) -> None:  # noqa: C901
        """Post-initialization checks and updates for dataclass."""
        # Convert the sink_data list of dicts to a dict of dicts for easier lookup with key from "name".
        # sink_data typically starts as a list of dicts from reading YAML, like this:
        # sink_data:
        #   - name: slack
        #     additional_metrics: ["num_documents_processed", "throughput_docs_per_sec"]
        #   - name: gdrive
        #     ...
        sink_data = {}
        # Will be a list of dicts if reading from YAML, in which case make it a dict of dicts with key
        # from "name" for easy lookup based on sink name.
        if isinstance(self.sink_data, list):
            for data in self.sink_data:
                sink_data[data["name"]] = data
        # If already a dict, use it directly and assume it is already in the correct format.
        elif isinstance(self.sink_data, dict):
            sink_data = self.sink_data
        else:
            msg = f"Invalid sink_data type: {type(self.sink_data)}"
            raise TypeError(msg)
        self.sink_data = sink_data

        # Convert the requirements list of dicts to a dict of dicts for easier lookup with key from "metric".
        # requirements typically starts as a list of dicts from reading YAML, like this:
        # requirements:
        #   - metric: throughput_docs_per_sec
        #     min_value: 200
        #   - metric: num_documents_processed
        #     ...
        requirements = {}
        # Will be a list of dicts if reading from YAML, in which case make it a dict of dicts with key
        # from "metric" for easy lookup based on metric name.
        if isinstance(self.requirements, list):
            for data in self.requirements:
                requirements[data["metric"]] = data
        # If already a dict, use it directly and assume it is already in the correct format.
        elif isinstance(self.requirements, dict):
            requirements = self.requirements
        else:
            msg = f"Invalid requirements type: {type(self.requirements)}"
            raise TypeError(msg)
        # For each requirement dict in requirements, check that if both min_value and max_value are present,
        # then max_value >= min_value. Raise ValueError if not.
        # Raise TypeError if req is not a dict.
        for metric_name, req in requirements.items():
            if not isinstance(req, dict):
                msg = f"Requirement for metric '{metric_name}' is not a dict: {type(req)}"
                raise TypeError(msg)
            has_min = "min_value" in req
            has_max = "max_value" in req
            if has_min and has_max:
                min_value = req["min_value"]
                max_value = req["max_value"]
                if max_value < min_value:
                    msg = f"Invalid requirement for metric '{metric_name}': max_value ({max_value}) < min_value ({min_value})"
                    raise ValueError(msg)
        self.requirements = requirements

    def get_command_to_run(
        self,
        session_entry_path: Path,
        path_resolver: PathResolver,
        dataset_resolver: DatasetResolver,
    ) -> str:
        if self.script:
            script_path = self.script_base_dir / self.script
            cmd = f"python {script_path} {self.args or ''}"

            cmd = self.substitute_paths_in_cmd(cmd, path_resolver, dataset_resolver)
            cmd = self.substitute_template_placeholders(cmd, session_entry_path)
        else:
            msg = f"Entry {self.name} must specify either cmd or script"
            raise ValueError(msg)

        return cmd

    def get_sink_data(self, sink_name: str) -> dict[str, Any]:
        return self.sink_data.get(sink_name, {})

    @staticmethod
    def substitute_paths_in_cmd(cmd: str, path_resolver: PathResolver, dataset_resolver: DatasetResolver) -> str:
        dataset_pattern = re.compile(r"\{dataset:([^,}]+),([^}]+)\}")

        def _replace_dataset(match: re.Match[str]) -> str:
            dataset_name = match.group(1).strip()
            dataset_format = match.group(2).strip()
            return str(dataset_resolver.resolve(dataset_name, dataset_format))

        path_pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

        def _replace_path(match: re.Match[str]) -> str:
            path_name = match.group(1).strip()
            # PathResolver.resolve() only matches specific paths intended to be mapped between host and container.
            # ValueError is raised if this is not one of those paths, meaning either the path is meant for template
            # substitution instead or possibly should be used as-is, in which case simply return the original string.
            try:
                return str(path_resolver.resolve(path_name))
            except ValueError:
                return match.group(0)

        return path_pattern.sub(_replace_path, dataset_pattern.sub(_replace_dataset, cmd))

    @staticmethod
    def substitute_template_placeholders(cmd: str, session_entry_path: Path) -> str:
        """Substitute template placeholders in command.
        Example:
        - {session_entry_dir}/results.json -> /path/to/session/entry/results.json
        """
        session_entry_pattern = re.compile(r"\{session_entry_dir\}")

        def replace_session_entry_path(match: re.Match[str]) -> str:  # noqa: ARG001
            return str(session_entry_path)

        return session_entry_pattern.sub(replace_session_entry_path, cmd)
