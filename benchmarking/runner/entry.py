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
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from runner.utils import get_total_memory_bytes

if TYPE_CHECKING:
    from runner.datasets import DatasetResolver
    from runner.path_resolver import PathResolver

_curator_repo_path = Path(__file__).parent.parent.parent
_entry_script_base_path = _curator_repo_path / "benchmarking/scripts"


@dataclass
class Entry:
    name: str
    script: str | None = None
    args: str | None = None
    script_base_path: Path = _entry_script_base_path
    timeout_s: int | None = None
    sink_data: list[dict[str, Any]] | dict[str, Any] = field(default_factory=dict)
    requirements: list[dict[str, Any]] | dict[str, Any] = field(default_factory=dict)
    ray: dict[str, Any] = field(default_factory=dict)  # supports only single node: num_cpus,num_gpus,object_store_gb
    # If set, overrides the session-level object_store_size setting for this entry
    # Value will be either number of bytes (int), fraction of system memory (float), or None or "default" (string) both
    # representing the default object store size as used by "ray start".
    object_store_size: int | float | str | None = None
    # If set, overrides the session-level delete_scratch setting for this entry
    delete_scratch: bool | None = None

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Post-initialization checks and updates for dataclass."""
        # Process object_store_size by converting values representing fractions of system memory to bytes.
        if isinstance(self.object_store_size, float):
            self.object_store_size = int(get_total_memory_bytes() * self.object_store_size)

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
        # For each requirement dict in requirements, ensure that max_value >= min_value if both are present.
        for metric_name, req in requirements.items():
            if not isinstance(req, dict):
                msg = f"Requirement for metric '{metric_name}' is not a dict: {type(req)}"
                raise TypeError(msg)
            has_exact = "exact_value" in req
            has_min = "min_value" in req
            has_max = "max_value" in req
            if has_exact and (has_min or has_max):
                msg = (
                    f"Invalid requirement for metric '{metric_name}': 'exact_value' "
                    f"cannot be combined with 'min_value' or 'max_value'."
                )
                raise ValueError(msg)
            if has_min and has_max:
                min_value = req["min_value"]
                max_value = req["max_value"]
                if max_value < min_value:
                    msg = f"Invalid requirement for metric '{metric_name}': max_value ({max_value}) < min_value ({min_value})"
                    raise ValueError(msg)
        self.requirements = requirements

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entry:
        """Create Entry from dict, ignoring extra keys.

        Args:
            data: Dictionary containing entry configuration data.

        Returns:
            Entry instance with only valid fields populated.
        """
        # Get only the fields that are defined in the dataclass
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def get_command_to_run(
        self,
        session_entry_path: Path,
        path_resolver: PathResolver,
        dataset_resolver: DatasetResolver,
    ) -> str:
        if self.script:
            script = self.script
            script = self.substitute_reserved_placeholders(script, session_entry_path, dataset_resolver)
            script = self.substitute_container_or_host_paths(script, path_resolver, allow_other_placeholders=False)

            # Using the Path "/" operator means that if script is an abs path here then
            # self.script_base_path will be ignored automatically.
            script_path = self.script_base_path / script
            cmd = f"python {script_path} {self.args or ''}"

            cmd = self.substitute_reserved_placeholders(cmd, session_entry_path, dataset_resolver)
            cmd = self.substitute_container_or_host_paths(cmd, path_resolver, allow_other_placeholders=False)
        else:
            msg = f"Entry {self.name} must specify a script to run"
            raise ValueError(msg)

        return cmd

    def get_sink_data(self, sink_name: str) -> dict[str, Any]:
        return self.sink_data.get(sink_name, {})

    @staticmethod
    def substitute_container_or_host_paths(
        cmd: str, path_resolver: PathResolver, allow_other_placeholders: bool = True
    ) -> str:
        """
        Substitute paths in the command string that are intended to be resolved by PathResolver.

        This replaces placeholders in the form {path_name} with their corresponding host or container path.
        If the placeholder does not correspond to a known path (as defined in PathResolver), it is left unchanged.
        If allow_other_placeholders is False, then ValueError is raised if the placeholder does not correspond to a recognized path type.
        """
        path_pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

        def _replace_path(match: re.Match[str]) -> str:
            path_name = match.group(1).strip()
            # PathResolver.resolve() only matches specific paths intended to be mapped between host and container.
            # ValueError is raised if the placeholder (word inside {}) is not one of those paths.
            try:
                return str(path_resolver.resolve(path_name))
            except ValueError as e:
                if allow_other_placeholders:
                    return match.group(0)
                else:
                    msg = f"Unknown path placeholder: {path_name}"
                    raise ValueError(msg) from e

        return path_pattern.sub(_replace_path, cmd)

    @staticmethod
    def substitute_reserved_placeholders(cmd: str, session_entry_path: Path, dataset_resolver: DatasetResolver) -> str:
        """Substitute reserved placeholders in command.
        Example:
        - {session_entry_dir}/results.json -> /path/to/session/entry/results.json
        """
        session_entry_dir_pattern = re.compile(r"\{session_entry_dir\}")

        def _replace_session_entry_dir(match: re.Match[str]) -> str:  # noqa: ARG001
            return str(session_entry_path)

        curator_repo_dir_pattern = re.compile(r"\{curator_repo_dir\}")

        def _replace_curator_repo_dir(match: re.Match[str]) -> str:  # noqa: ARG001
            return str(_curator_repo_path)

        dataset_pattern = re.compile(r"\{dataset:([^,}]+),([^}]+)\}")

        def _replace_dataset(match: re.Match[str]) -> str:
            dataset_name = match.group(1).strip()
            dataset_format = match.group(2).strip()
            return str(dataset_resolver.resolve(dataset_name, dataset_format))

        new_cmd = cmd
        new_cmd = session_entry_dir_pattern.sub(_replace_session_entry_dir, new_cmd)
        new_cmd = curator_repo_dir_pattern.sub(_replace_curator_repo_dir, new_cmd)
        new_cmd = dataset_pattern.sub(_replace_dataset, new_cmd)
        return new_cmd  # noqa: RET504
