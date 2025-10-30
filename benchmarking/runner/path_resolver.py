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

from pathlib import Path
from typing import Any


class PathResolver:
    """
    Resolves host/container paths loaded from a dictionary parsed from YAML,
    as in the 'paths' section of a YAML config.
    Example YAML config:
    paths:
      results_dir:
        container: /path/to/results
        host: /path/to/results
      artifacts_dir:
        container: /path/to/artifacts
        host: /path/to/artifacts
      datasets_dir:
        container: /path/to/datasets
        host: /path/to/datasets

    Resulting dictionary returned from yaml.safe_load() for the above 'paths' section:
    {
      "results_dir": {"container": "/path/to/results", "host": "/path/to/results"},
      "artifacts_dir": {"container": "/path/to/artifacts", "host": "/path/to/artifacts"},
      "datasets_dir": {"container": "/path/to/datasets", "host": "/path/to/datasets"},
    }
    """

    def __init__(self, paths_dict: dict[str, dict[str, Any]]) -> None:
        """
        :param paths_dict: dictionary mapping dir_type to dicts containing 'container' and/or 'host'
        """
        self.paths_dict: dict[str, dict[str, Any]] = paths_dict

    def resolve(self, dir_type: str) -> Path:
        """
        Given a directory type (e.g., 'results_dir'), return the first
        existing path among 'container' and 'host'. Checks 'container' first, then 'host'.
        Returns the path (Path) if found, else raises FileNotFoundError.
        """
        if dir_type not in self.paths_dict:
            msg = f"Unknown dir_type: {dir_type}, expected one of: {', '.join(self.paths_dict.keys())}"
            raise ValueError(msg)

        dvals: dict[str, Any] = self.paths_dict[dir_type]
        for key in ("container", "host"):
            path = Path(dvals.get(key))
            if path and path.exists():
                return path

        msg = f"No existing path found for '{dir_type}'. Checked: "
        msg += f"container={dvals.get('container')}, host={dvals.get('host')}"
        raise FileNotFoundError(msg)
