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

CONTAINER_RESULTS_DIR = "/results"
CONTAINER_ARTIFACTS_DIR = "/artifacts"
CONTAINER_DATASETS_DIR = "/datasets"


class PathResolver:
    """
    Resolves host/container paths for results, artifacts, and datasets.
    """

    def __init__(self, data: dict) -> None:
        """
        :param data: dictionary containing the paths for results, artifacts, and datasets
        """
        # Determine if running inside a Docker container
        # This is a commonly-used heuristic: /proc/1/cgroup often includes 'docker' or 'kubepods' in containers
        in_docker = False
        try:
            with open("/proc/1/cgroup") as f:
                contents = f.read()
                if "docker" in contents or "kubepods" in contents:
                    self.in_docker = True
        except FileNotFoundError:
            # If /proc/1/cgroup doesn't exist, assume not in Docker
            in_docker = False

        if in_docker:
            self.path_map = {
                "results_path": CONTAINER_RESULTS_DIR,
                "artifacts_path": CONTAINER_ARTIFACTS_DIR,
                "datasets_path": CONTAINER_DATASETS_DIR,
            }
        else:
            self.path_map = {
                "results_path": data["results_path"],
                "artifacts_path": data["artifacts_path"],
                "datasets_path": data["datasets_path"],
            }

    def resolve(self, dir_type: str) -> Path:
        """
        Given a directory type (e.g., 'results_path'), return the path.
        """
        if dir_type not in self.path_map:
            msg = f"Unknown dir_type: {dir_type}"
            raise ValueError(msg)

        return Path(self.path_map[dir_type])
