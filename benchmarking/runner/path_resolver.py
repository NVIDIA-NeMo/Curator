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

CONTAINER_CURATOR_DIR = "/opt/Curator"

# Currently all set to /MOUNT to make it obvious in log/error messages from the
# container that these paths are available on the host. The assumption is that
# /MOUNT is simply prepended to the host absolute path, meaning each mounted dir
# will automatically be unique (unless the user intentionally used the same host
# paths). This also makes copy-and-paste easier, since all but "/MOUNT" can be
# copied and used on the host as-is.
CONTAINER_CONFIG_DIR_ROOT = "/MOUNT"
CONTAINER_RESULTS_DIR_ROOT = "/MOUNT"
CONTAINER_DATASETS_DIR_ROOT = "/MOUNT"


class PathResolver:
    """
    Resolves host/container paths for results and datasets.
    """

    def __init__(self, data: dict) -> None:
        """
        data is a dictionary containing the paths for results and datasets
        Resolved paths for a container are simply a root dir (see above) prepended to the host path.
        """
        # TODO: Is this the best way to determine if running inside a Docker container?
        in_docker = Path("/.dockerenv").exists()
        (rp, dp) = (Path(data["results_path"]), Path(data["datasets_path"]))
        self.path_map = {
            "results_path": Path(f"{CONTAINER_RESULTS_DIR_ROOT}/{rp}") if in_docker else rp,
            "datasets_path": Path(f"{CONTAINER_DATASETS_DIR_ROOT}/{dp}") if in_docker else dp,
        }

    def resolve(self, dir_type: str) -> Path:
        """
        Given a directory type (e.g., 'results_path'), return the path.
        """
        if dir_type not in self.path_map:
            msg = f"Unknown dir_type: {dir_type}"
            raise ValueError(msg)

        return self.path_map[dir_type]
