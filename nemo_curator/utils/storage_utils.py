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

import pathlib


def _get_local_path(localpath: pathlib.Path, *args: str) -> pathlib.Path:
    """Construct a full local path from a base path and additional components.
    Args:
        localpath: The base local path.
        *args: Additional path components.
    Returns:
        The full local path as a Path object.
    """
    return pathlib.Path(localpath, *args)


def get_full_path(path: str | pathlib.Path, *args: str) -> pathlib.Path:
    """Construct a full path from a base path and additional components.
    Args:
        path: The base path.
        *args: Additional path components.
    Returns:
        The full path as a StoragePrefix or Path object.
    """
    # Convert string paths to the appropriate type
    if isinstance(path, str):
        return _get_local_path(pathlib.Path(path), *args)

    return pathlib.Path(path, *args)
