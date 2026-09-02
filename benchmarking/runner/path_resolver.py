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

import os
from collections.abc import Iterator
from pathlib import Path

CONTAINER_CURATOR_DIR = "/opt/Curator"
CURATOR_BENCHMARK_PATH_MODE_ENV = "CURATOR_BENCHMARK_PATH_MODE"

# Prefix prepended to host paths when resolving container paths that do not have an explicit
# container_path specified. Using /MOUNT makes it obvious in log/error messages from the
# container that these paths are available on the host, and allows copy-and-paste of all
# but the "/MOUNT" prefix to get the equivalent host path.
DEFAULT_CONTAINER_PATH_PREFIX = "/MOUNT"
VALID_PATH_MODES = {"auto", "host", "container"}


def set_path_mode(mode: str) -> None:
    """Set the process-wide path-resolution mode used by PathResolver."""
    mode = mode.strip().lower()
    if mode not in VALID_PATH_MODES:
        msg = f"{CURATOR_BENCHMARK_PATH_MODE_ENV} must be one of {sorted(VALID_PATH_MODES)}; got {mode!r}"
        raise ValueError(msg)
    os.environ[CURATOR_BENCHMARK_PATH_MODE_ENV] = mode


def get_path_mode() -> str:
    """Return the requested path-resolution mode."""
    mode = os.environ.get(CURATOR_BENCHMARK_PATH_MODE_ENV, "auto").strip().lower() or "auto"
    if mode not in VALID_PATH_MODES:
        msg = f"{CURATOR_BENCHMARK_PATH_MODE_ENV} must be one of {sorted(VALID_PATH_MODES)}; got {mode!r}"
        raise ValueError(msg)
    return mode


def _auto_detect_container() -> bool:
    """Best-effort compatibility fallback for callers that do not set path mode."""
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def use_container_paths() -> bool:
    """Return whether configured path names should resolve to container paths."""
    mode = get_path_mode()
    if mode == "container":
        return True
    if mode == "host":
        return False
    return _auto_detect_container()


class PathResolver:
    """
    Resolves host/container paths for results and datasets.
    """

    def __init__(self, data: dict) -> None:
        """
        data is a dictionary containing path configuration, either via the 'paths' list or
        the deprecated 'results_path', 'datasets_path', and 'model_weights_path' fields.

        For 'paths' entries, each item must have a 'name' and 'host_path'. An optional
        'container_path' overrides the default container path (which is the host_path
        prefixed with '/MOUNT').
        """
        use_container_path = use_container_paths()
        self.path_map: dict[str, Path] = {}
        self._volume_pairs: list[tuple[Path, Path]] = []

        if "paths" in data:
            for path_entry in data["paths"]:
                name = path_entry["name"]
                host_path = Path(path_entry["host_path"])
                raw_container = path_entry.get("container_path")
                container_path = (
                    Path(raw_container) if raw_container else Path(f"{DEFAULT_CONTAINER_PATH_PREFIX}/{host_path}")
                )
                self.path_map[name] = container_path if use_container_path else host_path
                self._volume_pairs.append((host_path, container_path))
        else:
            # Legacy top-level YAML path fields are deprecated. The path
            # names themselves, including "results_path", are still valid
            # when provided through the preferred "paths" list above.
            for name, host_path in [
                ("results_path", Path(data["results_path"])),
                ("datasets_path", Path(data["datasets_path"])),
                ("model_weights_path", Path(data["model_weights_path"])),
            ]:
                container_path = Path(f"{DEFAULT_CONTAINER_PATH_PREFIX}/{host_path}")
                self.path_map[name] = container_path if use_container_path else host_path
                self._volume_pairs.append((host_path, container_path))

    def volume_mount_pairs(self) -> Iterator[tuple[Path, Path]]:
        """Yield (host_path, container_path) pairs for all configured paths."""
        yield from self._volume_pairs

    def unmap_container_path(self, path: Path) -> Path:
        """Return the host path for a path under a configured container mount."""
        matches: list[tuple[int, Path]] = []
        for host_path, container_path in self._volume_pairs:
            # relative_to() succeeds only when path is inside this container
            # mount. Otherwise it raises ValueError and we try the next mount.
            try:
                relative_path = path.relative_to(container_path)
            except ValueError:
                continue
            # Store both match specificity and the host-visible path so
            # nested/overlapping mounts resolve to the longest matching mount.
            matches.append((len(container_path.parts), host_path / relative_path))

        if not matches:
            return path

        _, host_visible_path = max(matches, key=lambda match: match[0])
        return host_visible_path

    def resolve(self, name: str) -> Path:
        """
        Given a path name (e.g., 'results_path'), return the resolved host or container path.
        """
        if name not in self.path_map:
            msg = f"Unknown path name: {name}"
            raise ValueError(msg)

        return self.path_map[name]
