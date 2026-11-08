# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Copy the mounted benchmark suite without exposing its Curator library to imports.

This standard-library-only helper is shared by container runs and install-mode
environment setup. It neither installs dependencies nor changes Curator-under-test.
"""

import os
import shutil
from pathlib import Path


def prepare_benchmark_source(source: Path, destination: Path) -> None:
    """Replace the owned runtime copy with the mounted benchmarks and tutorials."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        msg = "Benchmark source mount and runtime copy must not overlap."
        raise ValueError(msg)
    subdirs = ("benchmarking", "tutorials")
    for subdir in subdirs:
        if not (source / subdir).is_dir():
            msg = f"Mounted benchmark source is missing directory: {source / subdir}"
            raise FileNotFoundError(msg)

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for subdir in subdirs:
        shutil.copytree(source / subdir, destination / subdir, symlinks=True)


if __name__ == "__main__":
    prepare_benchmark_source(
        Path(os.environ.get("CURATOR_BENCHMARK_SOURCE_MOUNT_DIR", "/tmp/.curator-benchmark-source")),  # noqa: S108
        Path(os.environ.get("CURATOR_BENCHMARK_SOURCE_DIR", "/opt/curator-benchmark-source")),
    )
