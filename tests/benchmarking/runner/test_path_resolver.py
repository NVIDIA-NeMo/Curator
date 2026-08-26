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

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.path_resolver import PathResolver


def test_unmap_container_path_returns_host_visible_path() -> None:
    path_resolver = PathResolver(
        {
            "paths": [
                {
                    "name": "results_path",
                    "host_path": "/host/results",
                    "container_path": "/container/results",
                }
            ]
        }
    )

    assert path_resolver.unmap_container_path(Path("/container/results/run")) == Path("/host/results/run")


def test_unmap_container_path_returns_unmapped_path_unchanged() -> None:
    path_resolver = PathResolver(
        {
            "paths": [
                {
                    "name": "results_path",
                    "host_path": "/host/results",
                    "container_path": "/container/results",
                }
            ]
        }
    )

    assert path_resolver.unmap_container_path(Path("/other/results/run")) == Path("/other/results/run")
