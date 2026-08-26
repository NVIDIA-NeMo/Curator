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

from pathlib import Path


def resolve_benchmark_suite_dir(path: str | Path | None = None) -> Path:
    """Return the Curator checkout that provides the benchmark suite."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))

    this_file = Path(__file__).resolve()
    candidates.extend(
        [
            this_file.parents[2],
            Path.cwd(),
            Path.cwd().parent,
        ]
    )

    for candidate in candidates:
        expanded = candidate.expanduser().resolve()
        if (expanded / "benchmarking" / "pyproject.toml").exists():
            return expanded
        if expanded.name == "benchmarking" and (expanded / "pyproject.toml").exists():
            return expanded.parent

    if path is not None:
        msg = f"Benchmark suite directory does not look like a Curator checkout: {path}"
        raise ValueError(msg)

    msg = "Could not infer the benchmark suite directory. Run from a Curator checkout or pass --benchmark-suite-dir."
    raise ValueError(msg)


def benchmark_package_dir(suite_dir: str | Path | None = None) -> Path:
    """Return the benchmark package directory within a Curator checkout."""
    return resolve_benchmark_suite_dir(suite_dir) / "benchmarking"
