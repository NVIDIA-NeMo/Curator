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

"""Filesystem regression tests for the isolated benchmark source copy."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/tools"))

from prepare_benchmark_source import prepare_benchmark_source


def test_refresh_replaces_sources_and_excludes_curator(tmp_path: Path) -> None:
    source = tmp_path / "mount"
    destination = tmp_path / "runtime"
    for name in ("benchmarking", "tutorials", "nemo_curator"):
        (source / name).mkdir(parents=True)
        (source / name / "example.py").write_text("original")
    prepare_benchmark_source(source, destination)
    (source / "benchmarking/example.py").unlink()
    (source / "tutorials/example.py").write_text("updated")
    (source / "benchmarking/config.yaml").write_text("updated config")

    prepare_benchmark_source(source, destination)

    assert not (destination / "benchmarking/example.py").exists()
    assert (destination / "tutorials/example.py").read_text() == "updated"
    assert (destination / "benchmarking/config.yaml").read_text() == "updated config"
    assert not (destination / "nemo_curator").exists()
    assert (source / "nemo_curator/example.py").read_text() == "original"


def test_missing_tutorials_does_not_delete_existing_copy(tmp_path: Path) -> None:
    source = tmp_path / "mount"
    (source / "benchmarking").mkdir(parents=True)
    destination = tmp_path / "runtime"
    destination.mkdir()
    (destination / "keep").touch()

    with pytest.raises(FileNotFoundError, match="tutorials"):
        prepare_benchmark_source(source, destination)

    assert (destination / "keep").exists()


@pytest.mark.parametrize("destination", ["mount", "mount/runtime", "."])
def test_overlapping_paths_are_rejected(tmp_path: Path, destination: str) -> None:
    with pytest.raises(ValueError, match="overlap"):
        prepare_benchmark_source(tmp_path / "mount", tmp_path / destination)
