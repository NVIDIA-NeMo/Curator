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

from pathlib import Path, PurePosixPath

import pytest

from nemo_curator.stages.audio.io._nemo_speech_paths import contained_output_path, validate_local_output_dir


@pytest.mark.parametrize("relative", ["", ".", PurePosixPath("."), "/absolute", "../outside", "nested/../../outside"])
def test_rejects_root_absolute_and_traversing_outputs(tmp_path: Path, relative: str | PurePosixPath) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        contained_output_path(tmp_path, relative)


@pytest.mark.parametrize("leaf", [False, True])
def test_rejects_symlink_components_even_when_target_is_inside_root(tmp_path: Path, leaf: bool) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        contained_output_path(tmp_path, "alias" if leaf else "alias/audio.opus")


def test_normalizes_safe_paths_without_creating_outputs(tmp_path: Path) -> None:
    result = contained_output_path(tmp_path, r"nested\audio.opus")
    assert result == tmp_path / "nested/audio.opus"
    assert not result.parent.exists()
    assert validate_local_output_dir(str(tmp_path)) == tmp_path


@pytest.mark.parametrize("path", ["relative", "s3://bucket/output", "file:///tmp/output"])
def test_output_directory_must_be_absolute_and_local(path: str) -> None:
    with pytest.raises(ValueError, match="absolute local path"):
        validate_local_output_dir(path)
