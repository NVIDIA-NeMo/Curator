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

import os
import urllib.error
from pathlib import Path

import pytest

from nemo_curator.stages.audio.datasets.file_utils import download_file


def test_download_file_streams_url_without_wget(tmp_path: Path) -> None:
    source = tmp_path / "src" / "archive.tgz"
    source.parent.mkdir()
    source.write_bytes(b"payload")
    target = tmp_path / "dst"
    target.mkdir()

    path = download_file(source.as_uri(), str(target))

    assert path == str(target / "archive.tgz")
    assert Path(path).read_bytes() == b"payload"
    assert os.listdir(target) == ["archive.tgz"]  # no temporary file left behind


def test_failed_download_leaves_no_partial_file(tmp_path: Path) -> None:
    with pytest.raises(urllib.error.URLError):
        download_file((tmp_path / "missing.tgz").as_uri(), str(tmp_path))
    assert os.listdir(tmp_path) == []
