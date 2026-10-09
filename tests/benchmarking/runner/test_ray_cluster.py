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

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.ray_cluster import _compress_large_logs


def test_compress_large_logs_preserves_content_and_small_files(tmp_path: Path) -> None:
    large_log = tmp_path / "nested" / "frontend.log"
    large_log.parent.mkdir()
    content = b"Repeated diagnostic message\n" * 500_000
    large_log.write_bytes(content)
    small_log = tmp_path / "small.log"
    small_log.write_bytes(b"Important startup error\n")
    link = tmp_path / "linked.log"
    link.symlink_to(large_log)

    _compress_large_logs(tmp_path)

    compressed = large_log.with_suffix(".log.gz")
    assert not large_log.exists()
    with gzip.open(compressed, "rb") as log_file:
        assert log_file.read() == content
    assert compressed.stat().st_size < len(content) // 10
    assert small_log.read_bytes() == b"Important startup error\n"
    assert link.is_symlink()


def test_compression_failure_preserves_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "frontend.log"
    with log.open("wb") as log_file:
        log_file.truncate(11 * 1024 * 1024)

    def fail_open(*_args, **_kwargs) -> None:
        msg = "Disk full"
        raise OSError(msg)

    monkeypatch.setattr(gzip, "open", fail_open)
    _compress_large_logs(tmp_path)

    assert log.stat().st_size == 11 * 1024 * 1024
