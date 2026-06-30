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

import json
from pathlib import Path

import pytest

import nemo_curator.utils.atomic_io as atomic_io_module
from nemo_curator.utils.atomic_io import write_json_atomically, write_json_atomically_if_absent


class TestAtomicIo:
    def test_write_json_atomically_creates_parent_and_writes_json(self, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "payload.json"

        write_json_atomically(output_path, {"b": 2, "a": 1}, separators=(",", ":"))

        assert output_path.read_text() == '{"a":1,"b":2}\n'
        assert json.loads(output_path.read_text()) == {"a": 1, "b": 2}

    def test_write_json_atomically_cleans_temp_file_on_failure(self, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "payload.json"

        with pytest.raises(TypeError):
            write_json_atomically(output_path, {"bad": object()})

        assert not output_path.exists()
        assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))

    def test_write_json_atomically_succeeds_when_directory_fsync_is_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_path = tmp_path / "payload.json"

        def fail_directory_fsync(_path: Path) -> None:
            msg = "directory fsync unsupported"
            raise OSError(msg)

        monkeypatch.setattr(atomic_io_module, "fsync_directory", fail_directory_fsync)

        write_json_atomically(output_path, {"status": "completed"})

        assert json.loads(output_path.read_text()) == {"status": "completed"}

    def test_write_json_atomically_if_absent_does_not_replace_existing_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "payload.json"

        assert write_json_atomically_if_absent(output_path, {"writer": 1}) is True
        assert write_json_atomically_if_absent(output_path, {"writer": 2}) is False

        assert json.loads(output_path.read_text()) == {"writer": 1}
        assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))

    def test_write_json_atomically_if_absent_does_not_fail_after_commit_when_cleanup_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_path = tmp_path / "payload.json"
        original_unlink = Path.unlink

        def fail_temp_cleanup(path: Path, *, missing_ok: bool = False) -> None:
            if path.suffix == ".tmp":
                msg = "cleanup unavailable"
                raise OSError(msg)
            original_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", fail_temp_cleanup)

        assert write_json_atomically_if_absent(output_path, {"writer": 1}) is True
        assert json.loads(output_path.read_text()) == {"writer": 1}
