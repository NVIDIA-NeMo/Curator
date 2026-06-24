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

from nemo_curator.utils.atomic_io import write_json_atomically


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
