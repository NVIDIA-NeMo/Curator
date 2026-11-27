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

from typing import TYPE_CHECKING

from nemo_curator.stages.audio.tagging.utils import load_vocab_file

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadVocabFile:
    def test_single_line(self, tmp_path: Path) -> None:
        p = tmp_path / "v.txt"
        p.write_text("abc xyz")
        result = load_vocab_file(str(p))
        assert "a" in result
        assert " " in result

    def test_multi_line(self, tmp_path: Path) -> None:
        p = tmp_path / "v.txt"
        p.write_text("a\nb\nc\n")
        result = load_vocab_file(str(p))
        assert "a" in result
        assert "b" in result
        assert "c" in result
