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
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.entry import Entry


def test_curator_repo_dir_placeholder_uses_explicit_curator_repo_dir(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CURATOR_BENCHMARK_CURATOR_REPO_DIR", "/custom/Curator")

    resolved = Entry.substitute_reserved_placeholders("{curator_repo_dir}/nemo_curator", tmp_path, object())

    assert resolved == "/custom/Curator/nemo_curator"


def test_curator_repo_dir_placeholder_fails_when_curator_repo_dir_is_not_set(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CURATOR_BENCHMARK_CURATOR_REPO_DIR", raising=False)

    with pytest.raises(RuntimeError, match="CURATOR_BENCHMARK_CURATOR_REPO_DIR must be set"):
        Entry.substitute_reserved_placeholders("{curator_repo_dir}/nemo_curator", tmp_path, object())
