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

"""Verify durable shard registration rejects conflicting or corrupt replay state."""

import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.io._nemo_speech_state import register_nemo_speech_shard


def test_registration_replay_preserves_original_count(tmp_path: Path) -> None:
    path = register_nemo_speech_shard(tmp_path, "corpus/shard", 2)
    original = path.read_bytes()
    assert register_nemo_speech_shard(tmp_path, "corpus/shard", 2) == path
    with pytest.raises(ValueError, match=r"Conflicting.*registration"):
        register_nemo_speech_shard(tmp_path, "corpus/shard", 3)
    assert path.read_bytes() == original
    assert json.loads(original)["expected_inputs"] == 2


def test_corrupt_registration_is_not_overwritten(tmp_path: Path) -> None:
    path = register_nemo_speech_shard(tmp_path, "corpus/shard", 2)
    path.write_text("{interrupted", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Invalid.*registration"):
        register_nemo_speech_shard(tmp_path, "corpus/shard", 2)
    assert path.read_text(encoding="utf-8") == "{interrupted"


@pytest.mark.parametrize("count", [0, -1])
def test_registration_rejects_empty_shards_without_writing(tmp_path: Path, count: int) -> None:
    with pytest.raises(ValueError, match="at least one input"):
        register_nemo_speech_shard(tmp_path, "corpus/shard", count)
    assert not list(tmp_path.iterdir())
