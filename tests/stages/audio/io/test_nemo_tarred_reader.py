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

import pytest

from nemo_curator.stages.audio.io.nemo_tarred_reader import NemoTarShardReaderStage, NemoTarredAudioReader


def test_reader_composite_forwards_duration_filter() -> None:
    reader = NemoTarredAudioReader(
        yaml_path="data_config.yaml",
        duration_key="segment_duration",
        max_duration_s=40.0,
    )

    shard_reader = reader.decompose()[1]
    assert isinstance(shard_reader, NemoTarShardReaderStage)
    assert shard_reader.duration_key == "segment_duration"
    assert shard_reader.max_duration_s == 40.0


def test_reader_rejects_non_positive_max_duration() -> None:
    with pytest.raises(ValueError, match="max_duration_s"):
        NemoTarShardReaderStage(max_duration_s=0)


def test_reader_manifest_lookup_accepts_common_path_variants(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"audio_filepath": "/data/shard/audio_0.wav", "duration": 1.0}\n',
        encoding="utf-8",
    )
    stage = NemoTarShardReaderStage()

    lookup, entry_count = stage._read_manifest(str(manifest))

    assert entry_count == 1
    assert lookup["/data/shard/audio_0.wav"]["duration"] == 1.0
    assert lookup["data/shard/audio_0.wav"]["duration"] == 1.0
    assert lookup["audio_0.wav"]["duration"] == 1.0
