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

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.io.nemo_tarred_reader import (
    NemoTarredAudioReader,
    NemoTarShardDiscoveryStage,
    NemoTarShardReaderStage,
    _iter_discovery_groups,
    _iter_input_cfg_entries,
)
from nemo_curator.tasks import FileGroupTask


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


def test_reader_manifest_lookup_accepts_common_path_variants(tmp_path: Path) -> None:
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


def test_manifest_lookup_disambiguates_shared_basename_by_path_suffix(tmp_path: Path) -> None:
    """When two entries share a basename, a member resolves by longest path suffix."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"audio_filepath": "spk_a/utt.wav", "duration": 1.0}\n'
        '{"audio_filepath": "spk_b/utt.wav", "duration": 2.0}\n',
        encoding="utf-8",
    )
    stage = NemoTarShardReaderStage()

    lookup, _ = stage._read_manifest(str(manifest))

    # Full path wins outright.
    assert lookup.match("spk_a/utt.wav")["duration"] == 1.0
    assert lookup.match("spk_b/utt.wav")["duration"] == 2.0
    # A bare, ambiguous basename resolves to nothing rather than a wrong entry.
    assert lookup.match("utt.wav") is None


def test_read_manifest_skips_lines_missing_filepath_key(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"audio_filepath": "a.wav", "duration": 1.0}\n'
        '{"duration": 2.0}\n'
        '{"audio_filepath": "c.wav", "duration": 3.0}\n',
        encoding="utf-8",
    )
    stage = NemoTarShardReaderStage()

    lookup, entry_count = stage._read_manifest(str(manifest))

    assert entry_count == 3
    assert lookup["a.wav"]["duration"] == 1.0
    assert lookup["c.wav"]["duration"] == 3.0


def test_read_manifest_skips_invalid_json_lines(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"audio_filepath": "ok.wav", "duration": 1.0}\n'
        "not json\n",
        encoding="utf-8",
    )
    stage = NemoTarShardReaderStage()

    lookup, entry_count = stage._read_manifest(str(manifest))

    assert entry_count == 2
    assert lookup["ok.wav"]["duration"] == 1.0


def test_reader_skips_tar_members_when_extractfile_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tar members where extractfile() returns None must be skipped, not raise AttributeError."""
    audio = np.zeros(16000, dtype=np.float32)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, 16000, format="WAV")
    wav_bytes = wav_buf.getvalue()

    tar_path = tmp_path / "shard.tar"
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="audio_0.wav")
        info.size = len(wav_bytes)
        tf.addfile(info, io.BytesIO(wav_bytes))

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"audio_filepath": "audio_0.wav", "duration": 1.0}) + "\n",
        encoding="utf-8",
    )

    def _extractfile_returns_none(_self: tarfile.TarFile, _member: tarfile.TarInfo) -> None:
        return None

    monkeypatch.setattr(tarfile.TarFile, "extractfile", _extractfile_returns_none)

    stage = NemoTarShardReaderStage()
    task = FileGroupTask(
        task_id="test_shard",
        dataset_name="test",
        data=[str(manifest_path), str(tar_path)],
        reader_config={"corpus": "test", "shard_key": "test_shard"},
    )

    results = stage.process(task)

    assert results == []


def test_iter_discovery_groups_rejects_empty_yaml() -> None:
    with pytest.raises(ValueError, match="empty"):
        _iter_discovery_groups(None, "bad.yaml")


@pytest.mark.parametrize("config", [{"corpus": "x"}, "scalar", 42])
def test_iter_discovery_groups_rejects_non_list_root(config: object) -> None:
    with pytest.raises(TypeError, match="must be a list"):
        _iter_discovery_groups(config, "bad.yaml")


def test_iter_discovery_groups_skips_non_mapping_entries() -> None:
    groups = _iter_discovery_groups([{"input_cfg": []}, "skip-me", {"input_cfg": []}], "ok.yaml")
    assert len(groups) == 2


def test_iter_input_cfg_entries_skips_non_list_and_non_mapping() -> None:
    assert _iter_input_cfg_entries({"input_cfg": "bad"}, "ok.yaml") == []
    assert _iter_input_cfg_entries({"input_cfg": ["bad", {"corpus": "c"}]}, "ok.yaml") == [{"corpus": "c"}]


def test_discovery_process_raises_on_empty_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("", encoding="utf-8")
    stage = NemoTarShardDiscoveryStage(yaml_path=str(yaml_path))

    with pytest.raises(ValueError, match="empty"):
        stage.process(None)  # type: ignore[arg-type]


def test_discovery_process_raises_on_scalar_yaml_root(tmp_path: Path) -> None:
    yaml_path = tmp_path / "scalar.yaml"
    yaml_path.write_text("just_a_string\n", encoding="utf-8")
    stage = NemoTarShardDiscoveryStage(yaml_path=str(yaml_path))

    with pytest.raises(TypeError, match="must be a list"):
        stage.process(None)  # type: ignore[arg-type]


def test_discovery_skips_corpus_missing_required_paths(tmp_path: Path) -> None:
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        """
- input_cfg:
  - corpus: broken
    type: nemo_tarred
    manifest_filepath: /data/broken/manifest_0.jsonl
  - corpus: good
    type: nemo_tarred
    manifest_filepath: /data/good/manifest_0.jsonl
    tarred_audio_filepaths: /data/good/audio_0.tar
""".lstrip(),
        encoding="utf-8",
    )
    stage = NemoTarShardDiscoveryStage(yaml_path=str(yaml_path))

    tasks = stage.process(None)  # type: ignore[arg-type]

    assert [task.task_id for task in tasks] == ["good/manifest_0"]
    assert tasks[0].data == ["/data/good/manifest_0.jsonl", "/data/good/audio_0.tar"]


def test_discovery_skips_manifest_path_that_cannot_map_to_corpus(tmp_path: Path) -> None:
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        """
- input_cfg:
  - corpus: good
    type: nemo_tarred
    manifest_filepath: /data/other/manifest_0.jsonl
    tarred_audio_filepaths: /data/other/audio_0.tar
  - corpus: good
    type: nemo_tarred
    manifest_filepath: /data/good/manifest_1.jsonl
    tarred_audio_filepaths: /data/good/audio_1.tar
""".lstrip(),
        encoding="utf-8",
    )
    stage = NemoTarShardDiscoveryStage(yaml_path=str(yaml_path))

    tasks = stage.process(None)  # type: ignore[arg-type]

    assert [task.task_id for task in tasks] == ["good/manifest_1"]
