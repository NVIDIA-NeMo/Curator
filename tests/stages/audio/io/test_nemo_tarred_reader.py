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

import json
import tarfile
from io import BytesIO
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import yaml

from nemo_curator.stages.audio.io.nemo_tarred_reader import (
    NemoTarredAudioReader,
    NemoTarShardDiscoveryStage,
    NemoTarShardReaderStage,
    _expand_nemo_path,
)
from nemo_curator.tasks import EmptyTask, FileGroupTask

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helper: _expand_nemo_path
# ---------------------------------------------------------------------------


def test_expand_nemo_path_no_pattern() -> None:
    assert _expand_nemo_path("/data/shard_0.tar") == ["/data/shard_0.tar"]


def test_expand_nemo_path_single_element() -> None:
    result = _expand_nemo_path("/data/shard__OP_0..0_CL_.tar")
    assert result == ["/data/shard_0.tar"]


def test_expand_nemo_path_range() -> None:
    result = _expand_nemo_path("/data/shard__OP_0..3_CL_.tar")
    assert result == [
        "/data/shard_0.tar",
        "/data/shard_1.tar",
        "/data/shard_2.tar",
        "/data/shard_3.tar",
    ]


# ---------------------------------------------------------------------------
# NemoTarShardDiscoveryStage
# ---------------------------------------------------------------------------


def test_discovery_requires_yaml_path() -> None:
    with pytest.raises(ValueError, match="yaml_path is required"):
        NemoTarShardDiscoveryStage(yaml_path="")


def _write_yaml_config(path: Path, corpora: list[dict]) -> None:
    config = [{"input_cfg": corpora}]
    path.write_text(yaml.dump(config), encoding="utf-8")


def test_discovery_emits_one_task_per_shard(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    _write_yaml_config(
        yaml_path,
        [
            {
                "corpus": "yodas",
                "type": "nemo_tarred",
                "manifest_filepath": "/data/manifest_0.json",
                "tarred_audio_filepaths": "/data/shard_0.tar",
            },
            {
                "corpus": "yodas",
                "type": "nemo_tarred",
                "manifest_filepath": "/data/manifest_1.json",
                "tarred_audio_filepaths": "/data/shard_1.tar",
            },
        ],
    )
    stage = NemoTarShardDiscoveryStage(yaml_path=str(yaml_path))

    tasks = stage.process(EmptyTask)
    assert len(tasks) == 2
    assert tasks[0].data == ["/data/manifest_0.json", "/data/shard_0.tar"]
    assert tasks[1].data == ["/data/manifest_1.json", "/data/shard_1.tar"]


def test_discovery_expands_brace_patterns(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    _write_yaml_config(
        yaml_path,
        [
            {
                "corpus": "libri",
                "type": "nemo_tarred",
                "manifest_filepath": "/data/manifest__OP_0..2_CL_.json",
                "tarred_audio_filepaths": "/data/shard__OP_0..2_CL_.tar",
            },
        ],
    )
    stage = NemoTarShardDiscoveryStage(yaml_path=str(yaml_path))

    tasks = stage.process(EmptyTask)
    assert len(tasks) == 3


def test_discovery_corpus_filter(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    _write_yaml_config(
        yaml_path,
        [
            {
                "corpus": "keep",
                "type": "nemo_tarred",
                "manifest_filepath": "/data/m0.json",
                "tarred_audio_filepaths": "/data/s0.tar",
            },
            {
                "corpus": "skip",
                "type": "nemo_tarred",
                "manifest_filepath": "/data/m1.json",
                "tarred_audio_filepaths": "/data/s1.tar",
            },
        ],
    )
    stage = NemoTarShardDiscoveryStage(yaml_path=str(yaml_path), corpus_filter=["keep"])

    tasks = stage.process(EmptyTask)
    assert len(tasks) == 1
    assert tasks[0].dataset_name == "keep"


def test_discovery_skips_non_nemo_tarred(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    _write_yaml_config(
        yaml_path,
        [
            {
                "corpus": "other",
                "type": "lhotse",
                "manifest_filepath": "/data/m.json",
                "tarred_audio_filepaths": "/data/s.tar",
            },
        ],
    )
    stage = NemoTarShardDiscoveryStage(yaml_path=str(yaml_path))

    tasks = stage.process(EmptyTask)
    assert len(tasks) == 0


def test_discovery_mismatched_manifest_tar_count(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    _write_yaml_config(
        yaml_path,
        [
            {
                "corpus": "bad",
                "type": "nemo_tarred",
                "manifest_filepath": "/data/manifest__OP_0..1_CL_.json",
                "tarred_audio_filepaths": "/data/shard_0.tar",
            },
        ],
    )
    stage = NemoTarShardDiscoveryStage(yaml_path=str(yaml_path))

    with pytest.raises(ValueError, match="Manifest/tar count mismatch"):
        stage.process(EmptyTask)


# ---------------------------------------------------------------------------
# NemoTarShardReaderStage
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_s: float = 0.1, sr: int = 16000) -> bytes:
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    buf = BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


def _make_test_shard(tmp_path: Path, entries: list[dict]) -> tuple[str, str]:
    manifest_path = tmp_path / "manifest.json"
    tar_path = tmp_path / "shard.tar"

    manifest_path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )

    with tarfile.open(tar_path, "w") as tar:
        for entry in entries:
            wav_data = _make_wav_bytes()
            info = tarfile.TarInfo(name=entry["audio_filepath"])
            info.size = len(wav_data)
            tar.addfile(info, BytesIO(wav_data))

    return str(manifest_path), str(tar_path)


def test_reader_emits_audio_tasks(tmp_path: Path) -> None:
    entries = [
        {"audio_filepath": "utt_0.wav", "text": "hello"},
        {"audio_filepath": "utt_1.wav", "text": "world"},
    ]
    manifest_path, tar_path = _make_test_shard(tmp_path, entries)

    stage = NemoTarShardReaderStage()
    task = FileGroupTask(
        task_id="test_shard",
        dataset_name="test",
        data=[manifest_path, tar_path],
        reader_config={"corpus": "test", "shard_idx": 0},
    )

    with patch("nemo_curator.stages.audio.io.nemo_tarred_reader._open_tar") as mock_open:
        mock_open.return_value = tarfile.open(tar_path, "r")  # noqa: SIM115
        results = stage.process(task)

    assert len(results) == 2
    for r in results:
        assert "waveform" in r.data
        assert "sample_rate" in r.data
        assert isinstance(r.data["waveform"], np.ndarray)
        assert r.data["waveform"].ndim == 1


def test_reader_skips_corrupt_audio(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    tar_path = tmp_path / "shard.tar"

    entry = {"audio_filepath": "bad.wav", "text": "corrupt"}
    manifest_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    with tarfile.open(tar_path, "w") as tar:
        bad_data = b"not a wav file"
        info = tarfile.TarInfo(name="bad.wav")
        info.size = len(bad_data)
        tar.addfile(info, BytesIO(bad_data))

    stage = NemoTarShardReaderStage()
    task = FileGroupTask(
        task_id="bad_shard",
        dataset_name="test",
        data=[str(manifest_path), str(tar_path)],
        reader_config={"corpus": "test", "shard_idx": 0},
    )

    with patch("nemo_curator.stages.audio.io.nemo_tarred_reader._open_tar") as mock_open:
        mock_open.return_value = tarfile.open(tar_path, "r")  # noqa: SIM115
        results = stage.process(task)

    assert len(results) == 0


def test_reader_converts_stereo_to_mono(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    tar_path = tmp_path / "shard.tar"

    entry = {"audio_filepath": "stereo.wav", "text": "stereo"}
    manifest_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    stereo = np.zeros((1600, 2), dtype=np.float32)
    buf = BytesIO()
    sf.write(buf, stereo, 16000, format="WAV")
    wav_data = buf.getvalue()

    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name="stereo.wav")
        info.size = len(wav_data)
        tar.addfile(info, BytesIO(wav_data))

    stage = NemoTarShardReaderStage()
    task = FileGroupTask(
        task_id="stereo_shard",
        dataset_name="test",
        data=[str(manifest_path), str(tar_path)],
        reader_config={"corpus": "test", "shard_idx": 0},
    )

    with patch("nemo_curator.stages.audio.io.nemo_tarred_reader._open_tar") as mock_open:
        mock_open.return_value = tarfile.open(tar_path, "r")  # noqa: SIM115
        results = stage.process(task)

    assert len(results) == 1
    assert results[0].data["waveform"].ndim == 1


# ---------------------------------------------------------------------------
# NemoTarredAudioReader (CompositeStage)
# ---------------------------------------------------------------------------


def test_composite_requires_yaml_path() -> None:
    with pytest.raises(ValueError, match="yaml_path is required"):
        NemoTarredAudioReader(yaml_path="")


def test_composite_decomposes_to_two_stages(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    _write_yaml_config(
        yaml_path,
        [
            {
                "corpus": "test",
                "type": "nemo_tarred",
                "manifest_filepath": "/data/m.json",
                "tarred_audio_filepaths": "/data/s.tar",
            },
        ],
    )
    reader = NemoTarredAudioReader(yaml_path=str(yaml_path))
    stages = reader.decompose()
    assert len(stages) == 2
    assert isinstance(stages[0], NemoTarShardDiscoveryStage)
    assert isinstance(stages[1], NemoTarShardReaderStage)
