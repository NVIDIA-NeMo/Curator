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
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.io.nemo_tarred_reader import (
    NemoTarredAudioReader,
    NemoTarShardDiscoveryStage,
    NemoTarShardReaderStage,
    _expand_nemo_path,
)
from nemo_curator.tasks import FileGroupTask


class TestExpandNemoPath:
    def test_no_pattern(self) -> None:
        assert _expand_nemo_path("/data/audio.tar") == ["/data/audio.tar"]

    def test_simple_range(self) -> None:
        result = _expand_nemo_path("/data/audio_OP_0..2_CL_.tar")
        assert result == ["/data/audio0.tar", "/data/audio1.tar", "/data/audio2.tar"]


class TestNemoTarShardDiscoveryStage:
    def test_requires_yaml_path(self) -> None:
        with pytest.raises(ValueError, match="yaml_path"):
            NemoTarShardDiscoveryStage(yaml_path="")

    def test_manifest_to_rel_path(self) -> None:
        result = NemoTarShardDiscoveryStage._manifest_to_rel_path(
            "/data/yodas/0_from_captions/en/sharded_manifests/manifest_42.jsonl",
            "yodas",
        )
        assert result == "yodas/0_from_captions/en/sharded_manifests/manifest_42"

    def test_manifest_to_rel_path_not_found(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            NemoTarShardDiscoveryStage._manifest_to_rel_path("/data/other/file.jsonl", "yodas")

    def test_manifest_to_rel_path_ambiguous(self) -> None:
        with pytest.raises(ValueError, match="appears"):
            NemoTarShardDiscoveryStage._manifest_to_rel_path(
                "/data/yodas/nested/yodas/file.jsonl", "yodas"
            )


class TestNemoTarShardReaderStage:
    def _create_tar_and_manifest(self, tmp_path: Path) -> tuple[str, str]:
        """Create a minimal tar + manifest for testing."""
        manifest_path = str(tmp_path / "manifest.jsonl")
        tar_path = str(tmp_path / "audio.tar")

        audio_data = np.zeros(16000, dtype=np.float32)
        buf = BytesIO()
        sf.write(buf, audio_data, 16000, format="WAV")
        audio_bytes = buf.getvalue()

        with tarfile.open(tar_path, "w") as tar:
            info = tarfile.TarInfo(name="utt_0.wav")
            info.size = len(audio_bytes)
            tar.addfile(info, BytesIO(audio_bytes))

        with open(manifest_path, "w") as f:
            f.write(json.dumps({"audio_filepath": "utt_0.wav", "duration": 1.0}) + "\n")

        return manifest_path, tar_path

    def test_reads_single_shard(self, tmp_path: Path) -> None:
        manifest_path, tar_path = self._create_tar_and_manifest(tmp_path)
        stage = NemoTarShardReaderStage()
        task = FileGroupTask(
            task_id="shard_0",
            dataset_name="test_ds",
            data=[manifest_path, tar_path],
            reader_config={"corpus": "test", "shard_key": "test/shard_0"},
        )
        results = stage.process(task)
        assert len(results) == 1
        assert results[0].data["sample_rate"] == 16000
        assert results[0].data["corpus"] == "test"
        assert isinstance(results[0].data["waveform"], np.ndarray)

    def test_missing_tar_raises(self, tmp_path: Path) -> None:
        manifest_path = str(tmp_path / "manifest.jsonl")
        with open(manifest_path, "w") as f:
            f.write(json.dumps({"audio_filepath": "utt_0.wav"}) + "\n")

        stage = NemoTarShardReaderStage()
        task = FileGroupTask(
            task_id="shard_0",
            dataset_name="test_ds",
            data=[manifest_path, str(tmp_path / "missing.tar")],
            reader_config={"corpus": "test", "shard_key": "test/shard_0"},
        )
        with pytest.raises(FileNotFoundError):
            stage.process(task)

    def test_shard_total_set(self, tmp_path: Path) -> None:
        manifest_path, tar_path = self._create_tar_and_manifest(tmp_path)
        stage = NemoTarShardReaderStage()
        task = FileGroupTask(
            task_id="shard_0",
            dataset_name="test_ds",
            data=[manifest_path, tar_path],
            reader_config={"corpus": "test", "shard_key": "test/shard_0"},
        )
        results = stage.process(task)
        assert results[0]._metadata["_shard_total"] == 1


class TestNemoTarredAudioReader:
    def test_requires_yaml_path(self) -> None:
        with pytest.raises(ValueError, match="yaml_path"):
            NemoTarredAudioReader(yaml_path="")

    def test_decompose_returns_two_stages(self, tmp_path: Path) -> None:
        yaml_path = str(tmp_path / "config.yaml")
        Path(yaml_path).write_text("[]")
        reader = NemoTarredAudioReader(yaml_path=yaml_path)
        stages = reader.decompose()
        assert len(stages) == 2
