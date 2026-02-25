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

"""Tests for ALMManifestReaderStage."""

import json

import pytest

from nemo_curator.stages.audio.alm import ALMManifestReaderStage
from nemo_curator.tasks import AudioBatch
from nemo_curator.tasks.tasks import EmptyTask


class TestALMManifestReader:
    """Unit tests for ALMManifestReaderStage."""

    def test_reads_single_manifest(self, tmp_path):
        entries = [
            {"audio_filepath": "a.wav", "audio_sample_rate": 16000, "segments": []},
            {"audio_filepath": "b.wav", "audio_sample_rate": 22050, "segments": []},
        ]
        manifest = tmp_path / "input.jsonl"
        manifest.write_text("\n".join(json.dumps(e) for e in entries))

        stage = ALMManifestReaderStage(manifest_path=str(manifest))
        result = stage.process(EmptyTask)

        assert len(result) == 2
        assert all(isinstance(r, AudioBatch) for r in result)
        assert result[0].data[0]["audio_filepath"] == "a.wav"
        assert result[1].data[0]["audio_filepath"] == "b.wav"

    def test_reads_multiple_manifests(self, tmp_path):
        m1 = tmp_path / "m1.jsonl"
        m2 = tmp_path / "m2.jsonl"
        m1.write_text(json.dumps({"audio_filepath": "a.wav", "segments": []}))
        m2.write_text(json.dumps({"audio_filepath": "b.wav", "segments": []}))

        stage = ALMManifestReaderStage(manifest_path=[str(m1), str(m2)])
        result = stage.process(EmptyTask)

        assert len(result) == 2
        paths = [r.data[0]["audio_filepath"] for r in result]
        assert paths == ["a.wav", "b.wav"]

    def test_one_audio_batch_per_entry(self, tmp_path):
        entries = [{"audio_filepath": f"{i}.wav", "segments": []} for i in range(5)]
        manifest = tmp_path / "input.jsonl"
        manifest.write_text("\n".join(json.dumps(e) for e in entries))

        stage = ALMManifestReaderStage(manifest_path=str(manifest))
        result = stage.process(EmptyTask)

        assert len(result) == 5
        for i, batch in enumerate(result):
            assert len(batch.data) == 1
            assert batch.data[0]["audio_filepath"] == f"{i}.wav"

    def test_skips_blank_lines(self, tmp_path):
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(
            json.dumps({"audio_filepath": "a.wav", "segments": []})
            + "\n\n  \n"
            + json.dumps({"audio_filepath": "b.wav", "segments": []})
            + "\n"
        )

        stage = ALMManifestReaderStage(manifest_path=str(manifest))
        result = stage.process(EmptyTask)

        assert len(result) == 2

    def test_empty_manifest(self, tmp_path):
        manifest = tmp_path / "empty.jsonl"
        manifest.write_text("")

        stage = ALMManifestReaderStage(manifest_path=str(manifest))
        result = stage.process(EmptyTask)

        assert result == []

    def test_preserves_nested_data(self, tmp_path):
        entry = {
            "audio_filepath": "a.wav",
            "audio_sample_rate": 16000,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.2,
                    "speaker": "spk_0",
                    "metrics": {"bandwidth": 8000},
                }
            ],
        }
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(json.dumps(entry))

        stage = ALMManifestReaderStage(manifest_path=str(manifest))
        result = stage.process(EmptyTask)

        loaded = result[0].data[0]
        assert loaded["segments"][0]["metrics"]["bandwidth"] == 8000
        assert loaded["segments"][0]["speaker"] == "spk_0"

    def test_duplicate_manifests_for_repeat(self, tmp_path):
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav", "segments": []}))

        stage = ALMManifestReaderStage(manifest_path=[str(manifest)] * 3)
        result = stage.process(EmptyTask)

        assert len(result) == 3
        assert all(r.data[0]["audio_filepath"] == "a.wav" for r in result)

    def test_manifest_path_coerced_to_list(self):
        stage = ALMManifestReaderStage(manifest_path=("a.jsonl", "b.jsonl"))
        assert isinstance(stage.manifest_path, list)
        assert stage.manifest_path == ["a.jsonl", "b.jsonl"]

    def test_string_path_stays_string(self):
        stage = ALMManifestReaderStage(manifest_path="single.jsonl")
        assert isinstance(stage.manifest_path, str)

    def test_xenna_stage_spec(self):
        stage = ALMManifestReaderStage(manifest_path="x.jsonl")
        assert stage.xenna_stage_spec() == {"num_workers_per_node": 1}

    def test_ray_stage_spec(self):
        stage = ALMManifestReaderStage(manifest_path="x.jsonl")
        assert stage.ray_stage_spec() == {"is_fanout_stage": True}


class TestALMManifestReaderIntegration:
    """Integration test using the real sample fixture."""

    def test_reads_sample_fixture(self):
        from pathlib import Path

        fixture = Path(__file__).parent.parent.parent.parent / "fixtures" / "audio" / "alm" / "sample_input.jsonl"
        stage = ALMManifestReaderStage(manifest_path=str(fixture))
        result = stage.process(EmptyTask)

        assert len(result) == 5
        for batch in result:
            assert isinstance(batch, AudioBatch)
            assert len(batch.data) == 1
            entry = batch.data[0]
            assert "audio_filepath" in entry
            assert "segments" in entry
            assert len(entry["segments"]) > 0
