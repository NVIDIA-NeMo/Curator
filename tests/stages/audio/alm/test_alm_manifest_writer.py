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

"""Tests for ALMManifestWriterStage."""

import json

from nemo_curator.stages.audio.alm import ALMManifestWriterStage
from nemo_curator.tasks import AudioBatch, FileGroupTask


class TestALMManifestWriter:
    """Unit tests for ALMManifestWriterStage."""

    def test_writes_entries_to_jsonl(self, tmp_path):
        out = tmp_path / "output.jsonl"
        writer = ALMManifestWriterStage(output_path=str(out))
        writer.setup()

        task = AudioBatch(
            data=[
                {"audio_filepath": "a.wav", "duration": 1.0},
                {"audio_filepath": "b.wav", "duration": 2.0},
            ],
            task_id="t1",
            dataset_name="ds",
        )
        writer.process(task)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["audio_filepath"] == "a.wav"
        assert json.loads(lines[1])["audio_filepath"] == "b.wav"

    def test_returns_file_group_task(self, tmp_path):
        out = tmp_path / "output.jsonl"
        writer = ALMManifestWriterStage(output_path=str(out))
        writer.setup()

        task = AudioBatch(data=[{"x": 1}], task_id="t1", dataset_name="ds")
        result = writer.process(task)

        assert isinstance(result, FileGroupTask)
        assert result.data == [str(out)]
        assert result.task_id == "t1"
        assert result.dataset_name == "ds"

    def test_propagates_metadata_and_stage_perf(self, tmp_path):
        out = tmp_path / "output.jsonl"
        writer = ALMManifestWriterStage(output_path=str(out))
        writer.setup()

        metadata = {"source_files": ["manifest.jsonl"]}
        stage_perf = {"some_stage": {"process_time": 0.5}}
        task = AudioBatch(
            data=[{"x": 1}],
            task_id="t1",
            dataset_name="ds",
            _metadata=metadata,
            _stage_perf=stage_perf,
        )
        result = writer.process(task)

        assert result._metadata == metadata
        assert result._stage_perf == stage_perf

    def test_appends_across_multiple_process_calls(self, tmp_path):
        out = tmp_path / "output.jsonl"
        writer = ALMManifestWriterStage(output_path=str(out))
        writer.setup()

        writer.process(AudioBatch(data=[{"entry": 1}], task_id="t1"))
        writer.process(AudioBatch(data=[{"entry": 2}], task_id="t2"))
        writer.process(AudioBatch(data=[{"entry": 3}], task_id="t3"))

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3
        assert [json.loads(l)["entry"] for l in lines] == [1, 2, 3]

    def test_setup_truncates_existing_file(self, tmp_path):
        out = tmp_path / "output.jsonl"
        out.write_text('{"old": "data"}\n')

        writer = ALMManifestWriterStage(output_path=str(out))
        writer.setup()

        assert out.read_text() == ""

    def test_setup_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "output.jsonl"
        writer = ALMManifestWriterStage(output_path=str(out))
        writer.setup()

        assert out.parent.exists()

    def test_handles_unicode_content(self, tmp_path):
        out = tmp_path / "output.jsonl"
        writer = ALMManifestWriterStage(output_path=str(out))
        writer.setup()

        task = AudioBatch(data=[{"text": "日本語テスト", "speaker": "Ñoño"}], task_id="t1")
        writer.process(task)

        loaded = json.loads(out.read_text().strip())
        assert loaded["text"] == "日本語テスト"
        assert loaded["speaker"] == "Ñoño"

    def test_preserves_nested_structures(self, tmp_path):
        out = tmp_path / "output.jsonl"
        writer = ALMManifestWriterStage(output_path=str(out))
        writer.setup()

        entry = {
            "audio_filepath": "a.wav",
            "windows": [
                {"segments": [{"start": 0.0, "end": 5.0, "speaker": "spk_0"}]},
            ],
            "stats": {"lost_bw": 3, "lost_sr": 0},
        }
        task = AudioBatch(data=[entry], task_id="t1")
        writer.process(task)

        loaded = json.loads(out.read_text().strip())
        assert loaded["windows"][0]["segments"][0]["speaker"] == "spk_0"
        assert loaded["stats"]["lost_bw"] == 3

    def test_empty_data_writes_nothing(self, tmp_path):
        out = tmp_path / "output.jsonl"
        writer = ALMManifestWriterStage(output_path=str(out))
        writer.setup()

        task = AudioBatch(data=[], task_id="t1")
        result = writer.process(task)

        assert out.read_text() == ""
        assert isinstance(result, FileGroupTask)

    def test_num_workers_returns_one(self):
        writer = ALMManifestWriterStage(output_path="/tmp/out.jsonl")
        assert writer.num_workers() == 1

    def test_xenna_stage_spec(self):
        writer = ALMManifestWriterStage(output_path="/tmp/out.jsonl")
        assert writer.xenna_stage_spec() == {"num_workers": 1}


class TestALMManifestWriterRoundTrip:
    """Round-trip test: write with writer, read back and verify."""

    def test_reader_writer_round_trip(self, sample_entries, tmp_path):
        from nemo_curator.stages.audio.alm import ALMManifestReaderStage
        from nemo_curator.tasks import FileGroupTask

        out = tmp_path / "round_trip.jsonl"

        writer = ALMManifestWriterStage(output_path=str(out))
        writer.setup()
        for i, entry in enumerate(sample_entries):
            task = AudioBatch(data=[entry], task_id=f"t{i}")
            writer.process(task)

        reader = ALMManifestReaderStage()
        result = reader.process(FileGroupTask(task_id="rt", dataset_name="rt", data=[str(out)]))

        assert len(result) == len(sample_entries)
        for orig, batch in zip(sample_entries, result):
            loaded = batch.data[0]
            assert loaded["audio_filepath"] == orig["audio_filepath"]
            assert len(loaded["segments"]) == len(orig["segments"])
