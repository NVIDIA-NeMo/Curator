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

"""Tests for common audio stages: GetAudioDurationStage, PreserveByValueStage,
ManifestReaderStage, ManifestReader, and ManifestWriterStage."""

import json
from pathlib import Path
from unittest import mock

import pytest

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    ManifestReader,
    ManifestReaderStage,
    ManifestWriterStage,
    PreserveByValueStage,
)
from nemo_curator.tasks import AudioTask, FileGroupTask
from tests import FIXTURES_DIR

ALM_FIXTURES_DIR = FIXTURES_DIR / "audio" / "alm"


def _make_file_group_task(paths: list[str]) -> FileGroupTask:
    return FileGroupTask(task_id="test", dataset_name="test", data=paths)


# ---------------------------------------------------------------------------
# PreserveByValueStage
# ---------------------------------------------------------------------------


def test_preserve_by_value_validate_input_valid() -> None:
    stage = PreserveByValueStage(input_value_key="wer", target_value=50, operator="le")
    assert stage.validate_input(AudioTask(data={"wer": 30})) is True


def test_preserve_by_value_validate_input_missing_column() -> None:
    stage = PreserveByValueStage(input_value_key="wer", target_value=50, operator="le")
    assert stage.validate_input(AudioTask(data={"text": "hello"})) is False


def test_preserve_by_value_process_raises_not_implemented() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    with pytest.raises(NotImplementedError, match="only supports process_batch"):
        stage.process(AudioTask(data={"v": 3}))


def test_preserve_by_value_process_batch_raises_on_missing_column() -> None:
    stage = PreserveByValueStage(input_value_key="wer", target_value=50, operator="le")
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"text": "hello"})])


def test_preserve_by_value_eq_keeps_match() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    result = stage.process_batch([AudioTask(data={"v": 3})])
    assert len(result) == 1
    assert isinstance(result[0], AudioTask)
    assert result[0].data["v"] == 3


def test_preserve_by_value_eq_filters_non_match() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    result = stage.process_batch([AudioTask(data={"v": 1})])
    assert len(result) == 0


def test_preserve_by_value_lt() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=5, operator="lt")
    assert len(stage.process_batch([AudioTask(data={"v": 2})])) == 1
    assert len(stage.process_batch([AudioTask(data={"v": 7})])) == 0


def test_preserve_by_value_ge() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=10, operator="ge")
    assert len(stage.process_batch([AudioTask(data={"v": 9})])) == 0
    assert len(stage.process_batch([AudioTask(data={"v": 10})])) == 1
    assert len(stage.process_batch([AudioTask(data={"v": 11})])) == 1


# ---------------------------------------------------------------------------
# GetAudioDurationStage
# ---------------------------------------------------------------------------


def test_get_audio_duration_validate_input_valid() -> None:
    stage = GetAudioDurationStage()
    assert stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"})) is True


def test_get_audio_duration_validate_input_missing_column() -> None:
    stage = GetAudioDurationStage()
    assert stage.validate_input(AudioTask(data={"text": "hello"})) is False


def test_get_audio_duration_process_batch_raises_on_missing_column() -> None:
    stage = GetAudioDurationStage()
    stage.setup()
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"text": "hello"})])


def test_get_audio_duration_success(tmp_path: Path) -> None:
    class FakeInfo:
        def __init__(self, frames: int, samplerate: int):
            self.frames = frames
            self.samplerate = samplerate

    fake_info = FakeInfo(frames=16000 * 2, samplerate=16000)
    with mock.patch("soundfile.info", return_value=fake_info):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        stage.setup()
        entry = AudioTask(data={"audio_filepath": (tmp_path / "fake.wav").as_posix()})
        result = stage.process(entry)
        assert isinstance(result, AudioTask)
        assert result.data["duration"] == 2.0


def test_get_audio_duration_error_sets_minus_one(tmp_path: Path) -> None:
    with mock.patch("soundfile.info", side_effect=RuntimeError("bad file")):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        stage.setup()
        entry = AudioTask(data={"audio_filepath": (tmp_path / "missing.wav").as_posix()})
        result = stage.process(entry)
        assert result.data["duration"] == -1.0


# ---------------------------------------------------------------------------
# ManifestReaderStage
# ---------------------------------------------------------------------------


class TestManifestReaderStage:
    """Unit tests for ManifestReaderStage (low-level stage)."""

    def test_reads_single_manifest(self, tmp_path: Path) -> None:
        entries = [
            {"audio_filepath": "a.wav", "audio_sample_rate": 16000, "segments": []},
            {"audio_filepath": "b.wav", "audio_sample_rate": 22050, "segments": []},
        ]
        manifest = tmp_path / "input.jsonl"
        manifest.write_text("\n".join(json.dumps(e) for e in entries))

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 2
        assert all(isinstance(r, AudioTask) for r in result)
        assert result[0].data["audio_filepath"] == "a.wav"
        assert result[1].data["audio_filepath"] == "b.wav"

    def test_reads_multiple_manifests(self, tmp_path: Path) -> None:
        m1 = tmp_path / "m1.jsonl"
        m2 = tmp_path / "m2.jsonl"
        m1.write_text(json.dumps({"audio_filepath": "a.wav", "segments": []}))
        m2.write_text(json.dumps({"audio_filepath": "b.wav", "segments": []}))

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(m1), str(m2)]))

        assert len(result) == 2
        paths = [r.data["audio_filepath"] for r in result]
        assert paths == ["a.wav", "b.wav"]

    def test_one_audio_entry_per_line(self, tmp_path: Path) -> None:
        entries = [{"audio_filepath": f"{i}.wav", "segments": []} for i in range(5)]
        manifest = tmp_path / "input.jsonl"
        manifest.write_text("\n".join(json.dumps(e) for e in entries))

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 5
        for i, audio_entry in enumerate(result):
            assert isinstance(audio_entry, AudioTask)
            assert audio_entry.data["audio_filepath"] == f"{i}.wav"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(
            json.dumps({"audio_filepath": "a.wav", "segments": []})
            + "\n\n  \n"
            + json.dumps({"audio_filepath": "b.wav", "segments": []})
            + "\n"
        )

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 2

    def test_empty_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "empty.jsonl"
        manifest.write_text("")

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert result == []

    def test_preserves_nested_data(self, tmp_path: Path) -> None:
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

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        loaded = result[0].data
        assert loaded["segments"][0]["metrics"]["bandwidth"] == 8000
        assert loaded["segments"][0]["speaker"] == "spk_0"

    def test_duplicate_manifests_for_repeat(self, tmp_path: Path) -> None:
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav", "segments": []}))

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)] * 3))

        assert len(result) == 3
        assert all(r.data["audio_filepath"] == "a.wav" for r in result)


class TestManifestReaderDirectory:
    """Tests for directory-based manifest discovery."""

    @staticmethod
    def _nested_dir() -> Path:
        return ALM_FIXTURES_DIR / "nested_manifests"

    def test_reads_all_jsonl_from_directory(self) -> None:
        nested = self._nested_dir()
        all_files = sorted(str(p) for p in nested.rglob("*.jsonl"))
        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task(all_files))

        assert len(result) == 20  # 4 files x 5 entries each
        assert all(isinstance(r, AudioTask) for r in result)

    def test_reads_from_subdirectory_a(self) -> None:
        subdir = self._nested_dir() / "subdir_a"
        files = sorted(str(p) for p in subdir.glob("*.jsonl"))
        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task(files))

        assert len(result) == 10  # 2 files x 5 entries each

    def test_reads_from_subdirectory_b(self) -> None:
        subdir = self._nested_dir() / "subdir_b"
        files = sorted(str(p) for p in subdir.glob("*.jsonl"))
        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task(files))

        assert len(result) == 10  # 2 files x 5 entries each

    def test_composite_discovers_nested_directory(self) -> None:
        nested = self._nested_dir()
        composite = ManifestReader(manifest_path=str(nested))
        stages = composite.decompose()

        partitioner = stages[0]
        assert partitioner.file_paths == str(nested)
        assert partitioner.file_extensions == [".jsonl", ".json"]

    def test_ignores_non_jsonl_files(self) -> None:
        nested = self._nested_dir()
        txt_files = list(nested.rglob("*.txt"))
        assert len(txt_files) > 0, "Test setup: .txt file should exist"

        jsonl_files = sorted(str(p) for p in nested.rglob("*.jsonl"))
        for f in jsonl_files:
            assert not f.endswith(".txt")


class TestManifestReaderIntegration:
    """Integration tests using real sample fixtures."""

    def test_reads_sample_fixture(self) -> None:
        fixture = ALM_FIXTURES_DIR / "sample_input.jsonl"
        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(fixture)]))

        assert len(result) == 5
        for audio_entry in result:
            assert isinstance(audio_entry, AudioTask)
            entry_data = audio_entry.data
            assert "audio_filepath" in entry_data
            assert "segments" in entry_data
            assert len(entry_data["segments"]) > 0

    def test_composite_end_to_end_with_directory(self) -> None:
        """End-to-end: ManifestReader composite with directory input through full pipeline."""
        nested = ALM_FIXTURES_DIR / "nested_manifests"

        pipeline = Pipeline(name="test_dir_e2e", description="Directory discovery end-to-end test")
        pipeline.add_stage(ManifestReader(manifest_path=str(nested)))
        pipeline.add_stage(
            ALMDataBuilderStage(
                target_window_duration=120.0,
                tolerance=0.1,
                min_sample_rate=16000,
                min_bandwidth=8000,
                min_speakers=2,
                max_speakers=5,
            )
        )
        pipeline.add_stage(ALMDataOverlapStage(overlap_percentage=50, target_duration=120.0))

        executor = XennaExecutor()
        results = pipeline.run(executor)

        output_entries = []
        for task in results or []:
            output_entries.append(task.data)

        assert len(output_entries) == 20  # 4 files x 5 entries
        total_windows = sum(len(e.get("filtered_windows", [])) for e in output_entries)
        assert total_windows == 100  # 25 per file x 4 files
        total_dur = sum(e.get("filtered_dur", 0) for e in output_entries)
        assert abs(total_dur - 12142.0) < 1.0


# ---------------------------------------------------------------------------
# ManifestWriterStage
# ---------------------------------------------------------------------------


class TestManifestWriterStage:
    """Unit tests for ManifestWriterStage."""

    def test_writes_entry_to_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        task = AudioTask(
            data={"audio_filepath": "a.wav", "duration": 1.0},
            task_id="t1",
            dataset_name="ds",
        )
        writer.process(task)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["audio_filepath"] == "a.wav"

    def test_returns_audio_task(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        task = AudioTask(data={"x": 1}, task_id="t1", dataset_name="ds")
        result = writer.process(task)

        assert isinstance(result, AudioTask)
        assert result.data == {"x": 1}
        assert result.task_id == "t1"
        assert result.dataset_name == "ds"

    def test_propagates_metadata_and_stage_perf(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        metadata = {"source_files": ["manifest.jsonl"]}
        stage_perf = [{"stage": "some_stage", "process_time": 0.5}]
        task = AudioTask(
            data={"x": 1},
            task_id="t1",
            dataset_name="ds",
            _metadata=metadata,
            _stage_perf=stage_perf,
        )
        result = writer.process(task)

        assert result._metadata == metadata
        assert result._stage_perf == stage_perf

    def test_appends_across_multiple_process_calls(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        writer.process(AudioTask(data={"entry": 1}, task_id="t1"))
        writer.process(AudioTask(data={"entry": 2}, task_id="t2"))
        writer.process(AudioTask(data={"entry": 3}, task_id="t3"))

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3
        assert [json.loads(line)["entry"] for line in lines] == [1, 2, 3]

    def test_setup_truncates_existing_file(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        out.write_text('{"old": "data"}\n')

        writer = ManifestWriterStage(output_path=str(out))
        writer.setup()

        assert out.read_text() == ""

    def test_setup_on_node_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()

        assert out.parent.exists()

    def test_handles_unicode_content(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        task = AudioTask(data={"text": "日本語テスト", "speaker": "Ñoño"}, task_id="t1")
        writer.process(task)

        loaded = json.loads(out.read_text().strip())
        assert loaded["text"] == "日本語テスト"
        assert loaded["speaker"] == "Ñoño"

    def test_preserves_nested_structures(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        entry = {
            "audio_filepath": "a.wav",
            "windows": [
                {"segments": [{"start": 0.0, "end": 5.0, "speaker": "spk_0"}]},
            ],
            "stats": {"lost_bw": 3, "lost_sr": 0},
        }
        task = AudioTask(data=entry, task_id="t1")
        writer.process(task)

        loaded = json.loads(out.read_text().strip())
        assert loaded["windows"][0]["segments"][0]["speaker"] == "spk_0"
        assert loaded["stats"]["lost_bw"] == 3

    def test_num_workers_returns_one(self, tmp_path: Path) -> None:
        writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
        assert writer.num_workers() == 1

    def test_xenna_stage_spec(self, tmp_path: Path) -> None:
        writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
        assert writer.xenna_stage_spec() == {"num_workers": 1}


class TestManifestWriterRoundTrip:
    """Round-trip test: write with writer, read back and verify."""

    def test_reader_writer_round_trip(self, sample_entries: list[dict], tmp_path: Path) -> None:
        out = tmp_path / "round_trip.jsonl"

        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()
        for i, entry in enumerate(sample_entries):
            task = AudioTask(data=entry, task_id=f"t{i}")
            writer.process(task)

        reader = ManifestReaderStage()
        result = reader.process(FileGroupTask(task_id="rt", dataset_name="rt", data=[str(out)]))

        assert len(result) == len(sample_entries)
        for orig, audio_entry in zip(sample_entries, result, strict=True):
            loaded = audio_entry.data
            assert loaded["audio_filepath"] == orig["audio_filepath"]
            assert len(loaded["segments"]) == len(orig["segments"])
