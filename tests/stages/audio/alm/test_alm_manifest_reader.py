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

"""Tests for ALMManifestReaderStage and ALMManifestReader (CompositeStage)."""

import json
from pathlib import Path

from nemo_curator.stages.audio.alm import ALMManifestReader, ALMManifestReaderStage
from nemo_curator.tasks import AudioBatch, FileGroupTask


def _make_file_group_task(paths: list[str]) -> FileGroupTask:
    return FileGroupTask(task_id="test", dataset_name="test", data=paths)


class TestALMManifestReaderStage:
    """Unit tests for ALMManifestReaderStage (low-level stage)."""

    def test_reads_single_manifest(self, tmp_path: Path) -> None:
        entries = [
            {"audio_filepath": "a.wav", "audio_sample_rate": 16000, "segments": []},
            {"audio_filepath": "b.wav", "audio_sample_rate": 22050, "segments": []},
        ]
        manifest = tmp_path / "input.jsonl"
        manifest.write_text("\n".join(json.dumps(e) for e in entries))

        stage = ALMManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 2
        assert all(isinstance(r, AudioBatch) for r in result)
        assert result[0].data[0]["audio_filepath"] == "a.wav"
        assert result[1].data[0]["audio_filepath"] == "b.wav"

    def test_reads_multiple_manifests(self, tmp_path: Path) -> None:
        m1 = tmp_path / "m1.jsonl"
        m2 = tmp_path / "m2.jsonl"
        m1.write_text(json.dumps({"audio_filepath": "a.wav", "segments": []}))
        m2.write_text(json.dumps({"audio_filepath": "b.wav", "segments": []}))

        stage = ALMManifestReaderStage()
        result = stage.process(_make_file_group_task([str(m1), str(m2)]))

        assert len(result) == 2
        paths = [r.data[0]["audio_filepath"] for r in result]
        assert paths == ["a.wav", "b.wav"]

    def test_one_audio_batch_per_entry(self, tmp_path: Path) -> None:
        entries = [{"audio_filepath": f"{i}.wav", "segments": []} for i in range(5)]
        manifest = tmp_path / "input.jsonl"
        manifest.write_text("\n".join(json.dumps(e) for e in entries))

        stage = ALMManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 5
        for i, batch in enumerate(result):
            assert len(batch.data) == 1
            assert batch.data[0]["audio_filepath"] == f"{i}.wav"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(
            json.dumps({"audio_filepath": "a.wav", "segments": []})
            + "\n\n  \n"
            + json.dumps({"audio_filepath": "b.wav", "segments": []})
            + "\n"
        )

        stage = ALMManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 2

    def test_empty_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "empty.jsonl"
        manifest.write_text("")

        stage = ALMManifestReaderStage()
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

        stage = ALMManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        loaded = result[0].data[0]
        assert loaded["segments"][0]["metrics"]["bandwidth"] == 8000
        assert loaded["segments"][0]["speaker"] == "spk_0"

    def test_duplicate_manifests_for_repeat(self, tmp_path: Path) -> None:
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav", "segments": []}))

        stage = ALMManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)] * 3))

        assert len(result) == 3
        assert all(r.data[0]["audio_filepath"] == "a.wav" for r in result)


class TestALMManifestReaderComposite:
    """Tests for ALMManifestReader (CompositeStage)."""

    def test_decomposes_into_two_stages(self, tmp_path: Path) -> None:
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav", "segments": []}))

        composite = ALMManifestReader(manifest_path=str(tmp_path))
        stages = composite.decompose()

        assert len(stages) == 2
        assert stages[0].__class__.__name__ == "FilePartitioningStage"
        assert isinstance(stages[1], ALMManifestReaderStage)

    def test_accepts_list_of_paths(self) -> None:
        composite = ALMManifestReader(manifest_path=["/a.jsonl", "/b.jsonl"])
        stages = composite.decompose()
        assert stages[0].file_paths == ["/a.jsonl", "/b.jsonl"]

    def test_files_per_partition_default(self) -> None:
        composite = ALMManifestReader(manifest_path="/data")
        stages = composite.decompose()
        assert stages[0].files_per_partition == 1


class TestALMManifestReaderDirectory:
    """Tests for directory-based manifest discovery."""

    @staticmethod
    def _nested_dir() -> Path:
        return Path(__file__).parent.parent.parent.parent / "fixtures" / "audio" / "alm" / "nested_manifests"

    def test_reads_all_jsonl_from_directory(self) -> None:
        nested = self._nested_dir()
        all_files = sorted(str(p) for p in nested.rglob("*.jsonl"))
        stage = ALMManifestReaderStage()
        result = stage.process(_make_file_group_task(all_files))

        assert len(result) == 20  # 4 files × 5 entries each
        assert all(isinstance(r, AudioBatch) for r in result)

    def test_reads_from_subdirectory_a(self) -> None:
        subdir = self._nested_dir() / "subdir_a"
        files = sorted(str(p) for p in subdir.glob("*.jsonl"))
        stage = ALMManifestReaderStage()
        result = stage.process(_make_file_group_task(files))

        assert len(result) == 10  # 2 files × 5 entries each

    def test_reads_from_subdirectory_b(self) -> None:
        subdir = self._nested_dir() / "subdir_b"
        files = sorted(str(p) for p in subdir.glob("*.jsonl"))
        stage = ALMManifestReaderStage()
        result = stage.process(_make_file_group_task(files))

        assert len(result) == 10  # 2 files × 5 entries each

    def test_composite_discovers_nested_directory(self) -> None:
        nested = self._nested_dir()
        composite = ALMManifestReader(manifest_path=str(nested))
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


class TestALMManifestReaderIntegration:
    """Integration tests using real sample fixtures."""

    def test_reads_sample_fixture(self) -> None:
        fixture = Path(__file__).parent.parent.parent.parent / "fixtures" / "audio" / "alm" / "sample_input.jsonl"
        stage = ALMManifestReaderStage()
        result = stage.process(_make_file_group_task([str(fixture)]))

        assert len(result) == 5
        for batch in result:
            assert isinstance(batch, AudioBatch)
            assert len(batch.data) == 1
            entry = batch.data[0]
            assert "audio_filepath" in entry
            assert "segments" in entry
            assert len(entry["segments"]) > 0

    def test_composite_end_to_end_with_directory(self) -> None:
        """End-to-end: ALMManifestReader composite with directory input through full pipeline."""
        from nemo_curator.backends.xenna import XennaExecutor
        from nemo_curator.pipeline import Pipeline
        from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage

        nested = Path(__file__).parent.parent.parent.parent / "fixtures" / "audio" / "alm" / "nested_manifests"

        pipeline = Pipeline(name="test_dir_e2e", description="Directory discovery end-to-end test")
        pipeline.add_stage(ALMManifestReader(manifest_path=str(nested)))
        pipeline.add_stage(ALMDataBuilderStage(
            target_window_duration=120.0,
            tolerance=0.1,
            min_sample_rate=16000,
            min_bandwidth=8000,
            min_speakers=2,
            max_speakers=5,
        ))
        pipeline.add_stage(ALMDataOverlapStage(overlap_percentage=50, target_duration=120.0))

        executor = XennaExecutor()
        results = pipeline.run(executor)

        output_entries = []
        for task in results or []:
            output_entries.extend(task.data)

        assert len(output_entries) == 20  # 4 files × 5 entries
        total_windows = sum(len(e.get("filtered_windows", [])) for e in output_entries)
        assert total_windows == 100  # 25 per file × 4 files
        total_dur = sum(e.get("filtered_dur", 0) for e in output_entries)
        assert abs(total_dur - 12142.0) < 1.0
