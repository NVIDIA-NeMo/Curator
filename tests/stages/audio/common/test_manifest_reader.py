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
from pathlib import Path

import pytest

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.stages.audio.common import (
    ManifestReader,
    ManifestReaderStage,
    ManifestWriterStage,
)
from nemo_curator.tasks import AudioTask, FileGroupTask
from tests import FIXTURES_DIR

ALM_FIXTURES_DIR = FIXTURES_DIR / "audio" / "alm"


def _make_file_group_task(paths: list[str]) -> FileGroupTask:
    return FileGroupTask(dataset_name="test", data=paths)


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

    def test_worker_defaults(self) -> None:
        stage = ManifestReaderStage()
        assert stage.num_workers() == 1
        assert stage.ray_stage_spec()["is_fanout_stage"] is True
        assert stage.xenna_stage_spec() == {}

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


def _write_rows(path: Path, rows: list) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _long_form(**kwargs: object) -> ManifestReaderStage:
    """The reader as the ALM pretrain pipeline configures it."""
    options = {"id_key": "id", "segments_key": "segments", "audio_dir": "/data", "skip_invalid_rows": True}
    return ManifestReaderStage(**{**options, **kwargs})


class TestManifestReaderLongFormOptions:
    """The options that let ManifestReader read the ALM pretrain long-form manifest."""

    def test_defaults_leave_rows_and_contract_unchanged(self, tmp_path: Path) -> None:
        manifest = _write_rows(tmp_path / "in.jsonl", [{"id": 7, "audio_filepath": "./a.wav"}])
        stage = ManifestReaderStage()
        out = stage.process(_make_file_group_task([manifest]))
        assert out[0].data == {"id": 7, "audio_filepath": "./a.wav"}
        assert out[0].dataset_name == "test"
        assert out[0].filepath_key is None
        contract = build_contract(stage)
        assert contract.writes.data_keys == ["audio_filepath"]
        assert contract.conditional_writes == []
        assert contract.gates.per_row_independent is True

    def test_long_form_row_is_stringified_reanchored_and_stamped(self, tmp_path: Path) -> None:
        manifest = _write_rows(tmp_path / "in.jsonl", [{"id": 7, "audio_filepath": "./a.wav"}])
        out = _long_form(dataset_name="long_form_audio").process(_make_file_group_task([manifest]))
        # ``id`` leaves the reader as a string (snippet ids and tar member names embed it).
        assert out[0].data == {"id": "7", "audio_filepath": "/data/a.wav"}
        assert out[0].dataset_name == "long_form_audio"
        assert out[0].filepath_key == "audio_filepath"

    def test_skips_missing_empty_and_duplicate_ids_and_missing_paths(self, tmp_path: Path) -> None:
        manifest = _write_rows(
            tmp_path / "in.jsonl",
            [
                {"audio_filepath": "./a.wav"},
                {"id": "", "audio_filepath": "./b.wav"},
                {"id": "  ", "audio_filepath": "./c.wav"},
                {"id": "A", "audio_filepath": "./d.wav"},
                {"id": "A", "audio_filepath": "./e.wav"},
                {"id": "B"},
                {"id": "C", "audio_filepath": "./f.wav"},
            ],
        )
        out = _long_form().process(_make_file_group_task([manifest]))
        assert [(t.data["id"], t.data["audio_filepath"]) for t in out] == [("A", "/data/d.wav"), ("C", "/data/f.wav")]

    def test_ids_are_unique_across_manifests_read_by_one_stage(self, tmp_path: Path) -> None:
        m1 = _write_rows(tmp_path / "m1.jsonl", [{"id": "A", "audio_filepath": "a.wav"}])
        m2 = _write_rows(tmp_path / "m2.jsonl", [{"id": "A", "audio_filepath": "b.wav"}])
        stage = _long_form()
        assert len(stage.process(_make_file_group_task([m1]))) == 1
        assert stage.process(_make_file_group_task([m2])) == []

    def test_basename_mode_rejects_duplicate_basenames(self, tmp_path: Path) -> None:
        manifest = _write_rows(
            tmp_path / "in.jsonl",
            [{"id": "A", "audio_filepath": "./shard1/foo.wav"}, {"id": "B", "audio_filepath": "./shard2/foo.wav"}],
        )
        with pytest.raises(ValueError, match="duplicate audio basename"):
            _long_form().process(_make_file_group_task([manifest]))

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("relative", ["/data/shard1/foo.wav", "/data/shard2/foo.wav", "/abs/b.wav"]),
            ("as_is", ["shard1/foo.wav", "shard2/foo.wav", "/abs/b.wav"]),
        ],
    )
    def test_relative_and_as_is_modes(self, tmp_path: Path, mode: str, expected: list[str]) -> None:
        manifest = _write_rows(
            tmp_path / "in.jsonl",
            [
                {"id": "A", "audio_filepath": "shard1/foo.wav"},
                {"id": "B", "audio_filepath": "shard2/foo.wav"},
                {"id": "C", "audio_filepath": "/abs/b.wav"},
            ],
        )
        out = _long_form(audio_path_resolution=mode).process(_make_file_group_task([manifest]))
        assert [t.data["audio_filepath"] for t in out] == expected

    def test_invalid_configuration_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown audio_path_resolution"):
            ManifestReaderStage(audio_dir="/data", audio_path_resolution="not_a_mode")

    def test_invalid_json_raises_unless_skipped(self, tmp_path: Path) -> None:
        manifest = tmp_path / "in.jsonl"
        manifest.write_text('{"id": "A", "audio_filepath": "a.wav"}\n{not json\n', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            ManifestReaderStage().process(_make_file_group_task([str(manifest)]))
        assert len(_long_form().process(_make_file_group_task([str(manifest)]))) == 1

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _long_form().process(_make_file_group_task([str(tmp_path / "nope.jsonl")]))

    def test_contract_matches_emitted_schema(self, tmp_path: Path) -> None:
        manifest = _write_rows(
            tmp_path / "in.jsonl", [{"source_id": "A", "source_path": "a.wav", "turns": [{"start": 0, "end": 1}]}]
        )
        stage = _long_form(
            include_files_key="source_path",
            id_key="source_id",
            segments_key="turns",
        )
        contract = build_contract(stage)
        declared = set(contract.writes.data_keys) | {
            k for w in contract.conditional_writes for k in w.writes.data_keys
        }
        assert declared == {"source_path", "source_id", "turns"}
        assert "turns" not in contract.writes.data_keys
        assert contract.gates.per_row_independent is False
        assert static_contract(ManifestReaderStage).gates.per_row_independent is True
        assert stage.process(_make_file_group_task([manifest]))[0].data.keys() == declared

    @pytest.mark.parametrize(
        ("options", "independent"),
        [
            ({"id_key": "id"}, False),
            ({"audio_dir": "/data"}, False),
            ({"audio_dir": "/data", "audio_path_resolution": "relative"}, True),
            ({"segments_key": "segments"}, True),
        ],
    )
    def test_cross_row_checks_make_reader_not_row_independent(self, options: dict, independent: bool) -> None:
        assert build_contract(ManifestReaderStage(**options)).gates.per_row_independent is independent
        composite = ManifestReader(manifest_path="unused.jsonl", **options)
        assert build_contract(composite).gates.per_row_independent is independent

    def test_agent_ready_with_long_form_options(self, tmp_path: Path) -> None:
        manifest = _write_rows(
            tmp_path / "agent-ready.jsonl",
            [
                {"id": "A", "audio_filepath": "./a.wav", "segments": []},
                {"id": "B", "audio_filepath": "./b.wav"},
                {"id": "C", "audio_filepath": "./c.wav", "segments": [{"start": 0, "end": 1}]},
            ],
        )
        assert_agent_ready(
            _long_form(),
            lambda: _make_file_group_task([manifest]),
            expected_cardinality="1:N fan-out",
            available_keys=set(),
        )

    def test_composite_forwards_every_option(self) -> None:
        options = {
            "include_files_key": "path",
            "id_key": "uid",
            "segments_key": "turns",
            "audio_dir": "/data",
            "audio_path_resolution": "relative",
            "dataset_name": "ds",
            "skip_invalid_rows": True,
        }
        reader = ManifestReader(manifest_path="m.jsonl", **options).decompose()[1]
        assert {name: getattr(reader, name) for name in options} == options


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


class TestManifestWriterRoundTrip:
    """Round-trip test: write with writer, read back and verify."""

    def test_reader_writer_round_trip(self, sample_entries: list[dict], tmp_path: Path) -> None:
        out = tmp_path / "round_trip.jsonl"

        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()
        for _i, entry in enumerate(sample_entries):
            task = AudioTask(data=entry)
            writer.process(task)

        reader = ManifestReaderStage()
        result = reader.process(FileGroupTask(dataset_name="rt", data=[str(out)]))

        assert len(result) == len(sample_entries)
        for orig, audio_entry in zip(sample_entries, result, strict=True):
            loaded = audio_entry.data
            assert loaded["audio_filepath"] == orig["audio_filepath"]
            assert len(loaded["segments"]) == len(orig["segments"])
