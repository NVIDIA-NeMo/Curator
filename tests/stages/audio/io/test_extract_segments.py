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

"""Tests for nemo_curator.stages.audio.io.extract_segments."""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.io.extract_segments import (
    SegmentExtractionStage,
    _base_metadata,
    _extract_scores,
    _get_speaker_label,
    _intervals_from_diar_segments,
    _intervals_from_timestamps,
    _read_segment,
    _write_metadata_csv,
    detect_combo,
    load_manifest,
    load_manifests,
)
from nemo_curator.stages.audio.postprocessing.timestamp_mapper import TimestampMapperStage
from nemo_curator.tasks import AudioTask

SAMPLE_RATE = 16000
DURATION_SEC = 5.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def wav_dir(tmp_path: Path) -> Path:
    """Create a temp directory with two 5-second mono WAV files."""
    for name in ("file_a", "file_b"):
        audio = np.random.default_rng(42).uniform(-0.5, 0.5, NUM_SAMPLES).astype(np.float32)
        sf.write(str(tmp_path / f"{name}.wav"), audio, SAMPLE_RATE)
    return tmp_path


def _wav_path(wav_dir: Path, name: str = "file_a") -> str:
    return str(wav_dir / f"{name}.wav")


def _write_manifest(path: Path, entries: list[dict]) -> str:
    manifest = str(path / "manifest.jsonl")
    with open(manifest, "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in entries)
    return manifest


# ------------------------------------------------------------------
# Pure helper functions
# ------------------------------------------------------------------


class TestDetectCombo:
    def test_empty_entries(self) -> None:
        assert detect_combo([]) == 2

    def test_no_speaker_no_diar(self) -> None:
        assert detect_combo([{"original_file": "/a.wav", "original_start_ms": 0}]) == 2

    def test_speaker_and_diar(self) -> None:
        assert detect_combo([{"speaker_id": "speaker_0", "diar_segments": [[1.0, 2.0]]}]) == 3

    def test_speaker_no_diar(self) -> None:
        assert detect_combo([{"speaker_id": "speaker_0", "original_start_ms": 0}]) == 4


class TestExtractScores:
    def test_filters_structural_keys_and_rounds(self) -> None:
        entry = {
            "original_file": "/a.wav",
            "duration": 5.0,
            "speaker_id": "speaker_0",
            "utmos_mos": 4.12345,
            "custom_field": "hello",
        }
        scores = _extract_scores(entry)
        assert "original_file" not in scores
        assert "duration" not in scores
        assert "speaker_id" not in scores
        assert scores["utmos_mos"] == 4.1235
        assert scores["custom_field"] == "hello"

    def test_empty_entry(self) -> None:
        assert _extract_scores({}) == {}


class TestGetSpeakerLabel:
    def test_standard_format(self) -> None:
        assert _get_speaker_label({"speaker_id": "speaker_2"}) == ("speaker_2", "2")

    def test_missing_speaker(self) -> None:
        assert _get_speaker_label({}) == ("unknown", "unknown")

    def test_non_standard_id(self) -> None:
        assert _get_speaker_label({"speaker_id": "alice"}) == ("alice", "alice")


class TestIntervalsFromTimestamps:
    def test_basic(self) -> None:
        assert _intervals_from_timestamps({"original_start_ms": 1000, "original_end_ms": 3000, "duration": 2.0}) == [
            (1000, 3000, 2.0)
        ]

    def test_computed_duration(self) -> None:
        assert _intervals_from_timestamps({"original_start_ms": 500, "original_end_ms": 2500}) == [(500, 2500, 2.0)]

    def test_missing_keys_default_zero(self) -> None:
        assert _intervals_from_timestamps({}) == [(0, 0, 0.0)]


class TestIntervalsFromDiarSegments:
    def test_basic(self) -> None:
        result = _intervals_from_diar_segments({"diar_segments": [[1.0, 2.5], [3.0, 4.0]]})
        assert result == [(1000, 2500, 1.5), (3000, 4000, 1.0)]

    def test_empty_diar(self) -> None:
        assert _intervals_from_diar_segments({"speaker_id": "speaker_0"}) == []

    def test_sorted_output(self) -> None:
        result = _intervals_from_diar_segments({"diar_segments": [[3.0, 4.0], [1.0, 2.0]]})
        assert result[0][0] < result[1][0]


class TestReadSegment:
    def test_reads_correct_slice(self, wav_dir: Path) -> None:
        filepath = _wav_path(wav_dir)
        original, sr = sf.read(filepath)
        expected = original[int(1.0 * sr) : int(2.0 * sr)]

        result = _read_segment(filepath, 1000, 2000, sr)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_full_file(self, wav_dir: Path) -> None:
        filepath = _wav_path(wav_dir)
        original, sr = sf.read(filepath)

        result = _read_segment(filepath, 0, 5000, sr)
        assert len(result) == len(original)

    def test_zero_duration(self, wav_dir: Path) -> None:
        result = _read_segment(_wav_path(wav_dir), 1000, 1000, SAMPLE_RATE)
        assert len(result) == 0


class TestBaseMetadata:
    def test_without_speaker(self) -> None:
        entry = {"original_start_ms": 0, "original_end_ms": 2000, "duration": 2.0, "utmos_mos": 4.1}
        row = _base_metadata("out.wav", "/a.wav", entry, 0, 0, 2000, 2.0)
        assert row["filename"] == "out.wav"
        assert row["start_sec"] == 0.0
        assert row["end_sec"] == 2.0
        assert row["utmos_mos"] == 4.1
        assert "speaker_id" not in row

    def test_with_speaker(self) -> None:
        entry = {"speaker_id": "speaker_0", "num_speakers": 3, "original_start_ms": 0, "original_end_ms": 1000}
        row = _base_metadata("out.wav", "/a.wav", entry, 0, 0, 1000, 1.0)
        assert row["speaker_id"] == "speaker_0"
        assert row["num_speakers"] == 3


# ------------------------------------------------------------------
# Manifest loading
# ------------------------------------------------------------------


class TestLoadManifest:
    def test_load_valid(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path, [{"a": 1}, {"b": 2}])
        entries = load_manifest(path)
        assert len(entries) == 2
        assert entries[0] == {"a": 1}

    def test_skip_empty_and_malformed_lines(self, tmp_path: Path) -> None:
        p = str(tmp_path / "m.jsonl")
        with open(p, "w") as f:
            f.write('{"a":1}\n\nNOT JSON\n{"b":2}\n')
        assert len(load_manifest(p)) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        p = str(tmp_path / "empty.jsonl")
        with open(p, "w") as f:
            f.write("")
        assert load_manifest(p) == []


class TestLoadManifests:
    def test_directory_of_jsonl(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        for i in range(3):
            with open(str(manifest_dir / f"part_{i}.jsonl"), "w") as f:
                f.write(json.dumps({"idx": i}) + "\n")
        out_dir = str(tmp_path / "out")
        entries = load_manifests(str(manifest_dir), out_dir)
        assert len(entries) == 3
        assert os.path.exists(os.path.join(out_dir, "manifest.jsonl"))

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        assert load_manifests(str(tmp_path / "nope"), str(tmp_path / "out")) == []


# ------------------------------------------------------------------
# CSV metadata output
# ------------------------------------------------------------------


class TestWriteMetadataCsv:
    def test_writes_csv(self, tmp_path: Path) -> None:
        rows = [
            {"filename": "a.wav", "duration": 1.0, "utmos_mos": 4.2},
            {"filename": "b.wav", "duration": 2.0, "sigmos_ovrl": 3.5},
        ]
        csv_path = _write_metadata_csv(str(tmp_path), rows)
        with open(csv_path) as f:
            read_rows = list(csv.DictReader(f))
        assert len(read_rows) == 2

    def test_empty_rows_no_file(self, tmp_path: Path) -> None:
        assert _write_metadata_csv(str(tmp_path), []) == ""


# ------------------------------------------------------------------
# SegmentExtractionStage — init & interface
# ------------------------------------------------------------------


class TestSegmentExtractionStageInit:
    def test_valid_construction(self, tmp_path: Path) -> None:
        stage = SegmentExtractionStage(output_dir=str(tmp_path), output_format="flac")
        assert stage.name == "SegmentExtraction"
        assert stage.output_format == "flac"
        assert stage.batch_size == 64

    def test_missing_output_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="output_dir is required"):
            SegmentExtractionStage(output_dir="")

    def test_invalid_format_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="output_format must be one of"):
            SegmentExtractionStage(output_dir=str(tmp_path), output_format="mp3")

    def test_process_raises_not_implemented(self, tmp_path: Path) -> None:
        stage = SegmentExtractionStage(output_dir=str(tmp_path))
        task = AudioTask(data={"original_file": "/a.wav"}, dataset_name="d")
        with pytest.raises(NotImplementedError):
            stage.process(task)

    def test_inputs_outputs(self, tmp_path: Path) -> None:
        stage = SegmentExtractionStage(output_dir=str(tmp_path))
        assert stage.inputs() == ([], ["original_file"])
        assert stage.outputs() == ([], ["extracted_path"])

    def test_worker_defaults(self, tmp_path: Path) -> None:
        stage = SegmentExtractionStage(output_dir=str(tmp_path))
        assert stage.num_workers() == 1
        assert stage.xenna_stage_spec() == {}


# ------------------------------------------------------------------
# SegmentExtractionStage — process_batch
# ------------------------------------------------------------------


class TestSegmentExtractionStageProcessBatch:
    def test_empty_batch(self, tmp_path: Path) -> None:
        stage = SegmentExtractionStage(output_dir=str(tmp_path / "out"))
        assert stage.process_batch([]) == []

    def test_combo2_timestamps(self, wav_dir: Path, tmp_path: Path) -> None:
        out_dir = str(tmp_path / "extracted")
        stage = SegmentExtractionStage(output_dir=out_dir)
        tasks = [
            AudioTask(
                data={
                    "original_file": _wav_path(wav_dir),
                    "original_start_ms": 0,
                    "original_end_ms": 2000,
                    "duration": 2.0,
                },
                dataset_name="test",
            ),
            AudioTask(
                data={
                    "original_file": _wav_path(wav_dir),
                    "original_start_ms": 2500,
                    "original_end_ms": 4500,
                    "duration": 2.0,
                },
                dataset_name="test",
            ),
        ]
        result = stage.process_batch(tasks)
        assert len(result) == 2
        assert os.path.exists(os.path.join(out_dir, "file_a_segment_000.wav"))
        assert os.path.exists(os.path.join(out_dir, "file_a_segment_001.wav"))

    def test_combo3_diar_segments(self, wav_dir: Path, tmp_path: Path) -> None:
        out_dir = str(tmp_path / "extracted")
        stage = SegmentExtractionStage(output_dir=out_dir)
        tasks = [
            AudioTask(
                data={
                    "original_file": _wav_path(wav_dir),
                    "speaker_id": "speaker_0",
                    "num_speakers": 2,
                    "diar_segments": [[0.5, 1.5], [2.0, 3.0]],
                },
                dataset_name="test",
            ),
        ]
        result = stage.process_batch(tasks)
        assert len(result) == 1
        assert os.path.exists(os.path.join(out_dir, "file_a_speaker_0_segment_000.wav"))
        assert os.path.exists(os.path.join(out_dir, "file_a_speaker_0_segment_001.wav"))

    def test_combo4_speaker_timestamps(self, wav_dir: Path, tmp_path: Path) -> None:
        out_dir = str(tmp_path / "extracted")
        stage = SegmentExtractionStage(output_dir=out_dir)
        tasks = [
            AudioTask(
                data={
                    "original_file": _wav_path(wav_dir),
                    "speaker_id": "speaker_0",
                    "original_start_ms": 0,
                    "original_end_ms": 1000,
                    "duration": 1.0,
                },
                dataset_name="test",
            ),
            AudioTask(
                data={
                    "original_file": _wav_path(wav_dir),
                    "speaker_id": "speaker_1",
                    "original_start_ms": 1500,
                    "original_end_ms": 2500,
                    "duration": 1.0,
                },
                dataset_name="test",
            ),
        ]
        result = stage.process_batch(tasks)
        assert len(result) == 2
        assert os.path.exists(os.path.join(out_dir, "file_a_speaker_0_segment_000.wav"))
        assert os.path.exists(os.path.join(out_dir, "file_a_speaker_1_segment_000.wav"))

    def test_flac_format(self, wav_dir: Path, tmp_path: Path) -> None:
        out_dir = str(tmp_path / "extracted")
        stage = SegmentExtractionStage(output_dir=out_dir, output_format="flac")
        tasks = [
            AudioTask(
                data={
                    "original_file": _wav_path(wav_dir),
                    "original_start_ms": 0,
                    "original_end_ms": 1000,
                    "duration": 1.0,
                },
                dataset_name="test",
            ),
        ]
        stage.process_batch(tasks)
        assert os.path.exists(os.path.join(out_dir, "file_a_segment_000.flac"))

    def test_missing_original_file_skipped(self, tmp_path: Path) -> None:
        out_dir = str(tmp_path / "extracted")
        stage = SegmentExtractionStage(output_dir=out_dir)
        tasks = [
            AudioTask(
                data={
                    "original_file": "/nonexistent/audio.wav",
                    "original_start_ms": 0,
                    "original_end_ms": 1000,
                    "duration": 1.0,
                },
                dataset_name="test",
            ),
        ]
        result = stage.process_batch(tasks)
        assert len(result) == 1
        assert not os.path.exists(os.path.join(out_dir, "metadata.csv"))

    def test_metadata_csv_written(self, wav_dir: Path, tmp_path: Path) -> None:
        out_dir = str(tmp_path / "extracted")
        stage = SegmentExtractionStage(output_dir=out_dir)
        tasks = [
            AudioTask(
                data={
                    "original_file": _wav_path(wav_dir),
                    "original_start_ms": 0,
                    "original_end_ms": 1000,
                    "duration": 1.0,
                    "utmos_mos": 4.2,
                },
                dataset_name="test",
            ),
        ]
        stage.process_batch(tasks)
        with open(os.path.join(out_dir, "metadata.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert float(rows[0]["utmos_mos"]) == 4.2

    def test_audio_content_matches_source(self, wav_dir: Path, tmp_path: Path) -> None:
        original, sr = sf.read(_wav_path(wav_dir))
        expected_slice = original[int(1.0 * sr) : int(2.0 * sr)]

        out_dir = str(tmp_path / "extracted")
        stage = SegmentExtractionStage(output_dir=out_dir)
        tasks = [
            AudioTask(
                data={
                    "original_file": _wav_path(wav_dir),
                    "original_start_ms": 1000,
                    "original_end_ms": 2000,
                    "duration": 1.0,
                },
                dataset_name="test",
            ),
        ]
        stage.process_batch(tasks)

        extracted, _ = sf.read(os.path.join(out_dir, "file_a_segment_000.wav"))
        np.testing.assert_array_almost_equal(extracted, expected_slice, decimal=4)


# ------------------------------------------------------------------
# SegmentExtractionStage — extract_from_manifest
# ------------------------------------------------------------------


class TestExtractFromManifest:
    def test_end_to_end(self, wav_dir: Path, tmp_path: Path) -> None:
        entries = [
            {
                "original_file": _wav_path(wav_dir),
                "original_start_ms": 0,
                "original_end_ms": 2000,
                "duration": 2.0,
                "utmos_mos": 4.0,
            },
            {"original_file": _wav_path(wav_dir), "original_start_ms": 2500, "original_end_ms": 4500, "duration": 2.0},
        ]
        manifest_path = _write_manifest(tmp_path, entries)
        out_dir = str(tmp_path / "output")

        stage = SegmentExtractionStage(output_dir=out_dir)
        stage.extract_from_manifest(manifest_path)

        assert os.path.exists(os.path.join(out_dir, "file_a_segment_000.wav"))
        assert os.path.exists(os.path.join(out_dir, "file_a_segment_001.wav"))
        assert os.path.exists(os.path.join(out_dir, "metadata.csv"))
        assert os.path.exists(os.path.join(out_dir, "extraction_summary.json"))

        with open(os.path.join(out_dir, "extraction_summary.json")) as f:
            summary = json.load(f)
        assert summary["total_segments"] == 2
        assert abs(summary["total_duration_sec"] - 4.0) < 0.01

    def test_empty_manifest(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path, [])
        out_dir = str(tmp_path / "output")
        stage = SegmentExtractionStage(output_dir=out_dir)
        stage.extract_from_manifest(manifest_path)
        assert not os.path.exists(os.path.join(out_dir, "extraction_summary.json"))

    def test_directory_input(self, wav_dir: Path, tmp_path: Path) -> None:
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        for i, start in enumerate([0, 2000]):
            with open(str(manifest_dir / f"part_{i}.jsonl"), "w") as f:
                f.write(
                    json.dumps(
                        {
                            "original_file": _wav_path(wav_dir),
                            "original_start_ms": start,
                            "original_end_ms": start + 1000,
                            "duration": 1.0,
                        }
                    )
                    + "\n"
                )

        out_dir = str(tmp_path / "output")
        stage = SegmentExtractionStage(output_dir=out_dir)
        stage.extract_from_manifest(str(manifest_dir))

        assert os.path.exists(os.path.join(out_dir, "manifest.jsonl"))
        with open(os.path.join(out_dir, "extraction_summary.json")) as f:
            assert json.load(f)["total_segments"] == 2


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("outer_speaker", [None, "stale_parent"])
def test_mapper_dictionary_turns_extract_per_speaker_in_mixed_batches(
    wav_dir: Path, tmp_path: Path, reverse: bool, outer_speaker: str | None
) -> None:
    source = _wav_path(wav_dir)
    data = {
        "audio_filepath": source,
        "diar_segments": [
            {"start": 0.125, "end": 0.375, "speaker": "speaker_0"},
            {"start": 1.125, "end": 1.375, "speaker": "speaker_1"},
        ],
    }
    if outer_speaker is not None:
        data["speaker_id"] = outer_speaker
    mapper = TimestampMapperStage()
    diar = mapper.process(AudioTask(data=data))
    scalar = mapper.process(AudioTask(data={"audio_filepath": source, "start_ms": 2000, "end_ms": 2250}))
    assert isinstance(diar, AudioTask)
    assert isinstance(scalar, AudioTask)
    tasks = [scalar, diar] if reverse else [diar, scalar]
    output_dir = tmp_path / "extracted"
    stage = SegmentExtractionStage(output_dir=str(output_dir))
    stage.process_batch(tasks)

    outputs = sorted(output_dir.glob("*.wav"))
    assert len(outputs) == 3
    original, sr = sf.read(source)
    expected = {
        "file_a_speaker_0_segment_000.wav": original[2000:6000],
        "file_a_speaker_1_segment_000.wav": original[18000:22000],
        "file_a_segment_000.wav": original[32000:36000],
    }
    for path in outputs:
        audio, actual_sr = sf.read(path)
        assert actual_sr == sr
        np.testing.assert_array_equal(audio, expected[path.name])
    with (output_dir / "metadata.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row.get("speaker_id", "") for row in rows} == {"", "speaker_0", "speaker_1"}


def test_renamed_mapper_turns_reach_real_extraction(wav_dir: Path, tmp_path: Path) -> None:
    source = _wav_path(wav_dir)
    mapper = TimestampMapperStage(start_key="begin", end_key="finish", speaker_key="person")
    output = mapper.process(
        AudioTask(
            data={
                "audio_filepath": source,
                "diar_segments": [{"begin": 0.125, "finish": 0.375, "person": "speaker_0"}],
            }
        )
    )
    assert isinstance(output, AudioTask)
    output_dir = tmp_path / "extracted"
    SegmentExtractionStage(output_dir=str(output_dir)).process_batch([output])
    audio, sr = sf.read(output_dir / "file_a_speaker_0_segment_000.wav")
    assert sr == SAMPLE_RATE
    assert len(audio) == 4000


@pytest.mark.parametrize("dictionary", [False, True])
def test_mapper_positive_submillisecond_turn_is_exported(wav_dir: Path, tmp_path: Path, dictionary: bool) -> None:
    source = _wav_path(wav_dir)
    segment = {"start": 0.1001, "end": 0.1002, "speaker": "speaker_0"} if dictionary else [0.1001, 0.1002]
    output = TimestampMapperStage().process(
        AudioTask(data={"audio_filepath": source, "speaker_id": "speaker_0", "diar_segments": [segment]})
    )
    assert isinstance(output, AudioTask)
    output_dir = tmp_path / "extracted"
    SegmentExtractionStage(output_dir=str(output_dir)).process_batch([output])
    audio, sr = sf.read(output_dir / "file_a_speaker_0_segment_000.wav")
    assert sr == SAMPLE_RATE
    assert len(audio) == 16


def test_precise_positive_dictionary_turn_is_not_rejected_by_float_validation() -> None:
    from decimal import Decimal

    start, end = Decimal("1.00000000000000000001"), Decimal("1.00000000000000000002")
    assert _intervals_from_diar_segments({"diar_segments": [{"start": start, "end": end}]}) == [(1000, 1001, 0.001)]


def test_precise_mapper_range_is_extracted_despite_whitelisted_source_turns(wav_dir: Path, tmp_path: Path) -> None:
    source = _wav_path(wav_dir)
    output = TimestampMapperStage(passthrough_keys=["diar_segments"]).process(
        AudioTask(
            data={
                "audio_filepath": "concat.wav",
                "start_ms": 200,
                "end_ms": 400,
                "diar_segments": [{"start": 0.1, "end": 0.8, "speaker": "speaker_0"}],
            },
            _metadata={
                "segment_mappings": [
                    {"concat_start_ms": 0, "concat_end_ms": 1000, "original_file": source, "original_start_ms": 1000}
                ]
            },
        )
    )
    assert isinstance(output, AudioTask)
    output_dir = tmp_path / "extracted"
    SegmentExtractionStage(output_dir=str(output_dir)).process_batch([output])
    paths = list(output_dir.glob("*.wav"))
    assert [path.name for path in paths] == ["file_a_segment_000.wav"]
    audio, sr = sf.read(paths[0])
    original, _ = sf.read(source)
    assert sr == SAMPLE_RATE
    np.testing.assert_array_equal(audio, original[19200:22400])


def test_mixed_extended_precision_turns_keep_chronological_order() -> None:
    from decimal import Decimal

    turns = [
        {"start": np.longdouble("1.25"), "end": np.longdouble("1.5")},
        {"start": Decimal("0.25"), "end": Decimal("0.5")},
    ]
    assert _intervals_from_diar_segments({"diar_segments": turns}) == [(250, 500, 0.25), (1250, 1500, 0.25)]


@pytest.mark.parametrize("diar_segments", [None, [], [[0.125, 0.375]]])
def test_optional_unlabelled_turns_preserve_main_timestamp_extraction(
    wav_dir: Path, tmp_path: Path, diar_segments: list | None
) -> None:
    source = _wav_path(wav_dir)
    entry = {
        "original_file": source,
        "original_start_ms": 1000,
        "original_end_ms": 1500,
        "duration": 0.5,
        "diar_segments": diar_segments,
    }
    assert detect_combo([entry]) == 2
    output_dir = tmp_path / "extracted"
    SegmentExtractionStage(output_dir=str(output_dir)).process_batch([AudioTask(data=entry)])
    paths = list(output_dir.glob("*.wav"))
    assert [path.name for path in paths] == ["file_a_segment_000.wav"]
    audio, sr = sf.read(paths[0])
    original, _ = sf.read(source)
    assert sr == SAMPLE_RATE
    np.testing.assert_array_equal(audio, original[16000:24000])
