# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from textgrid import IntervalTier, TextGrid

from tutorials.audio.vad_refinement import rectify_vad_boundaries as refinement
from tutorials.audio.vad_refinement.rectify_vad_boundaries import (
    SortformerVADConfig,
    VADPredictions,
    _sortformer_frame_duration,
    atomic_vote_consensus,
    frame_probabilities_to_intervals,
    load_manifest,
    parse_args,
    process_manifest,
    project_consensus_to_words,
    read_rttm,
    refine_textgrid_boundaries,
)
from tutorials.audio.vad_refinement.textgrid_rttm import write_refined_textgrids


class FakeDetector:
    metadata: ClassVar[dict[str, object]] = {"name": "synthetic-test-detector"}

    def __init__(self, predictions: dict[str, VADPredictions]) -> None:
        self.predictions = predictions

    def __call__(self, audio_path: Path) -> VADPredictions:
        return self.predictions[audio_path.name]


def _write_textgrid(path: Path, words: list[tuple[float, float, str]], duration: float) -> None:
    textgrid = TextGrid(minTime=0.0, maxTime=duration)
    tier = IntervalTier(name="words", minTime=0.0, maxTime=duration)
    for start, end, word in words:
        tier.add(start, end, word)
    textgrid.append(tier)
    textgrid.write(str(path))


def test_atomic_vote_uses_last_shared_vad_boundary() -> None:
    words: list[list[float | str]] = [
        [7.95, 8.61, "many"],
        [8.83, 9.20, "next"],
    ]
    consensus, atomic_regions = atomic_vote_consensus(
        words,
        silero_intervals=[(7.80, 8.80)],
        sortformer_intervals=[(7.80, 8.73)],
        duration=10.0,
    )
    projected, changes, _ = project_consensus_to_words(words, consensus, offset=0.0)

    assert projected[0][1] == pytest.approx(8.73)
    assert projected[1][0] == pytest.approx(8.83)
    assert changes[0]["word"] == "many"
    assert atomic_regions > 0


def test_sortformer_frame_postprocessing_uses_hysteresis_and_joins_short_gap() -> None:
    intervals = frame_probabilities_to_intervals(
        [0.1, 0.7, 0.8, 0.2, 0.1, 0.8, 0.8, 0.1],
        frame_duration=0.02,
        audio_duration=0.16,
        config=SortformerVADConfig(
            onset=0.6,
            offset=0.4,
            min_speech_duration_ms=0,
            min_silence_duration_ms=50,
            speech_pad_ms=0,
        ),
    )

    assert intervals[0] == pytest.approx((0.02, 0.14))


def test_cli_requires_sortformer_as_the_only_secondary_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vad-refine",
            "--manifest",
            "session.jsonl",
            "--output-root",
            "output",
            "--sortformer-model",
            "sortformer.nemo",
        ],
    )

    args = parse_args()

    assert args.sortformer_model == "sortformer.nemo"
    assert not hasattr(args, "secondary_vad")
    assert not hasattr(args, "nemo_model")
    assert not hasattr(args, "maximum_word_pause")
    assert not hasattr(args, "fusion_mode")
    assert not hasattr(args, "final_activity_mode")


def test_cli_rejects_removed_heuristic_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vad-refine",
            "--manifest",
            "session.jsonl",
            "--output-root",
            "output",
            "--sortformer-model",
            "sortformer.nemo",
            "--maximum-word-pause",
            "0.2",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_atomic_vote_never_contracts_textgrid(tmp_path: Path) -> None:
    source = tmp_path / "segment.TextGrid"
    _write_textgrid(source, [(0.1, 0.9, "speech")], 1.0)

    refined_path, changes, _ = refine_textgrid_boundaries(
        {
            "session_id": "session",
            "speaker_id": "speaker",
            "offset": 0.0,
            "duration": 1.0,
            "mfa_textgrid": str(source),
        },
        VADPredictions(
            silero=((0.2, 0.8),),
            secondary=((0.22, 0.82),),
            secondary_name="sortformer",
        ),
    )

    words = [
        interval for interval in TextGrid.fromFile(str(refined_path)).getFirst("words").intervals if interval.mark
    ]
    assert words[0].minTime == pytest.approx(0.1)
    assert words[0].maxTime == pytest.approx(0.9)
    assert changes == []


def test_atomic_vote_requires_both_vads_outside_words() -> None:
    words: list[list[float | str]] = [
        [0.1, 0.4, "first"],
        [0.7, 0.9, "second"],
    ]
    consensus, _ = atomic_vote_consensus(
        words,
        silero_intervals=[(0.0, 1.0)],
        sortformer_intervals=[(0.1, 0.4), (0.7, 0.9)],
        duration=1.0,
    )
    projected, changes, _ = project_consensus_to_words(words, consensus, offset=0.0)

    assert projected == words
    assert changes == []


def test_session_textgrid_uses_one_tier_per_speaker(tmp_path: Path) -> None:
    first = tmp_path / "first.TextGrid"
    second = tmp_path / "second.TextGrid"
    _write_textgrid(first, [(0.0, 1.0, "first")], 1.0)
    _write_textgrid(second, [(0.0, 1.0, "second")], 1.0)

    write_refined_textgrids(
        [
            {
                "session_id": "meeting-001",
                "speaker_id": "speaker-00",
                "offset": 0.0,
                "duration": 1.0,
                "rttm_textgrid": str(first),
            },
            {
                "session_id": "meeting-001",
                "speaker_id": "speaker-01",
                "offset": 0.0,
                "duration": 1.0,
                "rttm_textgrid": str(second),
            },
        ],
        tmp_path / "combined",
        "meeting-001",
    )

    combined = TextGrid.fromFile(str(tmp_path / "combined" / "meeting-001_vad_refined.TextGrid"))
    assert [tier.name for tier in combined.tiers] == ["speaker-00", "speaker-01"]


def test_manifest_rejects_path_traversal_session_id(tmp_path: Path) -> None:
    manifest = tmp_path / "unsafe.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "session_id": "../outside",
                "speaker_id": "speaker-00",
                "audio_filepath": "audio.wav",
                "offset": 0.0,
                "duration": 1.0,
                "mfa_textgrid": "segment.TextGrid",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="path separators"):
        load_manifest(manifest)


def test_pipeline_writes_non_destructive_outputs(tmp_path: Path) -> None:
    audio = tmp_path / "speaker.wav"
    audio.touch()
    textgrid_path = tmp_path / "segment.TextGrid"
    _write_textgrid(
        textgrid_path,
        [(0.1, 0.4, "first"), (0.65, 0.9, "second")],
        1.0,
    )
    manifest = tmp_path / "session.jsonl"
    source_row = {
        "session_id": "meeting-001",
        "segment_id": "segment-001",
        "speaker_id": "speaker-00",
        "audio_filepath": str(audio),
        "offset": 1.0,
        "duration": 1.0,
        "mfa_textgrid": str(textgrid_path),
        "text": "first second",
    }
    original_manifest = json.dumps(source_row) + "\n"
    manifest.write_text(original_manifest, encoding="utf-8")

    summary = process_manifest(
        manifest,
        tmp_path / "output",
        FakeDetector(
            {
                audio.name: VADPredictions(
                    silero=((1.15, 1.85),),
                    secondary=((1.18, 1.82),),
                    secondary_name="sortformer",
                    sound_events=((1.45, 1.55),),
                )
            }
        ),
    )

    assert manifest.read_text(encoding="utf-8") == original_manifest
    session_rttm = Path(str(summary["session_rttm"]))
    assert session_rttm.is_file()
    _, intervals = read_rttm(session_rttm)
    assert intervals["speaker-00"][0][0] <= 1.1
    assert intervals["speaker-00"][-1][1] >= 1.9

    output_manifest = Path(str(summary["output_manifest"]))
    refined_row = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert Path(refined_row["vad_refined_textgrid"]).is_file()
    assert refined_row["speech_intervals_source"] == "constrained_equal_weight_atomic_vote"
    speaker_audit = summary["speakers"]["speaker-00"]
    assert speaker_audit["silero_raw_intervals"] == ((1.15, 1.85),)
    assert speaker_audit["sortformer_raw_intervals"] == ((1.18, 1.82),)
    segment_audit = speaker_audit["segments"][0]
    assert segment_audit["atomic_vote"]["algorithm"] == "constrained_equal_weight_atomic_vote"
    assert "word_group_changes" not in segment_audit
    assert summary["refinement_algorithm"]["postprocessing"] == "none"
    assert (tmp_path / "output" / "meeting-001" / "audit.json").is_file()
    assert (tmp_path / "output" / "meeting-001" / "textgrids" / "meeting-001_vad_refined.TextGrid").is_file()


@pytest.mark.parametrize(
    ("output_factor", "high_resolution", "expected"),
    [(2, False, 0.02), (None, False, 0.08), (None, True, 0.01)],
)
def test_sortformer_frame_duration_supports_public_legacy_and_modern_metadata(
    output_factor: int | None,
    high_resolution: bool,
    expected: float,
) -> None:
    model = SimpleNamespace(
        cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01)),
        encoder=SimpleNamespace(subsampling_factor=8),
        high_resolution=high_resolution,
    )
    if output_factor is not None:
        model.output_subsampling_factor = output_factor
    assert _sortformer_frame_duration(model) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("factor", "stride"),
    [(0, 0.01), (True, 0.01), (-1, 0.01), (1.5, 0.01), (8, 0), (8, float("nan")), (8, float("inf"))],
)
def test_sortformer_rejects_unusable_frame_metadata(factor: object, stride: float) -> None:
    model = SimpleNamespace(
        cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=stride)),
        output_subsampling_factor=factor,
    )
    with pytest.raises(ValueError, match=r"subsampling|window_stride"):
        _sortformer_frame_duration(model)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("offset", float("nan")),
        ("offset", float("inf")),
        ("duration", float("nan")),
        ("duration", float("inf")),
        ("offset", False),
        ("duration", True),
    ],
)
def test_manifest_rejects_nonfinite_and_boolean_time_coordinates(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    row = {
        "session_id": "session",
        "speaker_id": "speaker",
        "audio_filepath": "audio.wav",
        "mfa_textgrid": "words.TextGrid",
        "offset": 0.0,
        "duration": 1.0,
    }
    row[field] = value
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match="finite"):
        load_manifest(manifest)


def test_manifest_rejects_overflowing_segment_end(tmp_path: Path) -> None:
    row = {
        "session_id": "session",
        "speaker_id": "speaker",
        "audio_filepath": "audio.wav",
        "mfa_textgrid": "words.TextGrid",
        "offset": 1e308,
        "duration": 1e308,
    }
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="segment end must be finite"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    "audio_names",
    [("channel.wav", "channel.wav"), ("a b.wav", "a_b.wav"), ("session.wav", "other.wav"), ("alice.wav", "bob.wav")],
)
def test_session_and_speaker_rttms_cannot_overwrite_each_other(
    tmp_path: Path,
    audio_names: tuple[str, str],
) -> None:
    rows = []
    predictions = {}
    inputs = []
    for index, name in enumerate(audio_names):
        audio = tmp_path / f"speaker-{index}" / name
        audio.parent.mkdir()
        audio.touch()
        textgrid = tmp_path / f"words-{index}.TextGrid"
        _write_textgrid(textgrid, [(0.2, 0.4, "word")], 1.0)
        inputs.extend((audio, textgrid))
        rows.append(
            {
                "session_id": "session",
                "speaker_id": f"speaker-{index}",
                "audio_filepath": str(audio),
                "mfa_textgrid": str(textgrid),
                "offset": 0.0,
                "duration": 1.0,
                "text": "word",
            }
        )
        predictions[audio.name] = VADPredictions(
            silero=((0.1, 0.8),),
            secondary=((0.1, 0.8),),
            secondary_name="sortformer",
        )
    manifest = tmp_path / "input.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    inputs.append(manifest)
    original = {path: path.read_bytes() for path in inputs}
    summary = process_manifest(manifest, tmp_path / "output", FakeDetector(predictions))
    paths = summary["speaker_rttms"]
    assert len(set(paths.values())) == 2
    assert summary["session_rttm"] not in paths.values()
    assert set(read_rttm(Path(summary["session_rttm"]))[1]) == {"speaker-0", "speaker-1"}
    for speaker, path in paths.items():
        assert set(read_rttm(Path(path))[1]) == {speaker}
    if audio_names == ("alice.wav", "bob.wav"):
        assert Path(paths["speaker-0"]).name == "alice_vad_refined.rttm"
        assert Path(paths["speaker-1"]).name == "bob_vad_refined.rttm"
    assert {path: path.read_bytes() for path in original} == original


def test_overlapping_windows_are_rejected_before_model_loading_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        {
            "session_id": "session",
            "speaker_id": "speaker",
            "audio_filepath": "audio.wav",
            "mfa_textgrid": "words.TextGrid",
            "offset": offset,
            "duration": 1.0,
        }
        for offset in (0.0, 0.5)
    ]
    manifest = tmp_path / "input.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["vad-refine", "--manifest", str(manifest), "--output-root", str(output), "--sortformer-model", "unused.nemo"],
    )
    monkeypatch.setattr(refinement, "_create_detector", lambda _args: pytest.fail("Models must not load"))
    assert refinement.main() == 1
    message = capsys.readouterr().out
    assert "input data error: overlapping segment windows" in message
    assert "speaker 'speaker'" in message
    assert "rows 1 and 2" in message
    assert "previous end=1.0, next start=0.5" in message
    assert not output.exists()


def test_completed_manifest_does_not_load_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "session_id": "session",
        "speaker_id": "speaker",
        "audio_filepath": "audio.wav",
        "mfa_textgrid": "words.TextGrid",
        "offset": 0.0,
        "duration": 1.0,
    }
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    (output / "session").mkdir(parents=True)
    (output / "session" / "audit.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["vad-refine", "--manifest", str(manifest), "--output-root", str(output), "--sortformer-model", "unused.nemo"],
    )
    monkeypatch.setattr(refinement, "_create_detector", lambda _args: pytest.fail("Models must not load"))
    assert refinement.main() == 0


@pytest.mark.skipif(
    not (os.environ.get("CURATOR_VAD_MODEL") and os.environ.get("CURATOR_VAD_AUDIO")),
    reason="Set CURATOR_VAD_MODEL and CURATOR_VAD_AUDIO for the real-model CPU smoke test",
)
def test_real_silero_sortformer_pipeline(tmp_path: Path) -> None:
    """Check real inference and file generation, without measuring alignment accuracy."""
    import soundfile as sf

    from tutorials.audio.vad_refinement.sound_expert_intervals import SoundExpertConfig

    audio = Path(os.environ["CURATOR_VAD_AUDIO"]).resolve()
    model = Path(os.environ["CURATOR_VAD_MODEL"]).resolve()
    assert audio.is_file()
    assert model.is_file()
    info = sf.info(audio)
    duration = info.frames / info.samplerate
    assert duration >= 2.0
    words = [(0.2, 0.4, "anchor-one"), (0.8, 1.0, "anchor-two")]
    grid = tmp_path / "anchors.TextGrid"
    _write_textgrid(grid, words, duration)
    row = {
        "session_id": "real-model-smoke",
        "speaker_id": "speaker",
        "audio_filepath": str(audio),
        "mfa_textgrid": str(grid),
        "offset": 0.0,
        "duration": duration,
    }
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    original = grid.read_bytes()
    detector = refinement.SileroSortformerVADDetector(
        refinement.VADConfig(),
        SortformerVADConfig(model=str(model), device="cpu"),
        SoundExpertConfig(),
    )
    summary = process_manifest(manifest, tmp_path / "output", detector)
    assert summary["detectors"]["sortformer"]["frame_duration_sec"] > 0
    for source in ("silero_intervals", "sortformer_intervals"):
        intervals = summary["speakers"]["speaker"][source]
        assert intervals, f"No speech detected by {source}; use a speech recording"
        for start, end in intervals:
            assert math.isfinite(start)
            assert math.isfinite(end)
            assert 0 <= start < end <= duration
    refined_row = json.loads(Path(summary["output_manifest"]).read_text().strip())
    refined = refinement._marked_words(Path(refined_row["vad_refined_textgrid"]))
    assert [word[2] for word in refined] == [word[2] for word in words]
    for before, after in zip(words, refined, strict=True):
        assert after[0] <= before[0] + 1e-9
        assert after[1] >= before[1] - 1e-9
    assert refined[0][1] <= refined[1][0]
    assert grid.read_bytes() == original
    assert set(read_rttm(Path(summary["session_rttm"]))[1]) == {"speaker"}
    assert TextGrid.fromFile(str(tmp_path / "output/real-model-smoke/textgrids/real-model-smoke_vad_refined.TextGrid"))
    audit = tmp_path / "output/real-model-smoke/audit.json"
    assert json.loads(audit.read_text())["detectors"]["sound_expert"] is None


@pytest.mark.parametrize(
    ("silero", "sortformer"),
    [((), ()), (((7.5, 8.5),), ()), ((), ((7.5, 8.5),))],
)
def test_mfa_words_and_rttm_survive_empty_vad_support(
    tmp_path: Path,
    silero: tuple[tuple[float, float], ...],
    sortformer: tuple[tuple[float, float], ...],
) -> None:
    audio = tmp_path / "audio.wav"
    audio.touch()
    grid = tmp_path / "words.TextGrid"
    words = [(0.1, 0.3, "first"), (0.7, 0.9, "second")]
    _write_textgrid(grid, words, 1.0)
    original = grid.read_bytes()
    manifest = tmp_path / "input.jsonl"
    row = {
        "session_id": "session",
        "speaker_id": "speaker",
        "audio_filepath": str(audio),
        "mfa_textgrid": str(grid),
        "offset": 7.5,
        "duration": 1.0,
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    detector = FakeDetector(
        {audio.name: VADPredictions(silero=silero, secondary=sortformer, secondary_name="sortformer")}
    )
    summary = process_manifest(manifest, tmp_path / "output", detector)
    output_row = json.loads(Path(summary["output_manifest"]).read_text())
    refined = refinement._marked_words(Path(output_row["vad_refined_textgrid"]))
    assert [word[2] for word in refined] == [word[2] for word in words]
    for before, after in zip(words, refined, strict=True):
        assert after[:2] == pytest.approx(before[:2])
    _, intervals = read_rttm(Path(summary["session_rttm"]))
    assert len(intervals["speaker"]) == 2
    for before, after in zip(words, intervals["speaker"], strict=True):
        assert after == pytest.approx((7.5 + before[0], 7.5 + before[1]))
    assert grid.read_bytes() == original


@pytest.mark.parametrize(
    ("words", "consensus", "expected"),
    [
        (
            [[0.1, 0.4, "first"], [0.7, 0.9, "second"]],
            [(0.0, 1.0)],
            [(0.0, 0.55), (0.55, 1.0)],
        ),
        (
            [[0.2, 0.4, "first"], [1.0, 1.3, "second"], [2.4, 2.8, "third"]],
            [(0.0, 3.0)],
            [(0.0, 0.7), (0.7, 1.85), (1.85, 3.0)],
        ),
        (
            [[0.1, 0.4, "first"], [0.4, 0.9, "second"]],
            [(0.0, 1.0)],
            [(0.0, 0.4), (0.4, 1.0)],
        ),
        (
            [[0.1, 0.4, "first"], [0.7, 0.9, "second"]],
            [(0.0, 0.45), (0.65, 1.0)],
            [(0.0, 0.45), (0.65, 1.0)],
        ),
    ],
)
def test_word_projection_splits_only_shared_continuous_components_at_mfa_midpoints(
    words: list[list[float | str]],
    consensus: list[tuple[float, float]],
    expected: list[tuple[float, float]],
) -> None:
    projected, changes, _ = project_consensus_to_words(words, consensus, offset=7.0)
    assert [word[2] for word in projected] == [word[2] for word in words]
    for before, after, bounds in zip(words, projected, expected, strict=True):
        assert after[:2] == pytest.approx(bounds)
        assert after[0] <= before[0]
        assert after[1] >= before[1]
    for left, right in pairwise(projected):
        assert left[1] <= right[0]
    assert changes[0]["refined_start"] == pytest.approx(7.0)
