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
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from textgrid import IntervalTier, TextGrid

from tutorials.audio.vad_refinement.rectify_vad_boundaries import (
    SortformerVADConfig,
    VADPredictions,
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
