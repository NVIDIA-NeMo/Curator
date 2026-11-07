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
from pathlib import Path

import pytest
from textgrid import IntervalTier, TextGrid

from tutorials.audio.vad_refinement.build_seglst_variants import (
    build_seglst,
    build_seglst_from_intervals,
    build_variants,
)


@pytest.mark.parametrize(
    ("start", "end", "precision", "expected_start", "expected_end"),
    [
        (0.0, 10.0, 0, "0", "10"),
        (10.0, 100.0, 0, "10", "100"),
        (0.0, 10.0, 3, "0.000", "10.000"),
        (10.0, 100.0, 3, "10.000", "100.000"),
        (1.23456, 10.2003, 3, "1.235", "10.200"),
    ],
)
def test_timestamps_keep_requested_precision(
    start: float, end: float, precision: int, expected_start: str, expected_end: str
) -> None:
    rows = [
        {"session_id": "session", "speaker_id": "speaker", "offset": start, "duration": end - start, "text": "word"}
    ]
    entries = build_seglst_from_intervals(rows, {"speaker": [(start, end)]}, precision=precision)

    assert len(entries) == 1
    assert entries[0]["start_time"] == expected_start
    assert entries[0]["end_time"] == expected_end


def _write_textgrid(path: Path, start: float, end: float, word: str = "hello") -> None:
    textgrid = TextGrid(minTime=0.0, maxTime=1.0)
    tier = IntervalTier(name="words", minTime=0.0, maxTime=1.0)
    tier.add(start, end, word)
    textgrid.append(tier)
    textgrid.write(str(path))


def test_word_without_a_fragment_is_reported_instead_of_discarded(tmp_path: Path) -> None:
    source = tmp_path / "words.TextGrid"
    grid = TextGrid(minTime=0.0, maxTime=3.0)
    tier = IntervalTier(name="words", minTime=0.0, maxTime=3.0)
    for start, end, word in [(0.0, 0.5, "one"), (1.0, 1.5, "missing"), (2.0, 2.5, "three")]:
        tier.add(start, end, word)
    grid.append(tier)
    grid.write(str(source))
    row = {
        "session_id": "session",
        "speaker_id": "speaker",
        "segment_id": "segment",
        "offset": 0.0,
        "duration": 3.0,
        "text": "one missing three",
        "mfa_textgrid": str(source),
    }

    with pytest.raises(ValueError, match=r"aligned word 'missing'.*does not overlap any SegLST fragment"):
        build_seglst_from_intervals([row], {"speaker": [(0.0, 0.5), (2.0, 2.5)]})


def test_builds_mfa_and_refined_seglst_files(tmp_path: Path) -> None:
    source_grid = tmp_path / "source.TextGrid"
    refined_grid = tmp_path / "refined.TextGrid"
    _write_textgrid(source_grid, 0.1, 0.9)
    _write_textgrid(refined_grid, 0.0, 1.0)

    source_manifest = tmp_path / "source.jsonl"
    refined_manifest = tmp_path / "refined.jsonl"
    row = {
        "session_id": "meeting-001",
        "speaker_id": "speaker-00",
        "audio_filepath": str(tmp_path / "speaker.wav"),
        "offset": 1.0,
        "duration": 1.0,
        "mfa_textgrid": str(source_grid),
        "text": "hello",
    }
    source_manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    refined_manifest.write_text(
        json.dumps(
            {
                **row,
                "vad_refined_textgrid": str(refined_grid),
                "rttm_textgrid": str(refined_grid),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    refined_rttm = tmp_path / "refined.rttm"
    refined_rttm.write_text(
        "SPEAKER meeting-001 1 1.000000 1.000000 <NA> <NA> speaker-00 <NA> <NA>\n",
        encoding="utf-8",
    )

    summary = build_variants(
        source_manifest,
        refined_manifest,
        refined_rttm,
        tmp_path / "seglst",
    )

    mfa = json.loads(Path(str(summary["mfa_seglst"])).read_text(encoding="utf-8"))
    refined = json.loads(Path(str(summary["vad_refined_seglst"])).read_text(encoding="utf-8"))
    assert mfa[0]["start_time"] == "1.100"
    assert mfa[0]["end_time"] == "1.900"
    assert refined[0]["start_time"] == "1.000"
    assert refined[0]["end_time"] == "2.000"
    assert refined[0]["speaker"] == "speaker-00"


def test_continuous_activity_does_not_expand_seglst_across_adjacent_rows(tmp_path: Path) -> None:
    rows = []
    refined_rows = []
    for index, word in enumerate(("one", "two")):
        grid = tmp_path / f"word-{index}.TextGrid"
        _write_textgrid(grid, 0.0, 1.0, word)
        row = {
            "session_id": "session",
            "speaker_id": "speaker",
            "audio_filepath": str(tmp_path / "audio.wav"),
            "offset": float(index),
            "duration": 1.0,
            "mfa_textgrid": str(grid),
            "text": word,
        }
        rows.append(row)
        refined_rows.append({**row, "vad_refined_textgrid": str(grid), "rttm_textgrid": str(grid)})
    source = tmp_path / "source.jsonl"
    refined = tmp_path / "refined.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    refined.write_text("".join(json.dumps(row) + "\n" for row in refined_rows), encoding="utf-8")
    rttm = tmp_path / "continuous.rttm"
    rttm.write_text("SPEAKER session 1 0.000000 2.000000 <NA> <NA> speaker <NA> <NA>\n", encoding="utf-8")
    summary = build_variants(source, refined, rttm, tmp_path / "seglst")
    for variant in ("mfa_seglst", "vad_refined_seglst"):
        entries = json.loads(Path(summary[variant]).read_text())
        assert [(row["start_time"], row["end_time"], row["words"]) for row in entries] == [
            ("0.000", "1.000", "one"),
            ("1.000", "2.000", "two"),
        ]


def test_invalid_rttm_is_not_treated_as_empty_activity(tmp_path: Path) -> None:
    rttm = tmp_path / "invalid.rttm"
    rttm.write_text("broken RTTM record\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed RTTM line"):
        build_seglst([], rttm)


def test_original_and_refined_exports_preserve_the_same_short_pause(tmp_path: Path) -> None:
    grid_path = tmp_path / "words.TextGrid"
    grid = TextGrid(minTime=0.0, maxTime=1.0)
    tier = IntervalTier(name="words", minTime=0.0, maxTime=1.0)
    tier.add(0.0, 0.4, "hello")
    tier.add(0.5, 1.0, "world")
    grid.append(tier)
    grid.write(str(grid_path))
    row = {
        "session_id": "session",
        "speaker_id": "speaker",
        "offset": 0.0,
        "duration": 1.0,
        "text": "hello world",
        "mfa_textgrid": str(grid_path),
        "vad_refined_textgrid": str(grid_path),
        "rttm_textgrid": str(grid_path),
    }
    row["audio_filepath"] = str(tmp_path / "audio.wav")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    rttm = tmp_path / "refined.rttm"
    rttm.write_text(
        "SPEAKER session 1 0.0 0.4 <NA> <NA> speaker <NA> <NA>\n"
        "SPEAKER session 1 0.5 0.5 <NA> <NA> speaker <NA> <NA>\n"
    )

    summary = build_variants(manifest, manifest, rttm, tmp_path / "seglst")
    original = json.loads(Path(summary["mfa_seglst"]).read_text())
    refined = json.loads(Path(summary["vad_refined_seglst"]).read_text())

    assert original == refined
    assert [(entry["start_time"], entry["end_time"], entry["words"]) for entry in original] == [
        ("0.000", "0.400", "hello"),
        ("0.500", "1.000", "world"),
    ]


def test_empty_refined_rttm_with_mfa_words_is_an_error(tmp_path: Path) -> None:
    grid = tmp_path / "words.TextGrid"
    _write_textgrid(grid, 0.1, 0.9, "protected")
    manifest = tmp_path / "input.jsonl"
    row = {
        "session_id": "session",
        "speaker_id": "speaker",
        "audio_filepath": str(tmp_path / "audio.wav"),
        "offset": 0.0,
        "duration": 1.0,
        "mfa_textgrid": str(grid),
        "vad_refined_textgrid": str(grid),
        "rttm_textgrid": str(grid),
        "text": "protected",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rttm = tmp_path / "empty.rttm"
    rttm.write_text("", encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="RTTM contains no speech intervals"):
        build_variants(manifest, manifest, rttm, output)
    assert not output.exists()
