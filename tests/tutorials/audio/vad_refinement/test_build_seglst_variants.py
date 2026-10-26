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

from tutorials.audio.vad_refinement.build_seglst_variants import build_seglst, build_variants


def _write_textgrid(path: Path, start: float, end: float, word: str = "hello") -> None:
    textgrid = TextGrid(minTime=0.0, maxTime=1.0)
    tier = IntervalTier(name="words", minTime=0.0, maxTime=1.0)
    tier.add(start, end, word)
    textgrid.append(tier)
    textgrid.write(str(path))


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
    assert mfa[0]["start_time"] == "1.1"
    assert mfa[0]["end_time"] == "1.9"
    assert refined[0]["start_time"] == "1.0"
    assert refined[0]["end_time"] == "2.0"
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
            ("0.0", "1.0", "one"),
            ("1.0", "2.0", "two"),
        ]


def test_invalid_rttm_is_not_treated_as_empty_activity(tmp_path: Path) -> None:
    rttm = tmp_path / "invalid.rttm"
    rttm.write_text("broken RTTM record\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed RTTM line"):
        build_seglst([], rttm)


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
