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

from textgrid import IntervalTier, TextGrid

from tutorials.audio.vad_refinement.build_seglst_variants import build_variants


def _write_textgrid(path: Path, start: float, end: float) -> None:
    textgrid = TextGrid(minTime=0.0, maxTime=1.0)
    tier = IntervalTier(name="words", minTime=0.0, maxTime=1.0)
    tier.add(start, end, "hello")
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
