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

"""CPU tests for long-form cut planning (synthetic VAD probabilities, no model)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.segmentation.longform_cut_planner import LongformCutPlannerStage
from nemo_curator.stages.audio.segmentation.longform_cuts import (
    SILENCE,
    LongformCutConfig,
    check_tiling,
    chunks_to_cuts,
    cuts_to_spans,
    plan_longform_cuts,
)
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path

FR = 512 / 16000


def _speech(
    duration: float, pause_every: float = 7.0, pause_len: float = 0.8, quiet: list | None = None
) -> tuple[np.ndarray, list[tuple]]:
    """Synthetic recording: speech with a pause every ``pause_every`` s; ``quiet`` = silent [a, b) spans.

    Returns (frame probabilities, diarization intervals). One speaker per 60 s block.
    """
    n = int(duration / FR)
    t = (np.arange(n) + 0.5) * FR
    prob = np.full(n, 0.95)
    prob[(t % pause_every) < pause_len] = 0.02
    for a, b in quiet or []:
        prob[(t >= a) & (t < b)] = 0.02
    speech = prob >= 0.5
    diar, i = [], 0
    while i < n:
        if speech[i]:
            j = i
            while j < n and speech[j]:
                j += 1
            diar.append((i * FR, j * FR, f"speaker_{int(i * FR // 60) % 2}"))
            i = j
        else:
            i += 1
    return prob, diar


def _cuts(prob: np.ndarray, diar: list[tuple], duration: float, **kw: float) -> list[list]:
    return chunks_to_cuts(plan_longform_cuts(prob, FR, diar, duration, LongformCutConfig(**kw)))


def test_plan_tiles_recording_and_respects_lengths() -> None:
    duration = 1500.0
    prob, diar = _speech(duration)
    cuts = _cuts(prob, diar, duration)
    assert check_tiling(cuts, duration, 290.0) is None
    spans = cuts_to_spans(cuts)
    assert all(is_speech for _, _, is_speech in spans)
    lengths = [e - s for s, e, _ in spans]
    assert all(120.0 <= x <= 290.0 for x in lengths)
    # every inner cut sits in a pause (prob low at the cut frame)
    for _, e, _ in spans[:-1]:
        assert prob[int(e / FR)] < 0.5


def test_short_recording_is_one_chunk() -> None:
    prob, diar = _speech(200.0)
    assert _cuts(prob, diar, 200.0) == [[0.0, 200.0]]


def test_long_silence_becomes_silence_entry_with_margins() -> None:
    duration = 600.0
    prob, diar = _speech(duration, quiet=[(300.0, 330.0)])
    cuts = _cuts(prob, diar, duration)
    silences = [c for c in cuts if len(c) == 3]
    assert len(silences) == 1
    s, e, tag = silences[0]
    assert tag == SILENCE
    # the 0.5 s margin stays with the speech on both sides (pause before 300 s merges in)
    assert 300.0 < s < 301.0
    assert 329.0 < e < 330.0
    assert check_tiling(cuts, duration, 290.0) is None


def test_vad_veto_keeps_region_the_diarizer_missed() -> None:
    """Diarizer silent for 30 s but VAD hears speech (e.g. chanting) -> NOT skipped."""
    duration = 600.0
    prob, diar = _speech(duration)
    diar = [d for d in diar if d[1] <= 300.0 or d[0] >= 330.0]
    cuts = _cuts(prob, diar, duration)
    assert all(len(c) == 2 for c in cuts)


def test_short_fragment_between_silences_is_merged() -> None:
    duration = 400.0
    prob, diar = _speech(duration, quiet=[(100.0, 150.0), (160.0, 220.0)])
    cuts = _cuts(prob, diar, duration)
    speech = [(s, e) for s, e, sp in cuts_to_spans(cuts) if sp]
    assert all(e - s >= 30.0 for s, e in speech)
    assert check_tiling(cuts, duration, 290.0) is None
    unmerged = _cuts(prob, diar, duration, merge_below_sec=0.0)
    assert any(e - s < 30.0 for s, e, sp in cuts_to_spans(unmerged) if sp)


def test_no_pause_still_feasible_via_forced_cuts() -> None:
    duration = 700.0
    n = int(duration / FR)
    prob = np.full(n, 0.9)
    cuts = _cuts(prob, [(0.0, duration, "a")], duration)
    assert check_tiling(cuts, duration, 290.0) is None
    assert len(cuts) >= 3


@pytest.mark.parametrize(
    ("cuts", "error"),
    [
        ([], "empty"),
        ([[1.0, 10.0]], "first chunk"),
        ([[0.0, 5.0], [6.0, 10.0]], "gap"),
        ([[0.0, 300.0]], "longer"),
        ([[0.0, 300.0, SILENCE]], None),
    ],
)
def test_check_tiling(cuts: list, error: str | None) -> None:
    duration = cuts[-1][1] if cuts else 1.0
    result = check_tiling(cuts, duration, 290.0)
    assert (result is None) if error is None else (error in result)


def test_cuts_to_spans_rejects_unknown_tag() -> None:
    with pytest.raises(ValueError, match="invalid cut entry"):
        cuts_to_spans([[0.0, 1.0, "music"]])


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="min_sec"):
        LongformCutConfig(min_sec=300.0).validate()


# ---- stage ----------------------------------------------------------------------------------------


def _write_wav(path: Path, duration: float) -> str:
    sf.write(path, np.zeros(int(duration * 16000), np.float32), 16000)
    return str(path)


def _stage(prob: np.ndarray, **kw: object) -> LongformCutPlannerStage:
    stage = LongformCutPlannerStage(**kw)
    stage._frame_prob_fn = lambda _wav: prob
    return stage


def _task(path: str, diar: list | None, **extra: object) -> AudioTask:
    data = {"audio_filepath": path, "duration": 600.0, **extra}
    if diar is not None:
        data["diar_segments"] = [{"start": a, "end": b, "speaker": s} for a, b, s in diar]
    return AudioTask(task_id="rec", dataset_name="test", data=data)


def test_stage_writes_compact_cuts_and_drops_diarization(tmp_path: Path) -> None:
    prob, diar = _speech(600.0, quiet=[(300.0, 330.0)])
    task = _stage(prob).process(_task(_write_wav(tmp_path / "a.wav", 600.0), diar))
    cuts = task.data["cuts"]
    assert check_tiling(cuts, 600.0, 290.0) is None
    assert any(len(c) == 3 for c in cuts)
    assert "diar_segments" not in task.data
    assert "_skipme" not in task.data
    assert "additional_notes" not in task.data


def test_stage_keeps_diarization_when_asked(tmp_path: Path) -> None:
    prob, diar = _speech(600.0)
    task = _stage(prob, drop_diar_segments=False).process(_task(_write_wav(tmp_path / "a.wav", 600.0), diar))
    assert "diar_segments" in task.data


def test_stage_marks_missing_diarization(tmp_path: Path) -> None:
    prob, _ = _speech(600.0)
    task = _stage(prob).process(_task(_write_wav(tmp_path / "a.wav", 600.0), None))
    assert task.data["cuts"] == []
    assert "missing diarization" in task.data["_skipme"]


def test_stage_disables_skipping_when_diarization_empty_but_vad_hears_speech(tmp_path: Path) -> None:
    prob, _ = _speech(600.0, quiet=[(300.0, 330.0)])
    task = _stage(prob).process(_task(_write_wav(tmp_path / "a.wav", 600.0), []))
    assert all(len(c) == 2 for c in task.data["cuts"])
    assert "silence skipping disabled" in task.data["additional_notes"]["LongformCutPlanner"]


def test_stage_passes_through_already_skipped_rows(tmp_path: Path) -> None:
    prob, diar = _speech(600.0)
    task = _task(_write_wav(tmp_path / "a.wav", 600.0), diar, _skipme="read_error")
    task = _stage(prob).process(task)
    assert task.data["cuts"] == []
    assert task.data["_skipme"] == "read_error"


def test_stage_marks_unreadable_audio(tmp_path: Path) -> None:
    prob, diar = _speech(600.0)
    task = _stage(prob).process(_task(str(tmp_path / "missing.wav"), diar))
    assert "audio read error" in task.data["_skipme"]


def test_stage_notes_duration_mismatch(tmp_path: Path) -> None:
    prob, diar = _speech(600.0)
    task = _task(_write_wav(tmp_path / "a.wav", 600.0), diar)
    task.data["duration"] = 640.0
    task = _stage(prob).process(task)
    assert "manifest 640.00s" in task.data["additional_notes"]["LongformCutPlanner"]


def test_stage_rejects_bad_overrides() -> None:
    with pytest.raises(ValueError, match="unknown planner_overrides"):
        LongformCutPlannerStage(planner_overrides={"nope": 1})
    with pytest.raises(ValueError, match="as stage fields"):
        LongformCutPlannerStage(planner_overrides={"max_sec": 100.0})


def test_stage_is_agent_ready(tmp_path: Path) -> None:
    prob, diar = _speech(600.0)
    path = _write_wav(tmp_path / "a.wav", 600.0)

    def fixture() -> AudioTask:
        return _task(path, diar)

    assert_agent_ready(
        _stage(prob),
        fixture,
        expected_cardinality="1:1",
        available_keys={"audio_filepath", "diar_segments", "duration"},
    )
