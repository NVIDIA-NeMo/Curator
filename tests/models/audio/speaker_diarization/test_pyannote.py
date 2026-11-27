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

"""Unit tests for PyAnnoteDiarizationAdapter (no pyannote model is loaded)."""

from __future__ import annotations

import random
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from nemo_curator.models.audio.speaker_diarization import pyannote as module
from nemo_curator.models.audio.speaker_diarization.base import DiarizationInputError
from nemo_curator.models.audio.speaker_diarization.pyannote import PyAnnoteDiarizationAdapter, has_overlap

if TYPE_CHECKING:
    from pathlib import Path


def _span(start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end)


class _Annotation:
    def __init__(self, turns: list[tuple[float, float, str]], overlaps: list[SimpleNamespace]) -> None:
        self.turns = turns
        self.overlaps = overlaps

    def get_overlap(self) -> SimpleNamespace:
        return SimpleNamespace(segments_list_=self.overlaps)

    def crop(self, _segment: object) -> _Annotation:
        return self

    def itertracks(self, *, yield_label: bool) -> Any:  # noqa: ANN401
        assert yield_label
        for start, end, speaker in self.turns:
            yield _span(start, end), None, speaker


@pytest.mark.parametrize(
    ("turn", "overlaps", "expected"),
    [
        ((0.0, 2.0), [(1.0, 1.5)], True),
        ((1.0, 2.0), [(0.5, 1.5)], True),
        ((1.0, 2.0), [(0.5, 2.5)], True),
        ((3.0, 4.0), [(1.0, 2.0)], False),
        ((0.0, 0.5), [(1.0, 2.0)], False),
        ((0.0, 1.0), [], False),
    ],
)
def test_has_overlap(turn: tuple[float, float], overlaps: list[tuple[float, float]], expected: bool) -> None:
    assert has_overlap(_span(*turn), [_span(*o) for o in overlaps]) is expected


def _adapter(monkeypatch: pytest.MonkeyPatch, annotation: _Annotation, **kwargs: Any) -> PyAnnoteDiarizationAdapter:  # noqa: ANN401
    monkeypatch.setattr("pyannote.audio.pipelines.utils.hook.ProgressHook", nullcontext)
    adapter = PyAnnoteDiarizationAdapter(**kwargs)
    adapter._pipeline = lambda _payload, **_kwargs: annotation
    adapter._rng = random.Random(0)  # noqa: S311
    return adapter


def test_overlapped_turns_are_separated_and_short_turns_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    annotation = _Annotation(
        [(0.0, 0.2, "A"), (0.5, 2.0, "A"), (2.0, 3.0, "B")],
        overlaps=[_span(2.5, 2.8)],
    )
    adapter = _adapter(monkeypatch, annotation, min_length=0.5)

    (result,) = adapter.diarize_batch([{"waveform": np.zeros(40, dtype=np.float32), "sample_rate": 10}])

    assert result.segments == [{"speaker": "A", "start": 0.5, "end": 2.0}]
    assert result.overlap_segments == [{"speaker": "B", "start": 2.0, "end": 3.0}]


def test_long_turns_are_cut_at_vad_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    vad_calls = []

    def fake_vad(_model: object, audio: np.ndarray, fs: int, **kwargs: Any) -> list[dict[str, float]]:  # noqa: ANN401
        vad_calls.append((audio.shape, fs, kwargs))
        return [{"start": 0.0, "end": 1.0}, {"start": 1.5, "end": 2.0}, {"start": 2.5, "end": 9.0}]

    monkeypatch.setattr(module, "whisperx_vad_segments", fake_vad)
    adapter = _adapter(monkeypatch, _Annotation([(1.0, 11.0, "A")], overlaps=[]), min_length=2.0, max_length=5.0)

    (result,) = adapter.diarize_batch([{"waveform": np.zeros(120, dtype=np.float32), "sample_rate": 10}])

    assert vad_calls == [((1, 100), 10, {"merge_max_length": 5.0, "vad_onset": 0.5})]
    assert [(s["start"], s["end"]) for s in result.segments] == [(1.0, 3.0), (3.5, 10.0)]


def test_unreadable_file_is_a_task_local_input_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _adapter(monkeypatch, _Annotation([], overlaps=[]))
    with pytest.raises(DiarizationInputError, match="Could not decode"):
        adapter.diarize_batch([{"audio_filepath": str(tmp_path / "missing.wav")}])


def test_diarize_requires_load_model() -> None:
    with pytest.raises(RuntimeError, match="load_model"):
        PyAnnoteDiarizationAdapter().diarize_batch([])
