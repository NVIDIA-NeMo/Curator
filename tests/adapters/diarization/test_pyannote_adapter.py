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

"""Unit tests for PyAnnoteDiarizationAdapter.

Heavy PyAnnote internals are mocked. End-to-end inference is covered by
the e2e GPU test (``tests/stages/audio/tagging/e2e/test_tts_e2e.py``).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pyannote.audio")

from nemo_curator.adapters.diarization.pyannote import (
    PyAnnoteDiarizationAdapter,
    has_overlap,
)


# ---------------------------------------------------------------------------
# has_overlap helper (moved verbatim from the deleted pyannote.py stage)
# ---------------------------------------------------------------------------


class TestHasOverlap:
    def test_turn_overlaps_with_segment(self) -> None:
        turn = SimpleNamespace(start=0.0, end=2.0)
        overlaps = [SimpleNamespace(start=1.0, end=1.5)]
        assert has_overlap(turn, overlaps) is True

    def test_turn_after_overlap_returns_false(self) -> None:
        turn = SimpleNamespace(start=3.0, end=4.0)
        overlaps = [SimpleNamespace(start=1.0, end=2.0)]
        assert has_overlap(turn, overlaps) is False

    def test_turn_before_overlap_returns_false(self) -> None:
        turn = SimpleNamespace(start=0.0, end=0.5)
        overlaps = [SimpleNamespace(start=1.0, end=2.0)]
        assert has_overlap(turn, overlaps) is False

    def test_empty_overlaps_returns_false(self) -> None:
        turn = SimpleNamespace(start=0.0, end=1.0)
        assert has_overlap(turn, []) is False

    def test_overlap_fully_contains_turn(self) -> None:
        turn = SimpleNamespace(start=1.0, end=2.0)
        overlaps = [SimpleNamespace(start=0.5, end=2.5)]
        assert has_overlap(turn, overlaps) is True


# ---------------------------------------------------------------------------
# Adapter construction and protocol conformance
# ---------------------------------------------------------------------------


class TestPyAnnoteDiarizationAdapterConstruction:
    def test_defaults(self) -> None:
        a = PyAnnoteDiarizationAdapter()
        assert a.model_id == "pyannote/speaker-diarization-3.1"
        assert a.revision is None
        assert a.hf_token == ""
        assert a.device == "cuda"
        assert a.min_length == 0.5
        assert a.max_length == 40.0
        assert a.write_rttm is True
        assert a.random_seed is None
        assert a.last_metrics == {}

    def test_conforms_to_protocol(self) -> None:
        from nemo_curator.adapters.diarization.base import DiarizationAdapter

        a = PyAnnoteDiarizationAdapter()
        assert isinstance(a, DiarizationAdapter)


# ---------------------------------------------------------------------------
# prefetch_weights
# ---------------------------------------------------------------------------


class TestPrefetchWeights:
    def test_prefetch_requires_hf_token(self) -> None:
        with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError, match="HF_TOKEN"):
            PyAnnoteDiarizationAdapter.prefetch_weights("pyannote/speaker-diarization-3.1")

    def test_prefetch_calls_from_pretrained(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": "abc"}, clear=True), patch(
            "nemo_curator.adapters.diarization.pyannote.PyAnnotePipeline.from_pretrained"
        ) as mock_from:
            PyAnnoteDiarizationAdapter.prefetch_weights("pyannote/speaker-diarization-3.1")
            mock_from.assert_called_once_with("pyannote/speaker-diarization-3.1", token="abc")


# ---------------------------------------------------------------------------
# setup / teardown
# ---------------------------------------------------------------------------


class TestSetupTeardown:
    @patch("nemo_curator.adapters.diarization.pyannote.WhisperXVADModel")
    @patch("nemo_curator.adapters.diarization.pyannote.PyAnnotePipeline")
    def test_setup_wires_pipeline_and_vad(self, mock_pipeline_cls: MagicMock, mock_vad_cls: MagicMock) -> None:
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe

        a = PyAnnoteDiarizationAdapter(
            hf_token="tok",
            device="cpu",
            segmentation_batch_size=64,
            embedding_batch_size=32,
            random_seed=42,
        )
        a.setup()

        mock_pipeline_cls.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1", token="tok"
        )
        assert mock_pipe.segmentation_batch_size == 64
        assert mock_pipe.embedding_batch_size == 32
        mock_pipe.to.assert_called_once()
        mock_vad_cls.assert_called_once()

    @patch("nemo_curator.adapters.diarization.pyannote.WhisperXVADModel")
    @patch("nemo_curator.adapters.diarization.pyannote.PyAnnotePipeline")
    def test_teardown_clears_state(self, _mp: MagicMock, _mv: MagicMock) -> None:
        a = PyAnnoteDiarizationAdapter(hf_token="tok", device="cpu")
        a.setup()
        a.teardown()
        assert a._pipeline is None
        assert a._vad_model is None
        assert a._rng is None


# ---------------------------------------------------------------------------
# diarize_batch
# ---------------------------------------------------------------------------


class TestDiarizeBatch:
    def test_empty_batch_returns_empty_list(self) -> None:
        a = PyAnnoteDiarizationAdapter()
        assert a.diarize_batch([]) == []

    def test_diarize_batch_requires_setup(self) -> None:
        a = PyAnnoteDiarizationAdapter()
        with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
            a.diarize_batch([{"audio_filepath": "/tmp/x.wav"}])

    def test_missing_audio_filepath_yields_empty_result(self) -> None:
        a = PyAnnoteDiarizationAdapter()
        a._pipeline = MagicMock()  # bypass setup() guard
        a._vad_model = MagicMock()
        results = a.diarize_batch([{"audio_filepath": None}])
        assert len(results) == 1
        assert results[0].diar_segments == []
        assert results[0].overlap_segments == []
        assert a.last_metrics["batch_size"] == 1.0


# ---------------------------------------------------------------------------
# _add_vad_segments micro-split
# ---------------------------------------------------------------------------


class TestAddVadSegments:
    def _make_adapter(self) -> PyAnnoteDiarizationAdapter:
        a = PyAnnoteDiarizationAdapter(min_length=0.5, max_length=10.0, random_seed=0)
        a._vad_model = MagicMock()
        a._rng = __import__("random").Random(0)
        return a

    def test_short_turn_emits_single_segment(self) -> None:
        a = self._make_adapter()
        out: list = []
        # 5-sec turn (<= max_length=10) -> no VAD micro-split.
        import torch

        audio = torch.zeros(1, 16000 * 12, dtype=torch.float32)
        a._add_vad_segments(audio=audio, fs=16000, start=0.0, end=5.0, segments=out, speaker_id="A")
        assert len(out) == 1
        assert out[0].speaker == "A"
        assert out[0].start == 0.0
        assert out[0].end == 5.0

    def test_long_turn_triggers_vad_micro_split(self) -> None:
        a = self._make_adapter()
        out: list = []
        import torch

        audio = torch.zeros(1, 16000 * 30, dtype=torch.float32)
        # Adapter VAD returns two windows that each exceed any rand-sample length;
        # _add_vad_segments must emit one DiarSegment per window.
        a._vad_model.get_vad_segments.return_value = [
            {"start": 0.0, "end": 6.0},
            {"start": 6.5, "end": 12.0},
        ]
        a._add_vad_segments(audio=audio, fs=16000, start=10.0, end=25.0, segments=out, speaker_id="B")
        assert len(out) == 2
        # Offsets land in absolute clip-coordinate space (start + sub-window start).
        assert out[0].start == 10.0
        assert out[0].end == 16.0
        assert out[1].start == 16.5
        assert out[1].end == 22.0
        assert all(seg.speaker == "B" for seg in out)
