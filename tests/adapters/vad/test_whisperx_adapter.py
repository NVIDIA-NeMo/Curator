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

"""Unit tests for WhisperXVADAdapter (WhisperX VAD model mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("whisperx")

from nemo_curator.adapters.vad import VADResult, WhisperXVADAdapter


class TestWhisperXVADAdapterConstruction:
    def test_defaults(self) -> None:
        a = WhisperXVADAdapter()
        assert a.model_id == "whisperx/vad"
        assert a.device == "cuda"
        assert a.vad_onset == 0.5
        assert a.vad_offset == 0.363
        assert a.max_length == 40.0
        assert a.min_length == 0.5
        assert a.last_metrics == {}

    def test_conforms_to_protocol(self) -> None:
        from nemo_curator.adapters.vad import VADAdapter

        assert isinstance(WhisperXVADAdapter(), VADAdapter)


class TestWhisperXVADAdapterLifecycle:
    @patch("nemo_curator.adapters.vad.whisperx.WhisperXVADModel")
    def test_prefetch_constructs_cpu_model(self, mock_model_cls: MagicMock) -> None:
        WhisperXVADAdapter.prefetch_weights("whisperx/vad")
        mock_model_cls.assert_called_once_with(device="cpu")

    @patch("nemo_curator.adapters.vad.whisperx.WhisperXVADModel")
    def test_setup_wires_model(self, mock_model_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        a = WhisperXVADAdapter(device="cpu", vad_onset=0.7, vad_offset=0.2)
        a.setup()
        mock_model_cls.assert_called_once_with(device="cpu", vad_onset=0.7, vad_offset=0.2)
        mock_model.to.assert_called_once_with("cpu")

    @patch("nemo_curator.adapters.vad.whisperx.WhisperXVADModel")
    def test_teardown_clears_state(self, _mc: MagicMock) -> None:
        a = WhisperXVADAdapter(device="cpu")
        a.setup()
        a.teardown()
        assert a._vad_model is None


class TestWhisperXVADAdapterDetectBatch:
    def test_empty_batch_returns_empty(self) -> None:
        a = WhisperXVADAdapter()
        assert a.detect_batch([]) == []

    def test_requires_setup(self) -> None:
        a = WhisperXVADAdapter()
        with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
            a.detect_batch([{"audio_filepath": "/tmp/x.wav"}])

    def test_missing_audio_filepath_skipped(self) -> None:
        a = WhisperXVADAdapter()
        a._vad_model = MagicMock()
        results = a.detect_batch([{"audio_filepath": None}])
        assert len(results) == 1
        assert results[0].intervals == []
        assert a.last_metrics["skipped_short_total"] == 1.0

    @patch("nemo_curator.adapters.vad.whisperx.get_audio_duration", return_value=0.2)
    def test_short_clip_skipped(self, _mock_dur: MagicMock) -> None:
        a = WhisperXVADAdapter(min_length=0.5)
        a._vad_model = MagicMock()
        results = a.detect_batch([{"audio_filepath": "/tmp/x.wav"}])
        assert results[0].intervals == []
        # Skipped clips MUST NOT call the VAD model.
        a._vad_model.get_vad_segments.assert_not_called()

    @patch("nemo_curator.adapters.vad.whisperx.sf.read")
    @patch("nemo_curator.adapters.vad.whisperx.get_audio_duration", return_value=5.0)
    def test_happy_path_emits_intervals(
        self, _mock_dur: MagicMock, mock_read: MagicMock
    ) -> None:
        # 5-sec mono audio at 16 kHz
        mock_read.return_value = (np.zeros(16000 * 5, dtype=np.float32), 16000)
        a = WhisperXVADAdapter(device="cpu", min_length=0.5, max_length=40.0)
        a._vad_model = MagicMock()
        a._vad_model.get_vad_segments.return_value = [
            {"start": 0.5, "end": 2.5},
            {"start": 3.0, "end": 4.7},
        ]
        results = a.detect_batch([{"audio_filepath": "/tmp/x.wav"}])
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, VADResult)
        assert [(iv.start, iv.end) for iv in result.intervals] == [(0.5, 2.5), (3.0, 4.7)]
        # mono audio -> expand_dims along axis 0 -> (1, N)
        passed_audio = a._vad_model.get_vad_segments.call_args.args[0]
        assert passed_audio.shape == (1, 80000)
        # last_metrics is populated.
        assert a.last_metrics["batch_size"] == 1.0
        assert a.last_metrics["vad_intervals_detected_total"] == 2.0
