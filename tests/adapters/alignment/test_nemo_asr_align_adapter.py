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

"""Unit tests for NeMoASRAlignAdapter (NeMo ASR model mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("nemo.collections.asr")

from nemo_curator.adapters.alignment import (
    AlignmentResult,
    ForcedAlignmentAdapter,
    NeMoASRAlignAdapter,
)


class TestConstruction:
    def test_defaults(self) -> None:
        a = NeMoASRAlignAdapter()
        assert a.model_id == "nvidia/parakeet-tdt_ctc-1.1b"
        assert a.is_fastconformer is True
        assert a.decoder_type == "rnnt"
        assert a.timestamp_type == "word"
        assert a.compute_timestamps is True
        assert a.disable_word_confidence is False
        assert a.last_metrics == {}

    def test_rejects_unknown_decoder(self) -> None:
        with pytest.raises(ValueError, match="decoder_type"):
            NeMoASRAlignAdapter(decoder_type="beam")

    def test_rejects_unknown_timestamp_type(self) -> None:
        with pytest.raises(ValueError, match="timestamp_type"):
            NeMoASRAlignAdapter(timestamp_type="phoneme")

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(NeMoASRAlignAdapter(), ForcedAlignmentAdapter)


class TestPrefetchWeights:
    @patch("nemo_curator.adapters.alignment.nemo_asr_align.nemo_asr.models.ASRModel.from_pretrained")
    def test_calls_from_pretrained(self, mock_from: MagicMock) -> None:
        NeMoASRAlignAdapter.prefetch_weights("nvidia/parakeet-tdt_ctc-1.1b")
        mock_from.assert_called_once_with(
            model_name="nvidia/parakeet-tdt_ctc-1.1b", return_model_file=True
        )

    @patch(
        "nemo_curator.adapters.alignment.nemo_asr_align.nemo_asr.models.ASRModel.from_pretrained",
        side_effect=Exception("download failed"),
    )
    def test_failure_wrapped_as_runtime_error(self, _mock_from: MagicMock) -> None:
        with pytest.raises(RuntimeError, match="failed to download"):
            NeMoASRAlignAdapter.prefetch_weights("bad/model")

    @patch("nemo_curator.adapters.alignment.nemo_asr_align.nemo_asr.models.ASRModel.from_pretrained")
    def test_empty_model_id_noop(self, mock_from: MagicMock) -> None:
        NeMoASRAlignAdapter.prefetch_weights("")
        mock_from.assert_not_called()


class TestAlignBatch:
    def test_empty_batch_returns_empty(self) -> None:
        a = NeMoASRAlignAdapter()
        assert a.align_batch([]) == []

    def test_requires_setup(self) -> None:
        a = NeMoASRAlignAdapter()
        with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
            a.align_batch([{"audio_path": "/p/x.wav"}])

    def test_path_mode_dispatch(self) -> None:
        a = NeMoASRAlignAdapter(compute_timestamps=False)
        a._asr_model = MagicMock()
        a._override_cfg = MagicMock()
        a._asr_model.transcribe.return_value = [
            SimpleNamespace(text="hello world", timestamp={}, word_confidence=None),
            SimpleNamespace(text="bye", timestamp={}, word_confidence=None),
        ]
        results = a.align_batch([{"audio_path": "/p/1.wav"}, {"audio_path": "/p/2.wav"}])
        assert len(results) == 2
        assert all(isinstance(r, AlignmentResult) for r in results)
        assert results[0].text == "hello world"
        assert results[1].text == "bye"
        # Transcribe got the path list, not numpy arrays.
        passed = a._asr_model.transcribe.call_args.args[0]
        assert passed == ["/p/1.wav", "/p/2.wav"]
        assert a.last_metrics["mode_is_segment"] == 0.0
        assert a.last_metrics["batch_size"] == 2.0

    def test_segment_mode_dispatch(self) -> None:
        a = NeMoASRAlignAdapter(compute_timestamps=False)
        a._asr_model = MagicMock()
        a._override_cfg = MagicMock()
        a._asr_model.transcribe.return_value = [
            SimpleNamespace(text="seg-text", timestamp={}, word_confidence=None),
        ]
        seg = np.zeros(16000, dtype=np.float32)
        results = a.align_batch([{"audio_segment": seg, "sample_rate": 16000}])
        assert len(results) == 1
        assert results[0].text == "seg-text"
        # Transcribe got the numpy list.
        passed = a._asr_model.transcribe.call_args.args[0]
        assert len(passed) == 1
        assert passed[0] is seg
        assert a.last_metrics["mode_is_segment"] == 1.0

    def test_transcribe_returns_tuple_unwrapped(self) -> None:
        a = NeMoASRAlignAdapter(compute_timestamps=False)
        a._asr_model = MagicMock()
        a._override_cfg = MagicMock()
        hyp = [SimpleNamespace(text="x", timestamp={}, word_confidence=None)]
        a._asr_model.transcribe.return_value = (hyp, None)
        results = a.align_batch([{"audio_path": "/p/x.wav"}])
        assert results[0].text == "x"

    def test_batch_failure_falls_back_one_by_one_in_path_mode(self) -> None:
        a = NeMoASRAlignAdapter(compute_timestamps=False)
        a._asr_model = MagicMock()
        a._override_cfg = MagicMock()
        call_count = {"n": 0}

        def transcribe(args, override_config: object) -> object:
            del override_config
            call_count["n"] += 1
            # First call (batch) blows up; per-item retries succeed.
            if len(args) > 1:
                raise RuntimeError("boom")
            return [SimpleNamespace(text=f"ok-{args[0]}", timestamp={}, word_confidence=None)]

        a._asr_model.transcribe.side_effect = transcribe
        results = a.align_batch([{"audio_path": "/a.wav"}, {"audio_path": "/b.wav"}])
        assert [r.text for r in results] == ["ok-/a.wav", "ok-/b.wav"]
        # 1 batch call + 2 per-item retries.
        assert call_count["n"] == 3

    def test_batch_failure_in_segment_mode_raises(self) -> None:
        a = NeMoASRAlignAdapter(compute_timestamps=False)
        a._asr_model = MagicMock()
        a._override_cfg = MagicMock()
        a._asr_model.transcribe.side_effect = RuntimeError("boom")
        with pytest.raises(ValueError, match="segment mode"):
            a.align_batch([{"audio_segment": np.zeros(16000), "sample_rate": 16000}])


class TestGetAlignmentsText:
    def _adapter(self, *, decoder_type: str, is_fastconformer: bool) -> NeMoASRAlignAdapter:
        a = NeMoASRAlignAdapter(
            decoder_type=decoder_type,
            is_fastconformer=is_fastconformer,
            timestamp_type="word",
            compute_timestamps=True,
        )
        # Mock cfg.preprocessor.window_stride for time_stride math.
        a._asr_model = MagicMock()
        a._asr_model.cfg.preprocessor.window_stride = 0.01
        a._override_cfg = MagicMock()
        return a

    def test_ctc_path_mode_emits_word_alignment(self) -> None:
        a = self._adapter(decoder_type="ctc", is_fastconformer=True)
        # FastConformer time_stride = 8 * 0.01 = 0.08
        hyp = SimpleNamespace(
            text="ignored",
            timestamp={
                "word": [
                    {"word": "hello", "start_offset": 10, "end_offset": 20},
                    {"word": "world", "start_offset": 25, "end_offset": 40},
                ]
            },
            word_confidence=[0.9, 0.8],
        )
        a._asr_model.transcribe.return_value = [hyp]
        results = a.align_batch([{"audio_path": "/x.wav"}])
        assert len(results) == 1
        result = results[0]
        # 10 * 0.08 = 0.8, 20 * 0.08 = 1.6 (rounded to 3 dp)
        assert result.alignments[0].word == "hello"
        assert result.alignments[0].start == 0.8
        assert result.alignments[0].end == 1.6
        assert result.alignments[0].confidence == 0.9
        assert result.text == "hello world"

    def test_rnnt_path_subtracts_0_08(self) -> None:
        a = self._adapter(decoder_type="rnnt", is_fastconformer=True)
        # RNNT: start/end = max(0, offset * stride - 0.08)
        hyp = SimpleNamespace(
            text="ignored",
            timestamp={"word": [{"word": "hi", "start_offset": 10, "end_offset": 20}]},
            word_confidence=None,
        )
        a._asr_model.transcribe.return_value = [hyp]
        results = a.align_batch([{"audio_path": "/x.wav"}])
        # 10 * 0.08 = 0.8, minus 0.08 = 0.72; 20 * 0.08 = 1.6, minus 0.08 = 1.52
        assert results[0].alignments[0].start == 0.72
        assert results[0].alignments[0].end == 1.52
        assert results[0].alignments[0].confidence is None

    def test_question_mark_glyph_stripped_from_text(self) -> None:
        a = self._adapter(decoder_type="ctc", is_fastconformer=True)
        hyp = SimpleNamespace(
            text="ignored",
            timestamp={
                "word": [
                    {"word": "hello\u2047", "start_offset": 0, "end_offset": 10},
                    {"word": "world", "start_offset": 15, "end_offset": 20},
                ]
            },
            word_confidence=None,
        )
        a._asr_model.transcribe.return_value = [hyp]
        results = a.align_batch([{"audio_path": "/x.wav"}])
        # ⁇ (U+2047) stripped from the JOINED text only; individual words keep the raw form.
        assert results[0].text == "hello world"

    def test_compute_timestamps_false_returns_text_only(self) -> None:
        a = NeMoASRAlignAdapter(compute_timestamps=False)
        a._asr_model = MagicMock()
        a._override_cfg = MagicMock()
        hyp = SimpleNamespace(text="plain", timestamp={}, word_confidence=None)
        a._asr_model.transcribe.return_value = [hyp]
        results = a.align_batch([{"audio_path": "/x.wav"}])
        assert results[0].text == "plain"
        assert results[0].alignments == []
