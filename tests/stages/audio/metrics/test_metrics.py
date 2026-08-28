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

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from nemo_curator.stages.audio.metrics.bandwidth import BandwidthEstimationStage
from nemo_curator.stages.audio.metrics.squim import TorchSquimQualityMetricsStage
from nemo_curator.stages.audio.metrics.wer import ComputeWERStage, GetPairwiseWerStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


class TestBandwidthEstimationStage:
    """Tests for BandwidthEstimationStage."""

    def test_process(self, audio_task: Callable[..., AudioTask], audio_filepath: Path) -> None:
        stage = BandwidthEstimationStage()
        stage.setup()
        task = audio_task(
            audio_filepath=str(audio_filepath),
            segments=[{"speaker": "s1", "start": 0.0, "end": 1.0, "text": "hello world"}],
        )
        result = stage.process(task)
        out = result.data
        assert out["audio_filepath"] == str(audio_filepath)
        assert out["segments"][0]["metrics"]["bandwidth"] == 7500

    def test_no_segments_computes_on_entry(self, audio_task: Callable[..., AudioTask], audio_filepath: Path) -> None:
        """Without segments, bandwidth is computed on the full audio entry."""
        stage = BandwidthEstimationStage()
        stage.setup()
        task = audio_task(audio_filepath=str(audio_filepath), duration=10.0)
        result = stage.process(task)
        assert result.data["audio_filepath"] == str(audio_filepath)
        assert "metrics" in result.data
        assert "bandwidth" in result.data["metrics"]
        assert result.data["metrics"]["bandwidth"] > 0

    def test_waveform_residency_matches_file(self, audio_task: Callable[..., AudioTask], tmp_path: Path) -> None:
        """Bandwidth from an in-memory waveform equals the file result (input_residency)."""
        import numpy as np
        import soundfile as sf

        sr = 16000
        audio = (np.random.default_rng(0).standard_normal(sr) * 0.1).astype("float32")
        wav = tmp_path / "a.wav"
        sf.write(str(wav), audio, sr)

        file_stage = BandwidthEstimationStage()
        file_stage.setup()
        file_bw = file_stage.process(audio_task(audio_filepath=str(wav), duration=1.0)).data["metrics"]["bandwidth"]

        wf_stage = BandwidthEstimationStage(input_residency="waveform")
        wf_stage.setup()
        wf_bw = wf_stage.process(
            audio_task(waveform=torch.from_numpy(audio).unsqueeze(0), sample_rate=sr, duration=1.0)
        ).data["metrics"]["bandwidth"]

        assert wf_bw == file_bw

    def test_residency_validate_input(self) -> None:
        wf_stage = BandwidthEstimationStage(input_residency="waveform")
        assert wf_stage.validate_input(
            AudioTask(data={"waveform": torch.randn(1, 16000), "sample_rate": 16000, "duration": 1.0})
        )
        assert not wf_stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav", "duration": 1.0}))
        # default (file) mode unchanged: needs audio_filepath AND (segments OR duration)
        file_stage = BandwidthEstimationStage()
        assert file_stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav", "duration": 1.0}))
        assert not file_stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"}))


class TestSquimResidency:
    """SQUIM input_residency contract/validation (no model needed)."""

    def test_default_is_file_only(self) -> None:
        stage = TorchSquimQualityMetricsStage()
        assert stage.input_residency == "file"
        forms = {a for s in stage.describe().reads_one_of for a in s.accepts}
        assert forms == {"file"}
        assert stage.validate_input(AudioTask(data={"resampled_audio_filepath": "/a.wav"})) is True
        assert stage.validate_input(AudioTask(data={"waveform": torch.randn(1, 16000), "sample_rate": 16000})) is False

    def test_waveform_and_auto_residency(self) -> None:
        wf_stage = TorchSquimQualityMetricsStage(input_residency="waveform")
        assert {a for s in wf_stage.describe().reads_one_of for a in s.accepts} == {"waveform"}
        assert (
            wf_stage.validate_input(AudioTask(data={"waveform": torch.randn(1, 16000), "sample_rate": 16000})) is True
        )
        assert wf_stage.validate_input(AudioTask(data={"resampled_audio_filepath": "/a.wav"})) is False
        auto = TorchSquimQualityMetricsStage(input_residency="auto")
        assert {a for s in auto.describe().reads_one_of for a in s.accepts} == {"file", "waveform"}


class TestSquimZeroLengthSegments:
    """A degenerate segment must be skipped, not crash the stage."""

    def test_a_zero_length_segment_is_skipped_and_named(self, tmp_path: Path) -> None:
        """Extracting ``_resolve_entry_audio`` moved the ``audio_path`` binding out of this
        method but left the warning referring to it, so the first zero-length segment raised
        ``NameError``. Only the error path reached it, so ordinary audio never surfaced it.
        """
        import numpy as np
        import soundfile as sf

        wav = tmp_path / "a.wav"
        sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
        entry = {
            "resampled_audio_filepath": str(wav),
            "segments": [
                {"start": 0.0, "end": 0.0, "text": "degenerate", "speaker": "s0"},
                {"start": 0.0, "end": 0.5, "text": "usable", "speaker": "s0"},
            ],
        }
        collected = TorchSquimQualityMetricsStage()._collect_waveforms_for_entry(0, entry)
        assert [seg_idx for _, seg_idx, _ in collected] == [1]  # the zero-length one is skipped


class TestComputeWERStage:
    """Tests for ComputeWERStage helpers and process."""

    def test_get_char_rate(self) -> None:
        """get_char_rate returns chars per second."""
        stage = ComputeWERStage(language="en")
        assert stage.get_char_rate("hello", 1.0) == 5.0
        assert stage.get_char_rate("hi there", 2.0) == 3.5
        assert stage.get_char_rate("", 1.0) == 0.0
        assert stage.get_char_rate("x", 0.0) == 0.0

    def test_get_word_rate(self) -> None:
        """get_word_rate returns words per second."""
        stage = ComputeWERStage(language="en")
        assert stage.get_word_rate("one two three", 1.0) == 3.0
        assert stage.get_word_rate("one two", 2.0) == 1.0
        assert stage.get_word_rate("", 1.0) == 0.0

    def test_clean_text_retain_pncs(self) -> None:
        """clean_text with retain_pncs keeps punctuation."""
        stage = ComputeWERStage(language="en")
        out = stage.clean_text("  hello , world .  ", retain_pncs=True)
        assert out == "hello, world."

    def test_clean_text_lowercase_when_no_pncs(self) -> None:
        """clean_text with retain_pncs=False lowercases."""
        stage = ComputeWERStage(language="en")
        out = stage.clean_text("Hello World", retain_pncs=False)
        assert out == "hello world"

    def test_strip_spaces_before_punctuations(self) -> None:
        """Spaces before punctuation are stripped."""
        stage = ComputeWERStage(language="en")
        out = stage.strip_spaces_before_punctuations("hello , world .")
        assert " ," not in out

    def test_no_segments_computes_on_entry(self, audio_task: Callable[..., AudioTask]) -> None:
        """Without segments, WER is computed on the top-level entry."""
        stage = ComputeWERStage(language="en", hypothesis_text_key="text", reference_text_key="reference")
        stage.setup()
        task = audio_task(audio_item_id="x", duration=10.0, text="hello world", reference="hello world")
        result = stage.process(task)
        assert result.data["audio_item_id"] == "x"
        assert "metrics" in result.data
        assert "wer" in result.data["metrics"]
        assert "cer" in result.data["metrics"]
        assert "char_rate" in result.data["metrics"]
        assert "word_rate" in result.data["metrics"]
        assert result.data["metrics"]["wer"]["wer"] == 0.0
        assert result.data["metrics"]["cer"]["cer"] == 0.0
        assert result.data["metrics"]["word_rate"] == 0.2

    def test_process_computes_wer_cer_for_segments(self, audio_task: Callable[..., AudioTask]) -> None:
        """Segments with hypothesis and reference get WER/CER metrics."""
        stage = ComputeWERStage(language="en")
        task = audio_task(
            segments=[
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world",
                    "reference": "hello world",
                },
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "the cat",
                    "reference": "the dog",
                },
            ],
        )
        stage = ComputeWERStage(language="en", hypothesis_text_key="text", reference_text_key="reference")
        stage.setup()
        result = stage.process(task)
        out = result.data
        assert len(out["segments"]) == 2
        expected_wer = [0.0, 0.5]
        for idx, seg in enumerate(out["segments"]):
            assert "metrics" in seg
            assert "wer" in seg["metrics"]
            assert "cer" in seg["metrics"]
            assert "char_rate" in seg["metrics"]
            assert "word_rate" in seg["metrics"]
            assert abs(seg["metrics"]["wer"]["wer"] - expected_wer[idx]) < 1e-4


class TestTorchSquimQualityMetricsStage:
    """Tests for TorchSquimQualityMetricsStage on CPU and GPU."""

    def _make_task(self, audio_task: Callable[..., AudioTask], wav_filepath: Path) -> AudioTask:
        """Create a task with multiple segments spanning the audio file."""
        return audio_task(
            resampled_audio_filepath=str(wav_filepath),
            segments=[
                {"speaker": "s1", "start": 0.0, "end": 5.0, "text": "segment one"},
                {"speaker": "s1", "start": 5.0, "end": 15.0, "text": "segment two"},
                {"speaker": "s2", "start": 15.0, "end": 30.0, "text": "segment three"},
                {"speaker": "s2", "start": 30.0, "end": 45.0, "text": "segment four"},
                {"speaker": "s1", "start": 45.0, "end": 60.0, "text": "segment five"},
            ],
        )

    @pytest.mark.gpu
    def test_no_segments_computes_on_entry(self, audio_task: Callable[..., AudioTask], wav_filepath: Path) -> None:
        """Without segments, squim metrics are computed on the full audio entry."""
        stage = TorchSquimQualityMetricsStage(resources=Resources(cpus=1.0, gpus=1.0))
        stage.setup()
        task = audio_task(resampled_audio_filepath=str(wav_filepath), duration=60.0)
        result = stage.process_batch([task])[0]
        assert result.data["resampled_audio_filepath"] == str(wav_filepath)
        assert "metrics" in result.data
        assert "pesq_squim" in result.data["metrics"]
        assert "stoi_squim" in result.data["metrics"]
        assert "sisdr_squim" in result.data["metrics"]
        assert 1.0 <= result.data["metrics"]["pesq_squim"] <= 5.0
        assert 0.0 <= result.data["metrics"]["stoi_squim"] <= 1.0

    @pytest.mark.gpu
    def test_process(self, audio_task: Callable[..., AudioTask], wav_filepath: Path) -> None:
        """TorchSquim produces valid metrics on GPU."""
        stage = TorchSquimQualityMetricsStage(resources=Resources(cpus=1.0, gpus=1.0))
        stage.setup()

        task = self._make_task(audio_task, wav_filepath)

        # Warmup pass to exclude CUDA JIT compilation from timing
        warmup_task = audio_task(
            resampled_audio_filepath=str(wav_filepath),
            segments=[{"speaker": "s1", "start": 0.0, "end": 2.0, "text": "warmup"}],
        )
        stage.process_batch([warmup_task])
        torch.cuda.synchronize()

        result = stage.process_batch([task])[0]

        out = result.data
        for seg in out["segments"]:
            assert "metrics" in seg
            assert "pesq_squim" in seg["metrics"]
            assert "stoi_squim" in seg["metrics"]
            assert "sisdr_squim" in seg["metrics"]
            assert 1.0 <= seg["metrics"]["pesq_squim"] <= 5.0
            assert 0.0 <= seg["metrics"]["stoi_squim"] <= 1.0


class TestGetPairwiseWerStage:
    """Tests for GetPairwiseWerStage."""

    def test_process(self, audio_task: Callable[..., AudioTask]) -> None:
        """Computes WER between text and pred_text."""
        stage = GetPairwiseWerStage()
        task = audio_task(text="a b c", pred_text="a x c")
        result = stage.process(task)
        assert isinstance(result, AudioTask)
        assert result.data["wer_pct"] == pytest.approx(33.33, abs=0.1)

    def test_validate_input_valid(self, audio_task: Callable[..., AudioTask]) -> None:
        """Valid task passes validation."""
        stage = GetPairwiseWerStage()
        assert stage.validate_input(audio_task(text="a b c", pred_text="a x c")) is True

    def test_validate_input_missing_text(self, audio_task: Callable[..., AudioTask]) -> None:
        """Task missing text key fails validation."""
        stage = GetPairwiseWerStage()
        assert stage.validate_input(audio_task(pred_text="a x c")) is False

    def test_validate_input_missing_pred_text(self, audio_task: Callable[..., AudioTask]) -> None:
        """Task missing pred_text key fails validation."""
        stage = GetPairwiseWerStage()
        assert stage.validate_input(audio_task(text="a b c")) is False

    def test_process_batch_raises_on_missing_text(self, audio_task: Callable[..., AudioTask]) -> None:
        """process_batch raises ValueError on missing text."""
        stage = GetPairwiseWerStage()
        with pytest.raises(ValueError, match="failed validation"):
            stage.process_batch([audio_task(pred_text="a x c")])

    def test_process_batch_raises_on_missing_pred_text(self, audio_task: Callable[..., AudioTask]) -> None:
        """process_batch raises ValueError on missing pred_text."""
        stage = GetPairwiseWerStage()
        with pytest.raises(ValueError, match="failed validation"):
            stage.process_batch([audio_task(text="a b c")])


class TestLoopContainment:
    """Tests that per-segment errors don't abort remaining segments."""

    def test_wer_skips_segment_missing_keys(self, audio_task: Callable[..., AudioTask]) -> None:
        """ComputeWERStage skips segments missing text keys without aborting the loop."""
        stage = ComputeWERStage(
            language="en",
            hypothesis_text_key="text",
            reference_text_key="text_2",
        )
        stage.setup()
        task = audio_task(
            segments=[
                {"start": 0.0, "end": 1.0, "text": "hello world", "text_2": "hello world"},
                {"start": 1.0, "end": 2.0, "speaker": "A"},
                {"start": 2.0, "end": 3.0, "text": "foo bar", "text_2": "foo baz"},
            ]
        )
        result = stage.process(task)
        segs = result.data["segments"]
        assert "wer" in segs[0].get("metrics", {})
        assert "metrics" not in segs[1] or "wer" not in segs[1].get("metrics", {})
        assert "wer" in segs[2].get("metrics", {})

    def test_bandwidth_skips_zero_duration_segment(self, audio_task: Callable[..., AudioTask], tmp_path: Path) -> None:
        """BandwidthEstimation tags zero-duration segments without aborting."""
        import numpy as np
        import soundfile as sf

        wav_path = tmp_path / "test.wav"
        rng = np.random.default_rng(42)
        audio_data = rng.standard_normal(16000).astype(np.float32)
        sf.write(str(wav_path), audio_data, 16000)

        stage = BandwidthEstimationStage()
        task = audio_task(
            audio_filepath=str(wav_path),
            segments=[
                {"start": 0.0, "end": 0.5, "speaker": "A", "text": "hi"},
                {"start": 0.5, "end": 0.5, "speaker": "A", "text": "bad"},
                {"start": 0.5, "end": 1.0, "speaker": "A", "text": "ok"},
            ],
        )
        result = stage.process(task)
        segs = result.data["segments"]
        assert "bandwidth" in segs[0].get("metrics", {})
        assert "metric_skip_reason" in segs[1].get("metrics", {})
        assert "bandwidth" in segs[2].get("metrics", {})

    def test_wer_empty_reference_tags_skip_reason(self, audio_task: Callable[..., AudioTask]) -> None:
        """Empty reference text sets metric_skip_reason instead of computing inf WER."""
        stage = ComputeWERStage(
            language="en",
            hypothesis_text_key="text",
            reference_text_key="text_2",
        )
        stage.setup()
        task = audio_task(
            segments=[
                {"start": 0.0, "end": 1.0, "text": "hello", "text_2": ""},
            ]
        )
        result = stage.process(task)
        metrics = result.data["segments"][0]["metrics"]
        assert metrics["wer"] is None
        assert metrics["metric_skip_reason"] == "empty_reference"
