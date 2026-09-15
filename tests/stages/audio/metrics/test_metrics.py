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
from unittest import mock

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

    def test_default_text_keys_compute_metrics(self, audio_task: Callable[..., AudioTask]) -> None:
        """The documented text/text_ref defaults compute metrics without overrides."""
        stage = ComputeWERStage(language="en")
        stage.setup()
        task = audio_task(
            segments=[
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world",
                    "text_ref": "hello world",
                }
            ]
        )

        result = stage.process(task)

        assert result.data["segments"][0]["metrics"]["wer"]["wer"] == 0.0

    def test_missing_text_keys_warn_once_and_continue(self, audio_task: Callable[..., AudioTask]) -> None:
        """Missing hypothesis/reference fields are summarized without aborting later segments."""
        stage = ComputeWERStage(language="en")
        stage.setup()
        task = audio_task(
            segments=[
                {"start": 0.0, "end": 1.0, "text_ref": "missing hypothesis"},
                {"start": 1.0, "end": 2.0, "text": "missing reference"},
                {"start": 2.0, "end": 3.0, "text": "hello world", "text_ref": "hello world"},
            ]
        )

        with mock.patch("nemo_curator.stages.audio.metrics.wer.logger.warning") as warning:
            result = stage.process(task)

        warning.assert_called_once()
        warning_message = warning.call_args.args[0]
        assert "skipped WER computation for 2 segments" in warning_message
        assert "hypothesis_text_key='text' missing from 1 segment" in warning_message
        assert "reference_text_key='text_ref' missing from 1 segment" in warning_message
        segments = result.data["segments"]
        assert segments[0]["metrics"]["metric_skip_reason"] == "missing_configured_text_key"
        assert segments[1]["metrics"]["metric_skip_reason"] == "missing_configured_text_key"
        assert segments[2]["metrics"]["wer"]["wer"] == 0.0

    def test_get_wer_missing_key_warns(self) -> None:
        """Direct get_wer use is observable when a configured field is absent."""
        stage = ComputeWERStage(language="en")
        segment = {"start": 0.0, "end": 1.0, "text": "missing reference"}

        with mock.patch("nemo_curator.stages.audio.metrics.wer.logger.warning") as warning:
            stage.get_wer(segment)

        warning.assert_called_once()
        warning_message = warning.call_args.args[0]
        assert "reference_text_key='text_ref'" in warning_message
        assert segment["metrics"]["metric_skip_reason"] == "missing_configured_text_key"

    def test_process_batch_handles_top_level_missing_keys(self) -> None:
        """Executor-facing dispatch annotates a top-level entry instead of failing validation."""
        stage = ComputeWERStage(language="en")
        task = AudioTask(task_id="missing-entry", data={"duration": 1.0})

        with mock.patch("nemo_curator.stages.audio.metrics.wer.logger.warning") as warning:
            result = stage.process_batch([task])

        assert result == [task]
        assert task.data["metrics"]["metric_skip_reason"] == "missing_configured_text_key"
        warning.assert_called_once()
        assert "1 entry in task 'missing-entry'" in warning.call_args.args[0]

    def test_missing_key_warning_is_rate_limited_per_signature(self) -> None:
        """Repeated missing-key signatures produce one warning on a worker stage instance."""
        stage = ComputeWERStage(language="en")
        first_task = AudioTask(
            task_id="first-task",
            data={
                "segments": [
                    {"text": "missing reference one"},
                    {"text": "missing reference two"},
                ]
            },
        )
        second_task = AudioTask(
            task_id="second-task",
            data={"segments": [{"text": "missing reference again"}]},
        )

        with mock.patch("nemo_curator.stages.audio.metrics.wer.logger.warning") as warning:
            stage.process_batch([first_task, second_task])

        warning.assert_called_once()
        warning_message = warning.call_args.args[0]
        assert "2 segments in task 'first-task'" in warning_message
        assert "reference_text_key='text_ref' missing from 2 segments" in warning_message
        assert "suppressed on this worker" in warning_message

    def test_distinct_missing_key_signatures_each_warn_once(self) -> None:
        """Hypothesis-only and reference-only omissions retain separate diagnostics."""
        stage = ComputeWERStage(language="en")
        tasks = [
            AudioTask(task_id="missing-reference", data={"segments": [{"text": "hypothesis"}]}),
            AudioTask(task_id="missing-hypothesis", data={"segments": [{"text_ref": "reference"}]}),
            AudioTask(task_id="missing-reference-again", data={"segments": [{"text": "hypothesis"}]}),
        ]

        with mock.patch("nemo_curator.stages.audio.metrics.wer.logger.warning") as warning:
            stage.process_batch(tasks)

        assert warning.call_count == 2
        messages = [call.args[0] for call in warning.call_args_list]
        assert any("reference_text_key='text_ref'" in message for message in messages)
        assert any("hypothesis_text_key='text'" in message for message in messages)

    def test_rerun_valid_then_missing_clears_stale_wer_metrics(self, audio_task: Callable[..., AudioTask]) -> None:
        """A later missing key removes every WER-owned result from an earlier successful run."""
        stage = ComputeWERStage(language="en", compute_pnc_wer=True)
        stage.setup()
        task = audio_task(
            segments=[
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world",
                    "text_ref": "hello world",
                    "metrics": {"bandwidth": 8000},
                }
            ]
        )
        segment = task.data["segments"][0]

        stage.process(task)
        assert "wer_pnc" in segment["metrics"]
        del segment["text_ref"]

        with mock.patch("nemo_curator.stages.audio.metrics.wer.logger.warning"):
            stage.process(task)

        metrics = segment["metrics"]
        stale_keys = {"wer", "cer", "start_cer", "end_cer", "wer_pnc", "cer_pnc", "char_rate", "word_rate"}
        assert stale_keys.isdisjoint(metrics)
        assert metrics["metric_skip_reason"] == "missing_configured_text_key"
        assert metrics["bandwidth"] == 8000

    def test_rerun_missing_then_valid_clears_stale_skip_reason(self, audio_task: Callable[..., AudioTask]) -> None:
        """A successful rerun removes this stage's earlier missing-key annotation."""
        stage = ComputeWERStage(language="en")
        stage.setup()
        task = audio_task(segments=[{"start": 0.0, "end": 2.0, "text": "hello world"}])
        segment = task.data["segments"][0]

        with mock.patch("nemo_curator.stages.audio.metrics.wer.logger.warning"):
            stage.process(task)
        assert segment["metrics"]["metric_skip_reason"] == "missing_configured_text_key"

        segment["text_ref"] = "hello world"
        stage.process(task)

        assert "metric_skip_reason" not in segment["metrics"]
        assert segment["metrics"]["wer"]["wer"] == 0.0

    def test_successful_wer_preserves_unrelated_skip_reason(self, audio_task: Callable[..., AudioTask]) -> None:
        """Computing WER must not erase a generic skip reason owned by another metric stage."""
        stage = ComputeWERStage(language="en")
        task = audio_task(
            segments=[
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world",
                    "text_ref": "hello world",
                    "metrics": {"metric_skip_reason": "bandwidth unavailable"},
                }
            ]
        )
        segment = task.data["segments"][0]

        with mock.patch.object(stage, "normalize_and_clean_text", return_value=("hello world", "hello world")):
            stage.process(task)

        assert segment["metrics"]["metric_skip_reason"] == "bandwidth unavailable"
        assert segment["metrics"]["wer"]["wer"] == 0.0

    def test_failed_recomputation_clears_partial_wer_metrics(self, audio_task: Callable[..., AudioTask]) -> None:
        """A caught computation error cannot leave partial or previous WER results beside its skip reason."""
        stage = ComputeWERStage(language="en")
        task = audio_task(
            segments=[
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world",
                    "text_ref": "hello world",
                    "metrics": {"bandwidth": 8000, "wer": {"wer": 0.5}},
                }
            ]
        )
        segment = task.data["segments"][0]

        with (
            mock.patch.object(stage, "normalize_and_clean_text", return_value=("hello world", "hello world")),
            mock.patch(
                "nemo_curator.stages.audio.metrics.wer.word_error_rate_detail",
                side_effect=ValueError("invalid transcript"),
            ),
            mock.patch("nemo_curator.stages.audio.metrics.wer.logger.warning"),
        ):
            stage.process(task)

        metrics = segment["metrics"]
        stale_keys = {"wer", "cer", "start_cer", "end_cer", "wer_pnc", "cer_pnc", "char_rate", "word_rate"}
        assert stale_keys.isdisjoint(metrics)
        assert metrics["metric_skip_reason"] == "wer_error: invalid transcript"
        assert metrics["bandwidth"] == 8000

    def test_successful_rerun_clears_wer_error_reason(self, audio_task: Callable[..., AudioTask]) -> None:
        """A successful rerun removes an earlier WER-owned computation error."""
        stage = ComputeWERStage(language="en")
        task = audio_task(
            segments=[
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world",
                    "text_ref": "hello world",
                    "metrics": {"bandwidth": 8000},
                }
            ]
        )
        segment = task.data["segments"][0]

        with (
            mock.patch.object(stage, "normalize_and_clean_text", return_value=("hello world", "hello world")),
            mock.patch(
                "nemo_curator.stages.audio.metrics.wer.word_error_rate_detail",
                side_effect=ValueError("invalid transcript"),
            ),
            mock.patch("nemo_curator.stages.audio.metrics.wer.logger.warning"),
        ):
            stage.process(task)

        assert segment["metrics"]["metric_skip_reason"] == "wer_error: invalid transcript"

        with (
            mock.patch.object(stage, "normalize_and_clean_text", return_value=("hello world", "hello world")),
            mock.patch(
                "nemo_curator.stages.audio.metrics.wer.word_error_rate_detail",
                return_value=(0.0, 2, 0.0, 0.0, 0.0),
            ),
        ):
            stage.process(task)

        metrics = segment["metrics"]
        assert "metric_skip_reason" not in metrics
        assert metrics["wer"]["wer"] == 0.0
        assert metrics["bandwidth"] == 8000

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
