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

import inspect
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
import torch
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, IOSpec, StageContract
from nemo_curator.stages.audio._agent._agent_registry import build_contract, stage_params, static_contract
from nemo_curator.stages.audio._agent._catalog import find_producers
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready, assert_residency_consumption
from nemo_curator.stages.audio._agent._planning import validate_pipeline

from nemo_curator.stages.audio.common import PreserveByValueStage
from nemo_curator.stages.audio.metrics.bandwidth import BandwidthEstimationStage
from nemo_curator.stages.audio.metrics.squim import TorchSquimQualityMetricsStage
from nemo_curator.stages.audio.metrics.wer import ComputeWERStage, GetPairwiseWerStage
from nemo_curator.stages.base import ProcessingStage
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

    @pytest.mark.parametrize("nested", [False, True], ids=["top_level", "nested"])
    def test_custom_duration_key_is_used_as_end_fallback(self, nested: bool) -> None:
        stage = BandwidthEstimationStage(
            segments_key="clips",
            duration_key="clip_duration",
            metrics_key="scores",
            waveform_key="samples",
            sample_rate_key="rate",
            input_residency="waveform",
        )
        row = {"samples": np.random.default_rng(4).standard_normal(16000).astype(np.float32), "rate": 16000}
        if nested:
            row["clips"] = [{"start": 0.25, "clip_duration": 0.75, "text": "kept"}]
            target = row["clips"][0]
        else:
            row["start"] = 0.25
            row["clip_duration"] = 0.75
            row["scores"] = {"prior": 1}
            target = row

        stage.process(AudioTask(dataset_name="d", data=row))

        assert "bandwidth" in target["scores"]
        if not nested:
            assert target["scores"]["prior"] == 1


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

    def test_custom_keys_resolve_and_collect_without_model(self) -> None:
        stage = TorchSquimQualityMetricsStage(
            audio_filepath_key="path",
            segments_key="clips",
            metrics_key="scores",
            waveform_key="samples",
            sample_rate_key="rate",
            input_residency="waveform",
            target_sr=16000,
        )
        segment = {"start": 0.0, "end": 0.5, "text": "valid", "scores": {"prior": 1}}
        entry = {
            "samples": np.ones(16000, dtype=np.float32),
            "rate": 16000,
            "clips": [segment],
        }

        collected = stage._collect_waveforms_for_entry(0, entry)
        stage.update_metrics(segment, 1.0, 2.0, 3.0)

        assert collected[0][:2] == (0, 0)
        assert collected[0][2].shape == (8000,)
        assert segment["scores"] == {"prior": 1, "pesq_squim": 1.0, "stoi_squim": 2.0, "sisdr_squim": 3.0}


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
        """Segments using the documented default keys get WER/CER metrics."""
        stage = ComputeWERStage(language="en")
        task = audio_task(
            segments=[
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world",
                    "text_ref": "hello world",
                },
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "the cat",
                    "text_ref": "the dog",
                },
            ],
        )
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

    @pytest.mark.parametrize(
        "timing",
        [
            {"begin": 2.0, "finish": 4.0},
            {"begin": 2.0, "extent": 4.0},
        ],
        ids=["remapped_end", "remapped_duration_fallback"],
    )
    def test_remapped_timing_keys_produce_nonzero_correct_rates(self, timing: dict[str, float]) -> None:
        stage = ComputeWERStage(start_key="begin", end_key="finish", duration_key="extent")
        stage._normalizer = _IdentityNormalizer()
        segment = {
            **timing,
            "text": "ab cd",
            "text_ref": "ab cd",
        }

        stage.get_wer(segment)

        assert segment["metrics"]["char_rate"] == 2.0
        assert segment["metrics"]["word_rate"] == 1.0


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

    def test_custom_keys_compute_pairwise_wer(self) -> None:
        stage = GetPairwiseWerStage(text_key="reference", pred_text_key="hypothesis", wer_key="score")
        task = AudioTask(dataset_name="d", data={"reference": "same", "hypothesis": "same"})

        assert stage.process(task).data["score"] == 0.0


class TestLoopContainment:
    """Tests that per-segment errors don't abort remaining segments."""

    def test_wer_skips_segment_missing_keys(self, audio_task: Callable[..., AudioTask]) -> None:
        """ComputeWERStage warns once per missing-key set and continues after missing keys."""
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
                {"start": 2.0, "end": 3.0, "speaker": "B"},
                {"start": 3.0, "end": 4.0, "text": "missing reference"},
                {"start": 4.0, "end": 5.0, "text": "still missing reference"},
                {"start": 5.0, "end": 6.0, "text": "foo bar", "text_2": "foo baz"},
            ]
        )
        with mock.patch("nemo_curator.stages.audio.metrics.wer.logger.warning") as warning:
            result = stage.process(task)

        assert warning.call_count == 2
        warning_messages = [call.args[0] for call in warning.call_args_list]
        assert "hypothesis_text_key='text'" in warning_messages[0]
        assert "reference_text_key='text_2'" in warning_messages[0]
        assert "hypothesis_text_key='text'" not in warning_messages[1]
        assert "reference_text_key='text_2'" in warning_messages[1]
        segs = result.data["segments"]
        assert "wer" in segs[0].get("metrics", {})
        for segment in segs[1:5]:
            assert "metrics" not in segment or "wer" not in segment.get("metrics", {})
        assert "wer" in segs[5].get("metrics", {})

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


def _pcm16(channels: int, sample_rate: int = 16000) -> np.ndarray:
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    left = np.rint(22000 * np.sin(2 * np.pi * 1000 * time)).astype(np.int16)
    if channels == 1:
        return left
    right = np.rint(12000 * np.sin(2 * np.pi * 3000 * time)).astype(np.int16)
    return np.stack([left, right])


def _write_pcm16(path: Path, pcm: np.ndarray, sample_rate: int = 16000) -> None:
    file_layout = pcm if pcm.ndim == 1 else pcm.T
    sf.write(path, file_layout, sample_rate, subtype="PCM_16")


@pytest.mark.parametrize("channels", [1, 2], ids=["mono", "stereo"])
@pytest.mark.parametrize("resident_type", ["numpy", "torch"])
def test_bandwidth_pcm16_resident_matches_file(
    tmp_path: Path,
    channels: int,
    resident_type: str,
) -> None:
    pcm = _pcm16(channels)
    path = tmp_path / f"bandwidth-{channels}.wav"
    _write_pcm16(path, pcm)
    resident = torch.from_numpy(pcm.copy()) if resident_type == "torch" else pcm
    file_stage = BandwidthEstimationStage()
    resident_stage = BandwidthEstimationStage(input_residency="waveform")

    file_audio, file_rate = file_stage._resolve_entry_audio({"audio_filepath": str(path)})
    resident_audio, resident_rate = resident_stage._resolve_entry_audio({"waveform": resident, "sample_rate": 16000})
    file_task = AudioTask(dataset_name="d", data={"audio_filepath": str(path), "duration": 1.0})
    resident_task = AudioTask(
        dataset_name="d",
        data={"waveform": resident, "sample_rate": 16000, "duration": 1.0},
    )

    assert file_rate == resident_rate == 16000
    np.testing.assert_allclose(resident_audio, file_audio, atol=1e-7)
    assert (
        file_stage.process(file_task).data["metrics"]["bandwidth"]
        == resident_stage.process(resident_task).data["metrics"]["bandwidth"]
    )


@pytest.mark.parametrize("channels", [1, 2], ids=["mono", "stereo"])
@pytest.mark.parametrize("resident_type", ["numpy", "torch"])
def test_squim_pcm16_resident_collection_matches_file(
    tmp_path: Path,
    channels: int,
    resident_type: str,
) -> None:
    pcm = _pcm16(channels)
    path = tmp_path / f"squim-{channels}.wav"
    _write_pcm16(path, pcm)
    resident = torch.from_numpy(pcm.copy()) if resident_type == "torch" else pcm
    file_stage = TorchSquimQualityMetricsStage(target_sr=16000)
    resident_stage = TorchSquimQualityMetricsStage(target_sr=16000, input_residency="waveform")

    file_collected = file_stage._collect_waveforms_for_entry(
        0,
        {"resampled_audio_filepath": str(path)},
    )
    resident_collected = resident_stage._collect_waveforms_for_entry(
        0,
        {"waveform": resident, "sample_rate": 16000},
    )

    assert file_collected[0][:2] == resident_collected[0][:2] == (0, -1)
    torch.testing.assert_close(resident_collected[0][2], file_collected[0][2], atol=1e-7, rtol=0)


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros((1, 2, 3), dtype=np.float32),
        torch.zeros((1, 2, 3), dtype=torch.float32),
        np.zeros(16, dtype=np.uint16),
        torch.zeros(16, dtype=torch.int64),
    ],
    ids=["numpy_3d", "torch_3d", "numpy_unsigned", "torch_int64"],
)
@pytest.mark.parametrize("stage_cls", [BandwidthEstimationStage, TorchSquimQualityMetricsStage])
def test_metrics_resident_audio_rejects_unsupported_shape_or_dtype(stage_cls: type, waveform: object) -> None:
    stage = stage_cls(input_residency="waveform")
    with pytest.raises((TypeError, ValueError), match=r"Resident waveform|Unsupported resident waveform"):
        stage._resolve_entry_audio({"waveform": waveform, "sample_rate": 16000})


@pytest.mark.parametrize(
    ("stage", "path_key"),
    [
        (BandwidthEstimationStage(input_residency="auto"), "audio_filepath"),
        (TorchSquimQualityMetricsStage(input_residency="auto"), "resampled_audio_filepath"),
    ],
    ids=["bandwidth", "squim"],
)
@pytest.mark.parametrize(
    "resident_fragment",
    [
        {"waveform": np.zeros(16000, dtype=np.int16)},
        {"sample_rate": 16000},
    ],
    ids=["missing_sample_rate", "missing_waveform"],
)
def test_auto_residency_rejects_partial_pair_instead_of_different_file(
    tmp_path: Path,
    stage: object,
    path_key: str,
    resident_fragment: dict[str, object],
) -> None:
    path = tmp_path / f"{path_key}.wav"
    _write_pcm16(path, _pcm16(1))
    # The resident silence deliberately differs from the tone on disk. Falling
    # back would silently compute metrics from the wrong source.
    data = {path_key: str(path), **resident_fragment}
    task = AudioTask(dataset_name="d", data=data)

    with pytest.raises(ValueError, match="Incomplete resident audio"):
        stage.validate_input(task)
    with pytest.raises(ValueError, match="Incomplete resident audio"):
        stage._resolve_entry_audio(data)
    assert "metrics" not in data


@pytest.mark.parametrize("stage_cls", [BandwidthEstimationStage, TorchSquimQualityMetricsStage])
def test_metrics_stages_reject_input_residency_typo(stage_cls: type) -> None:
    with pytest.raises(ValueError, match="input_residency must be one of"):
        stage_cls(input_residency="wavefrom")


@pytest.mark.parametrize(
    ("stage_cls", "field_name"),
    [
        (BandwidthEstimationStage, "duration_key"),
        (BandwidthEstimationStage, "metrics_key"),
        (BandwidthEstimationStage, "waveform_key"),
        (BandwidthEstimationStage, "sample_rate_key"),
        (TorchSquimQualityMetricsStage, "metrics_key"),
        (TorchSquimQualityMetricsStage, "waveform_key"),
        (TorchSquimQualityMetricsStage, "sample_rate_key"),
        (ComputeWERStage, "metrics_key"),
        (ComputeWERStage, "start_key"),
        (ComputeWERStage, "end_key"),
        (ComputeWERStage, "duration_key"),
    ],
)
def test_metrics_stages_reject_empty_new_keys(stage_cls: type, field_name: str) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        stage_cls(**{field_name: ""})


@pytest.mark.parametrize(
    ("stage_cls", "kwargs"),
    [
        (BandwidthEstimationStage, {"waveform_key": "audio", "sample_rate_key": "audio"}),
        (TorchSquimQualityMetricsStage, {"waveform_key": "audio", "sample_rate_key": "audio"}),
        (ComputeWERStage, {"start_key": "timestamp", "end_key": "timestamp"}),
    ],
)
def test_metrics_stages_reject_colliding_new_keys(stage_cls: type, kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="Newly configurable keys must be distinct"):
        stage_cls(**kwargs)


def test_pairwise_wer_allows_legacy_in_place_output() -> None:
    task = AudioTask(dataset_name="d", data={"text": "same text", "pred": "same text"})

    GetPairwiseWerStage(text_key="text", pred_text_key="pred", wer_key="text").process(task)

    assert task.data["text"] == 0.0


def test_compute_wer_allows_legacy_cross_scope_segments_alias() -> None:
    stage = ComputeWERStage(segments_key="metrics")
    stage._normalizer = _IdentityNormalizer()
    task = AudioTask(
        dataset_name="d",
        data={"metrics": [{"text": "same", "text_ref": "same", "duration": 1.0}]},
    )

    stage.process(task)

    assert task.data["metrics"][0]["metrics"]["wer"]["wer"] == 0.0


@pytest.mark.parametrize("stage_cls", [BandwidthEstimationStage, TorchSquimQualityMetricsStage])
def test_audio_metric_stages_allow_legacy_segments_metrics_alias(
    stage_cls: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / f"{stage_cls.__name__}-segments.wav"
    _write_pcm16(path, _pcm16(1))
    segment = {"start": 0.0, "end": 1.0, "text": "kept"}

    if stage_cls is BandwidthEstimationStage:
        stage = stage_cls(segments_key="metrics")
        monkeypatch.setattr(stage, "_estimate_bandwidth", lambda *_args: 4321)
        task = AudioTask(dataset_name="d", data={"audio_filepath": str(path), "metrics": [segment]})
        stage.process(task)
        assert segment["metrics"]["bandwidth"] == 4321
    else:
        stage = stage_cls(segments_key="metrics", resources=Resources(gpus=0), model=_FakeSquimModel())
        task = AudioTask(dataset_name="d", data={"resampled_audio_filepath": str(path), "metrics": [segment]})
        stage.process_batch([task])
        assert segment["metrics"]["pesq_squim"] == 2.5


@pytest.mark.parametrize("stage_cls", [BandwidthEstimationStage, TorchSquimQualityMetricsStage])
def test_audio_metric_stages_allow_segmented_path_output_alias(
    stage_cls: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / f"{stage_cls.__name__}-path.wav"
    _write_pcm16(path, _pcm16(1))
    segment = {"start": 0.0, "end": 1.0, "text": "kept"}

    if stage_cls is BandwidthEstimationStage:
        stage = stage_cls(audio_filepath_key="metrics")
        monkeypatch.setattr(stage, "_estimate_bandwidth", lambda *_args: 2468)
        task = AudioTask(dataset_name="d", data={"metrics": str(path), "segments": [segment]})
        stage.process(task)
        assert segment["metrics"]["bandwidth"] == 2468
    else:
        stage = stage_cls(
            audio_filepath_key="metrics",
            resources=Resources(gpus=0),
            model=_FakeSquimModel(),
        )
        task = AudioTask(dataset_name="d", data={"metrics": str(path), "segments": [segment]})
        stage.process_batch([task])
        assert segment["metrics"]["pesq_squim"] == 2.5


def test_nonmapping_bandwidth_metrics_fail_before_fft(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = BandwidthEstimationStage()
    monkeypatch.setattr(
        stage,
        "_estimate_bandwidth",
        lambda *_args: pytest.fail("FFT must not run before metrics container validation"),
    )
    with pytest.raises(TypeError, match="MutableMapping"):
        stage.get_bandwidth({"duration": 1.0, "metrics": []}, np.ones(16000, dtype=np.float32), 16000)


def test_nonmapping_squim_metrics_fail_before_model_collection() -> None:
    stage = TorchSquimQualityMetricsStage(input_residency="waveform")
    entry = {
        "waveform": np.ones(16000, dtype=np.float32),
        "sample_rate": 16000,
        "metrics": None,
    }
    with pytest.raises(TypeError, match="MutableMapping"):
        stage._collect_waveforms_for_entry(0, entry)


def test_nonmapping_wer_metrics_fail_before_normalization() -> None:
    stage = ComputeWERStage()
    with pytest.raises(TypeError, match="MutableMapping"):
        stage.get_wer({"text": "hypothesis", "text_ref": "reference", "metrics": "bad"})


class _IdentityNormalizer:
    def normalize(self, text: str, **_kwargs: object) -> str:
        return text


class _FakeSquimModel:
    def __call__(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = waveforms.shape[0]
        return (
            torch.full((batch_size,), 0.8),
            torch.full((batch_size,), 2.5),
            torch.full((batch_size,), 3.0),
        )


def test_bandwidth_allows_legacy_empty_audio_filepath_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "empty-bandwidth-key.wav"
    _write_pcm16(path, _pcm16(1))
    stage = BandwidthEstimationStage(audio_filepath_key="")
    monkeypatch.setattr(stage, "_estimate_bandwidth", lambda *_args: 1234)
    task = AudioTask(dataset_name="d", data={"": str(path), "duration": 1.0})

    stage.process(task)

    assert task.data["metrics"]["bandwidth"] == 1234


def test_squim_allows_legacy_empty_segments_key(tmp_path: Path) -> None:
    path = tmp_path / "empty-squim-key.wav"
    _write_pcm16(path, _pcm16(1))
    stage = TorchSquimQualityMetricsStage(
        segments_key="",
        resources=Resources(gpus=0),
        model=_FakeSquimModel(),
    )
    segment = {"start": 0.0, "end": 1.0, "text": "kept"}
    task = AudioTask(dataset_name="d", data={"resampled_audio_filepath": str(path), "": [segment]})

    stage.process_batch([task])

    assert segment["metrics"]["pesq_squim"] == 2.5


def test_compute_wer_allows_legacy_empty_hypothesis_key() -> None:
    stage = ComputeWERStage(hypothesis_text_key="")
    stage._normalizer = _IdentityNormalizer()
    task = AudioTask(dataset_name="d", data={"": "same", "text_ref": "same", "duration": 1.0})

    stage.process(task)

    assert task.data["metrics"]["wer"]["wer"] == 0.0


def test_pairwise_wer_allows_legacy_empty_output_key() -> None:
    task = AudioTask(dataset_name="d", data={"text": "same", "pred_text": "same"})

    GetPairwiseWerStage(wer_key="").process(task)

    assert task.data[""] == 0.0


def _resident_metric_stage(stage_cls: type, residency: str) -> object:
    if stage_cls is TorchSquimQualityMetricsStage:
        return stage_cls(input_residency=residency, resources=Resources(gpus=0))
    return stage_cls(input_residency=residency)


@pytest.mark.parametrize("stage_cls", [BandwidthEstimationStage, TorchSquimQualityMetricsStage])
@pytest.mark.parametrize("residency", ["waveform", "auto"])
@pytest.mark.parametrize("sample_rate", [0, -1, True, 16000.5, "16000.5"])
def test_resident_sample_rate_rejects_invalid_values(
    stage_cls: type,
    residency: str,
    sample_rate: object,
) -> None:
    stage = _resident_metric_stage(stage_cls, residency)
    data = {"waveform": np.ones(16000, dtype=np.float32), "sample_rate": sample_rate}
    if stage_cls is BandwidthEstimationStage:
        data["duration"] = 1.0

    with pytest.raises(ValueError, match="positive, losslessly integral, non-boolean"):
        stage.validate_input(AudioTask(dataset_name="d", data=data))
    with pytest.raises(ValueError, match="positive, losslessly integral, non-boolean"):
        stage._resolve_entry_audio(data)


@pytest.mark.parametrize("stage_cls", [BandwidthEstimationStage, TorchSquimQualityMetricsStage])
@pytest.mark.parametrize("sample_rate", [16000, np.int64(16000), 16000.0, "16000", torch.tensor(16000)])
def test_resident_sample_rate_preserves_lossless_legacy_coercions(stage_cls: type, sample_rate: object) -> None:
    stage = _resident_metric_stage(stage_cls, "waveform")
    data = {"waveform": np.ones(16000, dtype=np.float32), "sample_rate": sample_rate}

    _audio, resolved_rate = stage._resolve_entry_audio(data)

    assert resolved_rate == 16000


def test_bandwidth_agent_ready_conformance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bandwidth-conformance.wav"
    _write_pcm16(path, _pcm16(1))
    stage = BandwidthEstimationStage()
    monkeypatch.setattr(stage, "_estimate_bandwidth", lambda *_args: 4321)
    task = AudioTask(dataset_name="d", data={"audio_filepath": str(path), "duration": 1.0})

    assert_agent_ready(
        stage,
        fixture_factory=lambda: task,
        available_keys={"audio_filepath", "duration"},
    )

    assert task.data["metrics"]["bandwidth"] == 4321


def test_squim_agent_ready_conformance() -> None:
    stage = TorchSquimQualityMetricsStage(
        input_residency="waveform",
        resources=Resources(gpus=0),
        model=_FakeSquimModel(),
    )
    task = AudioTask(
        dataset_name="d",
        data={"waveform": np.ones(16000, dtype=np.float32), "sample_rate": 16000},
    )

    assert_agent_ready(
        stage,
        fixture_factory=lambda: task,
        available_keys={"waveform", "sample_rate"},
    )

    assert task.data["metrics"] == {"pesq_squim": 2.5, "stoi_squim": 0.8, "sisdr_squim": 3.0}


def test_compute_wer_agent_ready_conformance() -> None:
    stage = ComputeWERStage()
    stage._normalizer = _IdentityNormalizer()
    task = AudioTask(
        dataset_name="d",
        data={"text": "same text", "text_ref": "same text", "duration": 1.0},
    )

    assert_agent_ready(
        stage,
        fixture_factory=lambda: task,
        available_keys={"text", "text_ref", "duration"},
    )

    assert task.data["metrics"]["wer"]["wer"] == 0.0


def test_pairwise_wer_agent_ready_conformance() -> None:
    stage = GetPairwiseWerStage()
    task = AudioTask(dataset_name="d", data={"text": "same text", "pred_text": "same text"})

    assert_agent_ready(
        stage,
        fixture_factory=lambda: task,
        available_keys={"text", "pred_text"},
    )

    assert task.data["wer_pct"] == 0.0


def test_bandwidth_residency_consumption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bandwidth-residency.wav"
    pcm = _pcm16(1)
    _write_pcm16(path, pcm)
    created: list[AudioTask] = []
    monkeypatch.setattr(BandwidthEstimationStage, "_estimate_bandwidth", lambda *_args: 2468)

    def file_fixture() -> AudioTask:
        task = AudioTask(dataset_name="d", data={"audio_filepath": str(path), "duration": 1.0})
        created.append(task)
        return task

    def waveform_fixture() -> AudioTask:
        task = AudioTask(
            dataset_name="d",
            data={"waveform": pcm, "sample_rate": 16000, "duration": 1.0},
        )
        created.append(task)
        return task

    assert_residency_consumption(
        lambda residency: BandwidthEstimationStage(input_residency=residency),
        file_fixture=file_fixture,
        waveform_fixture=waveform_fixture,
    )

    assert [task.data["metrics"]["bandwidth"] for task in created] == [2468, 2468]


def test_squim_residency_consumption(tmp_path: Path) -> None:
    path = tmp_path / "squim-residency.wav"
    pcm = _pcm16(1)
    _write_pcm16(path, pcm)
    created: list[AudioTask] = []

    def stage_factory(residency: str) -> TorchSquimQualityMetricsStage:
        return TorchSquimQualityMetricsStage(
            input_residency=residency,
            resources=Resources(gpus=0),
            model=_FakeSquimModel(),
        )

    def file_fixture() -> AudioTask:
        task = AudioTask(dataset_name="d", data={"resampled_audio_filepath": str(path)})
        created.append(task)
        return task

    def waveform_fixture() -> AudioTask:
        task = AudioTask(dataset_name="d", data={"waveform": pcm, "sample_rate": 16000})
        created.append(task)
        return task

    assert_residency_consumption(
        stage_factory,
        file_fixture=file_fixture,
        waveform_fixture=waveform_fixture,
    )

    assert all(task.data["metrics"]["pesq_squim"] == 2.5 for task in created)


def test_compute_wer_custom_keys_augment_existing_mapping() -> None:
    stage = ComputeWERStage(
        hypothesis_text_key="hypothesis",
        reference_text_key="reference",
        segments_key="clips",
        metrics_key="scores",
    )
    stage._normalizer = _IdentityNormalizer()
    task = AudioTask(
        dataset_name="d",
        data={
            "hypothesis": "same",
            "reference": "same",
            "duration": 1.0,
            "scores": {"prior": 1},
        },
    )

    stage.process(task)

    assert task.data["scores"]["prior"] == 1
    assert task.data["scores"]["wer"]["wer"] == 0.0
    assert stage.describe().reads_one_of[0].segment_data_keys == ["hypothesis", "reference"]


@pytest.mark.parametrize(
    ("stage_cls", "legacy_names", "new_names"),
    [
        (
            BandwidthEstimationStage,
            (
                "n_fft",
                "stride_seconds",
                "top_db",
                "frequency_threshold",
                "audio_filepath_key",
                "segments_key",
                "name",
            ),
            ("duration_key", "metrics_key", "waveform_key", "sample_rate_key", "input_residency"),
        ),
        (
            TorchSquimQualityMetricsStage,
            (
                "audio_filepath_key",
                "target_sr",
                "batch_size",
                "compute_batch_size",
                "segments_key",
                "name",
                "resources",
                "model",
            ),
            ("metrics_key", "waveform_key", "sample_rate_key", "input_residency"),
        ),
        (
            ComputeWERStage,
            (
                "language",
                "hypothesis_text_key",
                "reference_text_key",
                "num_words_threshold",
                "num_words_look_back",
                "compute_pnc_wer",
                "pnc_chars",
                "edge_length",
                "segments_key",
                "name",
                "_normalizer",
            ),
            ("metrics_key", "start_key", "end_key", "duration_key"),
        ),
    ],
)
def test_metrics_stages_preserve_exact_legacy_positional_slots(
    stage_cls: type,
    legacy_names: tuple[str, ...],
    new_names: tuple[str, ...],
) -> None:
    parameters = inspect.signature(stage_cls.__init__).parameters

    assert (
        tuple(
            name
            for name, parameter in parameters.items()
            if name != "self" and parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        == legacy_names
    )
    assert (
        tuple(name for name, parameter in parameters.items() if parameter.kind is inspect.Parameter.KEYWORD_ONLY)
        == new_names
    )


def test_metrics_stages_bind_legacy_positional_values() -> None:
    bandwidth = BandwidthEstimationStage(256, 0.02, 80.0, -40.0, "path", "clips", "legacy-bandwidth")
    resources = Resources(cpus=2.0, gpus=0.0)
    model = object()
    squim = TorchSquimQualityMetricsStage("path", 8000, 8, 4, "clips", "legacy-squim", resources, model)
    normalizer = object()
    wer = ComputeWERStage(
        "de",
        "hypothesis",
        "reference",
        100,
        4,
        True,
        ".,",
        8,
        "clips",
        "legacy-wer",
        normalizer,
    )

    assert bandwidth.name == "legacy-bandwidth"
    assert bandwidth.duration_key == "duration"
    assert squim.name == "legacy-squim"
    assert squim.resources is resources
    assert squim.model is model
    assert squim.metrics_key == "metrics"
    assert wer.name == "legacy-wer"
    assert wer._normalizer is normalizer
    assert wer.metrics_key == "metrics"
    assert (wer.start_key, wer.end_key, wer.duration_key) == ("start", "end", "duration")


def test_squim_static_hints_and_hidden_runtime_model() -> None:
    contract = static_contract(TorchSquimQualityMetricsStage)
    params = {param.name for param in stage_params(TorchSquimQualityMetricsStage)}

    assert contract.gates.requires_gpu is True
    assert contract.gates.requires_internet_first_run is True
    assert contract.gates.per_row_independent is False
    assert "model" not in params
    assert inspect.signature(TorchSquimQualityMetricsStage.__init__).parameters["model"].kind is (
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    )


def test_conditional_metric_outputs_remain_discoverable() -> None:
    assert {
        "BandwidthEstimationStage",
        "ComputeWERStage",
        "TorchSquimQualityMetricsStage",
    } <= set(find_producers("metrics"))
    assert "GetPairwiseWerStage" in find_producers("score")


class _MetricsConsumer(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    name = "MetricsConsumer"

    def __init__(self, *, nested: bool = False, key: str = "metrics") -> None:
        self.nested = nested
        self.metrics_key = key

    def describe(self) -> StageContract:
        if self.nested:
            return StageContract(reads=IOSpec(segment_data_keys=[self.metrics_key]))
        return StageContract(reads=IOSpec(data_keys=[self.metrics_key]))

    def process(self, task: AudioTask) -> AudioTask:
        return task


def test_conditional_metric_outputs_are_not_guaranteed_planner_writes(tmp_path: Path) -> None:
    path = tmp_path / "skip.wav"
    _write_pcm16(path, _pcm16(1))

    bandwidth = BandwidthEstimationStage()
    top_task = AudioTask(
        dataset_name="d",
        data={"audio_filepath": str(path), "duration": 1.0, "text": ""},
    )
    bandwidth.process(top_task)
    assert "metrics" not in top_task.data
    top_report = validate_pipeline(
        [bandwidth, _MetricsConsumer()],
        initial_roles={"audio_filepath", "duration", "text"},
        initial_keys={"audio_filepath", "duration", "text"},
    )
    # Conditional outputs are never GUARANTEED planner writes: the key is absent from
    # ``produced_keys`` and the consumer's read is flagged as ``conditional_read`` rather than
    # silently accepted -- but the chain still composes, so ``ok`` holds.
    assert top_report.ok
    assert "metrics" not in top_report.produced_keys
    assert any(issue.code == "conditional_read" for issue in top_report.issues)

    squim = TorchSquimQualityMetricsStage(input_residency="waveform")
    segment_task = AudioTask(
        dataset_name="d",
        data={
            "waveform": np.ones(16000, dtype=np.float32),
            "sample_rate": 16000,
            "segments": [{"start": 0.0, "end": 1.0, "text": "", "speaker": "s"}],
        },
    )
    assert squim._collect_waveforms_for_entry(0, segment_task.data) == []
    assert "metrics" not in segment_task.data["segments"][0]
    segment_report = validate_pipeline(
        [squim, _MetricsConsumer(nested=True)],
        initial_roles={"waveform", "sample_rate", "segments"},
        initial_keys={"waveform", "sample_rate", "segments"},
    )
    assert segment_report.ok
    assert any(issue.code == "conditional_read" for issue in segment_report.issues)

    compute_wer = ComputeWERStage()
    compute_wer._normalizer = _IdentityNormalizer()
    wer_task = AudioTask(
        dataset_name="d",
        data={
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "same text", "text_ref": "same text"},
            ]
        },
    )
    compute_wer.process(wer_task)
    assert wer_task.data["segments"][0]["metrics"]["wer"]["wer"] == 0.0
    valid_wer_input = validate_pipeline(
        [compute_wer],
        initial_roles={"segments"},
        initial_keys={"segments"},
        initial_segment_roles={"text", "reference_text"},
        initial_segment_keys={"text", "text_ref"},
    )
    assert valid_wer_input.ok
    assert valid_wer_input.keys_ok
    assert {"text", "text_ref"} <= valid_wer_input.produced_keys
    missing_segment_text = validate_pipeline(
        [compute_wer],
        initial_roles={"segments"},
        initial_keys={"segments"},
    )
    assert not missing_segment_text.ok
    wer_report = validate_pipeline(
        [compute_wer, _MetricsConsumer(nested=True)],
        initial_roles={"segments"},
        initial_keys={"segments"},
        initial_segment_roles={"text", "reference_text"},
        initial_segment_keys={"text", "text_ref"},
    )
    assert wer_report.ok
    assert any(issue.code == "conditional_read" for issue in wer_report.issues)

    pairwise = GetPairwiseWerStage()
    pair_task = AudioTask(dataset_name="d", data={"text": "reference", "pred_text": None})
    pairwise.process(pair_task)
    assert "wer_pct" not in pair_task.data
    selector = PreserveByValueStage("wer_pct", 20.0, "le")
    pair_report = validate_pipeline(
        [pairwise, selector],
        initial_roles={"text", "pred_text"},
        initial_keys={"text", "pred_text"},
    )
    assert pair_report.ok
    producer_only = validate_pipeline(
        [pairwise], initial_roles={"text", "pred_text"}, initial_keys={"text", "pred_text"}
    )
    assert "wer_pct" not in producer_only.produced_keys
    assert any(issue.code == "conditional_read" for issue in pair_report.issues)
    # ...and the runtime consequence the warning describes: a row where the producing branch
    # did not run lacks the key, so the selector's own validation rejects it.
    with pytest.raises(ValueError, match="failed validation"):
        selector.process_batch([pair_task])

    for stage in (bandwidth, squim, ComputeWERStage(), pairwise):
        contract = build_contract(stage)
        assert contract.writes == IOSpec()
        assert contract.conditional_writes
