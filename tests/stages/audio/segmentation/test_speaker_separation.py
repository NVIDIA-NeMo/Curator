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

from unittest.mock import MagicMock, patch

import torch
from pydub import AudioSegment

from nemo_curator.stages.audio.segmentation.speaker_separation import SpeakerSeparationStage
from nemo_curator.tasks import AudioBatch


def _make_audio_segment(duration_ms: int = 5000, sample_rate: int = 48000) -> AudioSegment:
    return AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)


def _make_item(duration_sec: float = 10.0, sample_rate: int = 48000) -> dict:
    num_samples = int(duration_sec * sample_rate)
    return {"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate}


def _make_batch(duration_sec: float = 10.0, sample_rate: int = 48000) -> AudioBatch:
    return AudioBatch(
        data=[_make_item(duration_sec, sample_rate)],
        task_id="test",
        dataset_name="test",
    )


class TestSpeakerSeparationStage:
    """Tests for SpeakerSeparationStage."""

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_process_returns_per_speaker_batches(self, mock_init) -> None:
        stage = SpeakerSeparationStage(min_duration=0.5)

        separator = MagicMock()
        speaker_data = {
            "speaker_0": (_make_audio_segment(3000), 3.0),
            "speaker_1": (_make_audio_segment(4000), 4.0),
        }
        separator.get_speaker_audio_data.return_value = speaker_data
        stage._separator = separator

        result = stage.process(_make_batch())

        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, AudioBatch)
            item = r.data[0]
            assert "speaker_id" in item
            assert "num_speakers" in item
            assert item["num_speakers"] == 2
            assert "duration_sec" in item

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_process_output_keys(self, mock_init) -> None:
        stage = SpeakerSeparationStage(min_duration=0.5)

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {
            "spk_0": (_make_audio_segment(5000), 5.0),
        }
        stage._separator = separator

        result = stage.process(_make_batch())

        assert len(result) == 1
        item = result[0].data[0]
        assert item["speaker_id"] == "spk_0"
        assert item["num_speakers"] == 1
        assert item["duration_sec"] == 5.0
        assert "waveform" in item
        assert "sample_rate" in item

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_min_duration_filters_short_speakers(self, mock_init) -> None:
        stage = SpeakerSeparationStage(min_duration=2.0)

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {
            "speaker_0": (_make_audio_segment(5000), 5.0),
            "speaker_1": (_make_audio_segment(1000), 1.0),
        }
        stage._separator = separator

        result = stage.process(_make_batch())

        assert len(result) == 1
        assert result[0].data[0]["speaker_id"] == "speaker_0"

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_no_speakers_returns_empty(self, mock_init) -> None:
        stage = SpeakerSeparationStage()

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {}
        stage._separator = separator

        result = stage.process(_make_batch())

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_no_audio_no_filepath_skipped(self, mock_init) -> None:
        stage = SpeakerSeparationStage()
        stage._separator = MagicMock()

        batch = AudioBatch(
            data=[{"some_key": "value"}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_separator_not_available(self) -> None:
        stage = SpeakerSeparationStage()
        stage._separator = None

        with patch.object(stage, "_initialize_separator"):
            result = stage.process(_make_batch())

        assert isinstance(result, list)
        assert len(result) == 0

    def test_pickling(self) -> None:
        """SpeakerSeparationStage should be picklable (for Ray workers)."""
        import pickle

        stage = SpeakerSeparationStage(min_duration=1.0, exclude_overlaps=False)
        pickled = pickle.dumps(stage)
        restored = pickle.loads(pickled)
        assert restored.min_duration == 1.0
        assert restored.exclude_overlaps is False
        assert restored._separator is None

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_separator_exception_skips_item(self, mock_init) -> None:
        """If separator raises on an item, that item is skipped (no crash)."""
        stage = SpeakerSeparationStage(min_duration=0.5)

        separator = MagicMock()
        separator.get_speaker_audio_data.side_effect = RuntimeError("Simulated crash")
        stage._separator = separator

        result = stage.process(_make_batch())

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_mixed_batch_error_skips_bad_item(self, mock_init) -> None:
        """Batch with 2 items: first raises, second succeeds. Only second produces output."""
        stage = SpeakerSeparationStage(min_duration=0.5)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Corrupted audio")
            return {"speaker_0": (_make_audio_segment(3000), 3.0)}

        separator = MagicMock()
        separator.get_speaker_audio_data.side_effect = side_effect
        stage._separator = separator

        batch = AudioBatch(
            data=[_make_item(), _make_item()],
            task_id="test-mixed",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert isinstance(result, list)
        assert len(result) == 1, f"Expected 1 result (bad item skipped), got {len(result)}"
        assert result[0].data[0]["speaker_id"] == "speaker_0"

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_default_model_path_is_hf(self, mock_init) -> None:
        """Default model_path should be the HuggingFace model ID."""
        stage = SpeakerSeparationStage()
        assert stage.model_path == "nvidia/diar_sortformer_4spk-v1"
