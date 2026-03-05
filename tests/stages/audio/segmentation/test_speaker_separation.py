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

from nemo_curator.stages.audio.configs.speaker import SpeakerSeparationConfig
from nemo_curator.stages.audio.segmentation.speaker_separation import SpeakerSeparationStage
from nemo_curator.tasks import AudioBatch


def _make_audio_segment(duration_ms: int = 5000) -> AudioSegment:
    return AudioSegment.silent(duration=duration_ms, frame_rate=48000)


class TestSpeakerSeparationConfig:
    """Tests for SpeakerSeparationConfig."""

    def test_defaults(self) -> None:
        cfg = SpeakerSeparationConfig()
        assert cfg.exclude_overlaps is True
        assert cfg.min_duration == 0.8
        assert cfg.gap_threshold == 0.1
        assert cfg.buffer_time == 0.5

    def test_from_dict(self) -> None:
        cfg = SpeakerSeparationConfig.from_dict(
            {"exclude_overlaps": False, "min_duration": 1.5}
        )
        assert cfg.exclude_overlaps is False
        assert cfg.min_duration == 1.5
        assert cfg.gap_threshold == 0.1

    def test_from_dict_none(self) -> None:
        cfg = SpeakerSeparationConfig.from_dict(None)
        assert cfg.exclude_overlaps is True

    def test_from_dict_ignores_unknown(self) -> None:
        cfg = SpeakerSeparationConfig.from_dict({"unknown": 42, "min_duration": 2.0})
        assert cfg.min_duration == 2.0

    def test_to_dict(self) -> None:
        cfg = SpeakerSeparationConfig(exclude_overlaps=False, min_duration=1.0)
        d = cfg.to_dict()
        assert d["exclude_overlaps"] is False
        assert d["min_duration"] == 1.0

    def test_roundtrip(self) -> None:
        original = SpeakerSeparationConfig(
            exclude_overlaps=False, min_duration=1.5, buffer_time=0.3
        )
        restored = SpeakerSeparationConfig.from_dict(original.to_dict())
        assert restored.exclude_overlaps == original.exclude_overlaps
        assert restored.min_duration == original.min_duration
        assert restored.buffer_time == original.buffer_time

    def test_get(self) -> None:
        cfg = SpeakerSeparationConfig()
        assert cfg.get("min_duration") == 0.8
        assert cfg.get("nonexistent", "default") == "default"


class TestSpeakerSeparationStage:
    """Tests for SpeakerSeparationStage."""

    def test_stage_properties(self) -> None:
        stage = SpeakerSeparationStage()
        assert stage.name == "SpeakerSeparation"
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == ([], ["speaker_id", "num_speakers", "duration_sec"])

    def test_config_overrides_params(self) -> None:
        cfg = SpeakerSeparationConfig(exclude_overlaps=False, min_duration=2.0)
        stage = SpeakerSeparationStage(config=cfg)
        assert stage.exclude_overlaps is False
        assert stage.min_duration == 2.0

    def test_ray_stage_spec_is_fanout(self) -> None:
        stage = SpeakerSeparationStage()
        spec = stage.ray_stage_spec()
        from nemo_curator.backends.experimental.utils import RayStageSpecKeys
        assert spec[RayStageSpecKeys.IS_FANOUT_STAGE] is True

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

        audio = _make_audio_segment(10000)
        batch = AudioBatch(
            data=[{"audio": audio}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

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

        batch = AudioBatch(
            data=[{"audio": _make_audio_segment()}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert len(result) == 1
        item = result[0].data[0]
        assert item["speaker_id"] == "spk_0"
        assert item["num_speakers"] == 1
        assert item["duration_sec"] == 5.0
        assert "audio" in item

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_min_duration_filters_short_speakers(self, mock_init) -> None:
        stage = SpeakerSeparationStage(min_duration=2.0)

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {
            "speaker_0": (_make_audio_segment(5000), 5.0),
            "speaker_1": (_make_audio_segment(1000), 1.0),
        }
        stage._separator = separator

        batch = AudioBatch(
            data=[{"audio": _make_audio_segment()}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert len(result) == 1
        assert result[0].data[0]["speaker_id"] == "speaker_0"

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_no_speakers_returns_empty(self, mock_init) -> None:
        stage = SpeakerSeparationStage()

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {}
        stage._separator = separator

        batch = AudioBatch(
            data=[{"audio": _make_audio_segment()}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

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

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_preserves_metadata_keys(self, mock_init) -> None:
        stage = SpeakerSeparationStage(min_duration=0.5)

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {
            "spk_0": (_make_audio_segment(3000), 3.0),
        }
        stage._separator = separator

        batch = AudioBatch(
            data=[{"audio": _make_audio_segment(), "text": "hello", "dataset_name": "ds1"}],
            task_id="my-task",
            dataset_name="my-ds",
        )
        result = stage.process(batch)

        assert len(result) == 1
        assert result[0].dataset_name == "my-ds"
        item = result[0].data[0]
        assert item["text"] == "hello"

    def test_separator_not_available(self) -> None:
        stage = SpeakerSeparationStage()
        stage._separator = None

        with patch.object(stage, "_initialize_separator"):
            batch = AudioBatch(
                data=[{"audio": _make_audio_segment()}],
                task_id="test",
                dataset_name="test",
            )
            result = stage.process(batch)

        assert isinstance(result, list)
        assert len(result) == 0
