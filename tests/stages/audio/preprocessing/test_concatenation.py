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

from pydub import AudioSegment

from nemo_curator.stages.audio.configs.concatenation import SegmentConcatenationConfig
from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage
from nemo_curator.tasks import AudioBatch


def _make_segment(duration_ms: int = 1000, frame_rate: int = 48000) -> AudioSegment:
    """Create a silent AudioSegment of a given duration."""
    return AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate)


class TestSegmentConcatenationConfig:
    """Tests for SegmentConcatenationConfig."""

    def test_defaults(self) -> None:
        cfg = SegmentConcatenationConfig()
        assert cfg.silence_duration_sec == 1.0
        assert cfg.audio_key == "audio"
        assert cfg.batch_size == 10

    def test_from_dict(self) -> None:
        cfg = SegmentConcatenationConfig.from_dict(
            {"silence_duration_sec": 0.5, "batch_size": 20}
        )
        assert cfg.silence_duration_sec == 0.5
        assert cfg.batch_size == 20
        assert cfg.audio_key == "audio"

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = SegmentConcatenationConfig.from_dict({"foo": "bar"})
        assert cfg.silence_duration_sec == 1.0

    def test_to_dict(self) -> None:
        cfg = SegmentConcatenationConfig(silence_duration_sec=2.0)
        d = cfg.to_dict()
        assert d["silence_duration_sec"] == 2.0
        assert d["audio_key"] == "audio"

    def test_roundtrip(self) -> None:
        original = SegmentConcatenationConfig(silence_duration_sec=0.25, batch_size=5)
        restored = SegmentConcatenationConfig.from_dict(original.to_dict())
        assert restored.silence_duration_sec == original.silence_duration_sec
        assert restored.batch_size == original.batch_size


class TestSegmentConcatenationStage:
    """Tests for SegmentConcatenationStage."""

    def test_stage_properties(self) -> None:
        stage = SegmentConcatenationStage()
        assert stage.name == "SegmentConcatenation"
        assert stage.inputs() == (["data"], [])
        _, output_keys = stage.outputs()
        assert "audio" in output_keys
        assert "num_segments" in output_keys
        assert "total_duration_sec" in output_keys
        assert "concatenated" in output_keys

    def test_config_overrides_params(self) -> None:
        cfg = SegmentConcatenationConfig(silence_duration_sec=2.5, audio_key="wav", batch_size=5)
        stage = SegmentConcatenationStage(config=cfg)
        assert stage.silence_duration_sec == 2.5
        assert stage.audio_key == "wav"
        assert stage.batch_size == 5

    def test_process_batch_concatenates_segments(self) -> None:
        seg1 = _make_segment(duration_ms=2000)
        seg2 = _make_segment(duration_ms=3000)

        tasks = [
            AudioBatch(data=[{"audio": seg1}], dataset_name="ds"),
            AudioBatch(data=[{"audio": seg2}], dataset_name="ds"),
        ]

        stage = SegmentConcatenationStage(silence_duration_sec=1.0)
        result = stage.process_batch(tasks)

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, AudioBatch)
        assert out.data[0]["num_segments"] == 2
        assert out.data[0]["concatenated"] is True
        expected_duration = (2000 + 1000 + 3000) / 1000.0
        assert abs(out.data[0]["total_duration_sec"] - expected_duration) < 0.1

    def test_process_batch_empty_input(self) -> None:
        stage = SegmentConcatenationStage()
        result = stage.process_batch([])
        assert result == []

    def test_process_batch_single_segment(self) -> None:
        seg = _make_segment(duration_ms=5000)
        tasks = [AudioBatch(data=[{"audio": seg}], dataset_name="ds")]

        stage = SegmentConcatenationStage(silence_duration_sec=0.5)
        result = stage.process_batch(tasks)

        assert len(result) == 1
        assert result[0].data[0]["num_segments"] == 1
        assert abs(result[0].data[0]["total_duration_sec"] - 5.0) < 0.1

    def test_silence_duration_in_output(self) -> None:
        seg1 = _make_segment(duration_ms=1000)
        seg2 = _make_segment(duration_ms=1000)

        tasks = [
            AudioBatch(data=[{"audio": seg1}], dataset_name="ds"),
            AudioBatch(data=[{"audio": seg2}], dataset_name="ds"),
        ]

        stage = SegmentConcatenationStage(silence_duration_sec=2.0)
        result = stage.process_batch(tasks)

        combined = result[0].data[0]["audio"]
        combined_duration_sec = len(combined) / 1000.0
        expected = 1.0 + 2.0 + 1.0
        assert abs(combined_duration_sec - expected) < 0.1

    def test_process_single_delegates_to_batch(self) -> None:
        seg = _make_segment(duration_ms=1000)
        task = AudioBatch(data=[{"audio": seg}], dataset_name="ds")

        stage = SegmentConcatenationStage()
        result = stage.process(task)

        assert isinstance(result, AudioBatch)
        assert result.data[0]["num_segments"] == 1

    def test_no_audio_segment_in_items(self) -> None:
        tasks = [AudioBatch(data=[{"other_key": "value"}], dataset_name="ds")]

        stage = SegmentConcatenationStage()
        result = stage.process_batch(tasks)

        assert result == []

    def test_preserves_dataset_name(self) -> None:
        seg = _make_segment(duration_ms=1000)
        tasks = [AudioBatch(data=[{"audio": seg}], dataset_name="my_dataset")]

        stage = SegmentConcatenationStage()
        result = stage.process_batch(tasks)

        assert result[0].dataset_name == "my_dataset"
