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


from typing import Any

from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import BaseASRProcessorStage
from nemo_curator.tasks import AudioTask


class ConcreteASRProcessor(BaseASRProcessorStage):
    """Concrete subclass for testing base behavior."""

    def setup(self, _worker_metadata: Any = None) -> None:  # noqa: ANN401
        pass

    def process(self, task: AudioTask) -> AudioTask:
        return task


class TestBaseASRProcessorStagePrepareSegmentBatch:
    """Tests for BaseASRProcessorStage._prepare_segment_batch_with_metadata."""

    def test_collects_segment_paths_without_cutting_audio(self) -> None:
        """When cut_audio_segments=False, collects resampled_audio_filepath from segments."""
        stage = ConcreteASRProcessor()
        metadata_batch = [
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "resampled_audio_filepath": "/path/1.wav",
                    },
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "resampled_audio_filepath": "/path/2.wav",
                    },
                ],
            },
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.5,
                        "resampled_audio_filepath": "/path/3.wav",
                    },
                ],
            },
        ]
        result = stage._prepare_segment_batch_with_metadata(
            metadata_batch, cut_audio_segments=False, segments_key="segments"
        )
        assert len(result) == 3
        assert result[0]["resampled_audio_filepath"] == "/path/1.wav"
        assert result[0]["metadata_idx"] == 0
        assert result[0]["segment_idx"] == 0
        assert result[1]["resampled_audio_filepath"] == "/path/2.wav"
        assert result[1]["metadata_idx"] == 0
        assert result[1]["segment_idx"] == 1
        assert result[2]["resampled_audio_filepath"] == "/path/3.wav"
        assert result[2]["metadata_idx"] == 1
        assert result[2]["segment_idx"] == 0

    def test_skips_segments_without_resampled_audio_filepath(self) -> None:
        """Segments missing resampled_audio_filepath are not included."""
        stage = ConcreteASRProcessor()
        metadata_batch = [
            {
                "segments": [
                    {"start": 0.0, "end": 1.0},
                    {"start": 1.0, "end": 2.0, "resampled_audio_filepath": "/only.wav"},
                ],
            },
        ]
        result = stage._prepare_segment_batch_with_metadata(metadata_batch, cut_audio_segments=False)
        assert len(result) == 1
        assert result[0]["resampled_audio_filepath"] == "/only.wav"

    def test_empty_segments_returns_empty_list(self) -> None:
        """Metadata with no segments or empty segments returns empty list."""
        stage = ConcreteASRProcessor()
        result = stage._prepare_segment_batch_with_metadata([{"segments": []}], cut_audio_segments=False)
        assert result == []
        result = stage._prepare_segment_batch_with_metadata([{}], cut_audio_segments=False)
        assert result == []
