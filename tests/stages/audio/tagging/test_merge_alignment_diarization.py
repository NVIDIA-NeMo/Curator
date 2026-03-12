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


from nemo_curator.stages.audio.tagging.merge_alignment_diarization import (
    MergeAlignmentDiarizationStage,
)


class TestMergeAlignmentDiarizationAlignWordsToSegments:
    """Tests for MergeAlignmentDiarizationStage.align_words_to_segments static method."""

    def test_aligns_words_to_segments_exact_fit(self) -> None:
        """Words fully inside segment get text and words assigned."""
        alignment = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        segments = [
            {"speaker": "s1", "start": 0.0, "end": 1.0},
        ]
        MergeAlignmentDiarizationStage.align_words_to_segments(
            alignment, segments, "text", "words"
        )
        assert segments[0]["text"] == "hello world"
        assert segments[0]["words"] == alignment

    def test_aligns_words_to_two_segments(self) -> None:
        """Words are split across segments by timestamp."""
        alignment = [
            {"word": "one", "start": 0.0, "end": 0.3},
            {"word": "two", "start": 0.3, "end": 0.6},
            {"word": "three", "start": 1.0, "end": 1.5},
        ]
        segments = [
            {"speaker": "s1", "start": 0.0, "end": 0.8},
            {"speaker": "s2", "start": 0.8, "end": 2.0},
        ]
        MergeAlignmentDiarizationStage.align_words_to_segments(
            alignment, segments, "text", "words"
        )
        assert segments[0]["text"] == "one two"
        assert len(segments[0]["words"]) == 2
        assert segments[1]["text"] == "three"
        assert len(segments[1]["words"]) == 1

    def test_empty_alignment_leaves_segments_unchanged(self) -> None:
        """Empty alignment does not add text/words keys or leaves them empty."""
        segments = [{"speaker": "s1", "start": 0.0, "end": 1.0}]
        MergeAlignmentDiarizationStage.align_words_to_segments(
            [], segments, "text", "words"
        )
        assert "text" not in segments[0]
        assert "words" not in segments[0]

    def test_empty_segments_does_nothing(self) -> None:
        """Empty segments list is a no-op."""
        alignment = [{"word": "x", "start": 0.0, "end": 0.5}]
        segments = []
        MergeAlignmentDiarizationStage.align_words_to_segments(
            alignment, segments, "text", "words"
        )
        assert segments == []

    def test_custom_text_and_words_keys(self) -> None:
        """Custom text_key and words_key are used."""
        alignment = [{"word": "hi", "start": 0.0, "end": 0.2}]
        segments = [{"speaker": "s1", "start": 0.0, "end": 1.0}]
        MergeAlignmentDiarizationStage.align_words_to_segments(
            alignment, segments, "transcript", "word_list"
        )
        assert segments[0]["transcript"] == "hi"
        assert segments[0]["word_list"] == alignment


class TestMergeAlignmentDiarizationStage:
    """Tests for MergeAlignmentDiarizationStage process_dataset_entry."""

    def test_process_dataset_entry_merges_alignment_into_segments(
        self, audio_batch
    ) -> None:
        """process_dataset_entry adds text and words to segments from alignment."""
        stage = MergeAlignmentDiarizationStage(text_key="text", words_key="words")
        batch = audio_batch(
            alignment=[
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ],
            segments=[{"speaker": "s1", "start": 0.0, "end": 1.0}],
        )
        batches = stage.process_dataset_entry(batch.data[0])
        assert len(batches) == 1
        out = batches[0].data[0]
        assert out["segments"][0]["text"] == "hello world"
        assert len(out["segments"][0]["words"]) == 2

    def test_process_dataset_entry_no_alignment_passthrough(self, audio_batch) -> None:
        """Entry without alignment is returned unchanged."""
        stage = MergeAlignmentDiarizationStage()
        batch = audio_batch(segments=[{"speaker": "s1", "start": 0.0, "end": 1.0}])
        batches = stage.process_dataset_entry(batch.data[0])
        assert len(batches) == 1
        assert batches[0].data[0]["segments"] == batch.data[0]["segments"]

    def test_process_dataset_entry_no_segments_passthrough(self, audio_batch) -> None:
        """Entry with alignment but no segments is returned unchanged."""
        stage = MergeAlignmentDiarizationStage()
        batch = audio_batch(
            alignment=[{"word": "x", "start": 0.0, "end": 0.5}],
            segments=[],
        )
        batches = stage.process_dataset_entry(batch.data[0])
        assert len(batches) == 1
        assert batches[0].data[0]["segments"] == []
