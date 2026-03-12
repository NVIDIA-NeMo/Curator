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

from nemo_curator.stages.audio.tagging.metrics.bandwidth import BandwidthEstimationStage
from nemo_curator.stages.audio.tagging.metrics.wer import ComputeWERStage

from tests import FIXTURES_DIR


class TestBandwidthEstimationStage:
    """Tests for BandwidthEstimationStage."""

    def test_process_dataset_entry(self, audio_batch, audio_filepath) -> None:
        stage = BandwidthEstimationStage()
        stage.setup()
        batch = audio_batch(
            audio_filepath=str(audio_filepath),
            segments=[
                {"speaker": "s1", "start": 0.0, "end": 1.0, "text": "hello world"}
            ],
        )
        batches = stage.process_dataset_entry(batch.data[0])
        assert len(batches) == 1
        assert (
            batches[0].data[0]["audio_filepath"]
            == str(audio_filepath)
        )
        assert batches[0].data[0]["segments"][0]["metrics"]["bandwidth"] == 7125


class TestComputeWERStage:
    """Tests for ComputeWERStage helpers and process_dataset_entry."""

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

    def test_process_dataset_entry_no_segments_passthrough(self, audio_batch) -> None:
        """Entry without segments is returned unchanged."""
        stage = ComputeWERStage(language="en")
        stage.setup()
        batch = audio_batch(audio_item_id="x")
        batches = stage.process_dataset_entry(batch.data[0])
        assert len(batches) == 1
        assert batches[0].data[0]["audio_item_id"] == "x"

    def test_process_dataset_entry_computes_wer_cer_for_segments(
        self, audio_batch
    ) -> None:
        """Segments with hypothesis and reference get WER/CER metrics."""
        stage = ComputeWERStage(language="en")
        batch = audio_batch(
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
        stage = ComputeWERStage(
            language="en", hypothesis_text_key="text", reference_text_key="reference"
        )
        stage.setup()
        batches = stage.process_dataset_entry(batch.data[0])
        assert len(batches) == 1
        out = batches[0].data[0]
        assert len(out["segments"]) == 2
        expected_wer = [0.0, 0.5]
        for idx, seg in enumerate(out["segments"]):
            assert "metrics" in seg
            assert "wer" in seg["metrics"]
            assert "cer" in seg["metrics"]
            assert "char_rate" in seg["metrics"]
            assert "word_rate" in seg["metrics"]
            assert abs(seg["metrics"]["wer"]["wer"] - expected_wer[idx]) < 1e-4
