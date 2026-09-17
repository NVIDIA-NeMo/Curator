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
from typing import Any
from unittest.mock import patch

from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.tagging.prepare_module_segments import (
    PrepareModuleSegmentsStage,
)
from nemo_curator.tasks import AudioTask


class TestPrepareModuleSegmentsStageIsValidSegment:
    """Tests for PrepareModuleSegmentsStage.is_valid_segment."""

    def test_valid_segment_with_text(self) -> None:
        """Segment with words and non-empty text is valid."""
        stage = PrepareModuleSegmentsStage(module="tts")
        segment = {
            "speaker": "s1",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ],
        }
        assert stage.is_valid_segment(segment) is True

    def test_invalid_single_overlong_word(self) -> None:
        """Single word longer than max_duration is invalid."""
        stage = PrepareModuleSegmentsStage(module="tts", max_duration=10.0)
        segment = {
            "speaker": "s1",
            "words": [{"word": "x", "start": 0.0, "end": 25.0}],
        }
        assert stage.is_valid_segment(segment) is False

    def test_invalid_empty_text(self) -> None:
        """Segment with no text (empty words or blank) is invalid."""
        stage = PrepareModuleSegmentsStage(module="tts")
        segment = {"speaker": "s1", "words": []}
        assert stage.is_valid_segment(segment) is False

    def test_invalid_zero_duration(self) -> None:
        """Segment whose end does not follow its start is invalid."""
        stage = PrepareModuleSegmentsStage(module="tts")
        segment = {
            "speaker": "s1",
            "words": [{"word": "hello", "start": 1.0, "end": 1.0}],
        }
        assert stage.is_valid_segment(segment) is False

    def test_valid_single_short_word(self) -> None:
        """A non-empty single word within the duration limit is valid."""
        stage = PrepareModuleSegmentsStage(module="tts", max_duration=10.0)
        segment = {
            "speaker": "s1",
            "words": [{"word": "hello", "start": 0.0, "end": 1.0}],
        }
        assert stage.is_valid_segment(segment) is True


class TestPrepareModuleSegmentsStageSplitSegmentByDuration:
    """Tests for PrepareModuleSegmentsStage.split_segment_by_duration."""

    def test_single_short_segment_unchanged(self) -> None:
        """Short segment within max_duration is returned as single segment."""
        stage = PrepareModuleSegmentsStage(module="tts", min_duration=1.0, max_duration=20.0)
        segment = {
            "speaker": "s1",
            "start": 0.0,
            "end": 5.0,
            "words": [
                {"word": "one", "start": 0.0, "end": 0.5, "speaker": "s1"},
                {"word": "two", "start": 0.5, "end": 1.0, "speaker": "s1"},
            ],
        }
        result = stage.split_segment_by_duration(segment)
        assert len(result) >= 1
        assert all(s["speaker"] == "s1" for s in result)
        all_words = [w for s in result for w in s["words"]]
        assert len(all_words) == 2


def _prepare_module_segments_sdp_style_input() -> dict[str, Any]:
    """Input data matching tests/processors/test_prepare_module_segments.py (SDP run_processors test)."""
    return {
        "audio_filepath": "a.wav",
        "segments": [
            {
                "speaker": "speaker1",
                "start": 2864.73,
                "end": 2865.76,
                "metrics": {
                    "pesq_squim": 3.0,
                    "stoi_squim": 0.996,
                    "sisdr_squim": 26.152,
                    "bandwidth": 12000,
                },
                "text": "can you see the",
                "words": [
                    {"word": "can", "start": 2864.8799999999997, "end": 2865.12},
                    {"word": "you", "start": 2865.12, "end": 2865.2},
                    {"word": "see", "start": 2865.2, "end": 2865.3599999999997},
                    {"word": "the", "start": 2865.3599999999997, "end": 2865.52},
                ],
            },
            {
                "speaker": "no-speaker",
                "start": 2865.76,
                "end": 2875.72,
                "text": "screen no my phone my computer phone rang right as we started chatting and i oh let me share it again i",
                "words": [
                    {"word": "screen", "start": 2865.52, "end": 2866.16},
                    {"word": "no", "start": 2866.3999999999996, "end": 2866.64},
                    {"word": "my", "start": 2866.8799999999997, "end": 2867.12},
                    {"word": "phone", "start": 2867.12, "end": 2867.6},
                    {"word": "my", "start": 2867.6, "end": 2867.7599999999998},
                    {"word": "computer", "start": 2867.7599999999998, "end": 2868.16},
                    {"word": "phone", "start": 2868.16, "end": 2868.3999999999996},
                    {"word": "rang", "start": 2868.3999999999996, "end": 2868.72},
                    {"word": "right", "start": 2868.72, "end": 2868.8799999999997},
                    {"word": "as", "start": 2868.8799999999997, "end": 2869.04},
                    {"word": "we", "start": 2869.04, "end": 2869.2},
                    {"word": "started", "start": 2869.3599999999997, "end": 2869.6},
                    {"word": "chatting", "start": 2869.6, "end": 2869.9199999999996},
                    {"word": "and", "start": 2869.9199999999996, "end": 2870.08},
                    {"word": "i", "start": 2870.08, "end": 2870.24},
                    {"word": "oh", "start": 2870.48, "end": 2870.72},
                    {"word": "let", "start": 2870.72, "end": 2870.8799999999997},
                    {"word": "me", "start": 2870.8799999999997, "end": 2871.12},
                    {"word": "share", "start": 2871.52, "end": 2871.9199999999996},
                    {"word": "it", "start": 2871.9199999999996, "end": 2872.08},
                    {"word": "again", "start": 2872.08, "end": 2872.3199999999997},
                    {"word": "i", "start": 2875.4399999999996, "end": 2875.68},
                ],
            },
            {
                "speaker": "speaker1",
                "start": 2875.72,
                "end": 2876.66,
                "metrics": {
                    "pesq_squim": 3.656,
                    "stoi_squim": 0.995,
                    "sisdr_squim": 21.172,
                    "bandwidth": 12093,
                },
                "text": "just shared it again",
                "words": [
                    {"word": "just", "start": 2875.68, "end": 2875.9199999999996},
                    {
                        "word": "shared",
                        "start": 2875.9199999999996,
                        "end": 2876.3199999999997,
                    },
                    {"word": "it", "start": 2876.3199999999997, "end": 2876.48},
                    {"word": "again", "start": 2876.48, "end": 2876.7999999999997},
                ],
            },
        ],
        "overlap_segments": [],
        "duration": 30,
    }


def test_prepare_module_segments_stage_sdp_style_input(
    audio_task: Callable[..., AudioTask],
) -> None:
    """PrepareModuleSegmentsStage with SDP-style input (speaker1, no-speaker, speaker1) yields 2 TTS segments.

    Mirrors tests/processors/test_prepare_module_segments.py: same input shape and stage params
    (module=tts, min_duration=5, max_duration=20, max_pause=2). no-speaker segment is excluded
    from TTS output; the two speaker1 segments become 2 output segments.
    """
    stage = PrepareModuleSegmentsStage(
        module="tts",
        min_duration=5,
        max_duration=20,
        max_pause=2,
        text_key="text",
        words_key="words",
        terminal_punct_marks=".!?。？？！。",  # noqa: RUF001 - fullwidth CJK punctuation is the test input
        full_utterance_ratio=1.0,
        punctuation_split_only=False,
    )
    data_entry = _prepare_module_segments_sdp_style_input()
    task = audio_task(**data_entry)
    result = stage.process(task)

    out = result.data
    assert "segments" in out
    assert len(out["segments"]) == 2
    assert out["segments"][0]["text"] == "can you see the"
    assert out["segments"][0]["speaker"] == "speaker1"
    assert "metrics" in out["segments"][0]
    assert "pesq_squim" in out["segments"][0]["metrics"]
    assert out["segments"][1]["text"] == "just shared it again"
    assert out["segments"][1]["speaker"] == "speaker1"
    assert "metrics" in out["segments"][1]


class TestPerEntryRandomSeed:
    """Verify that per-entry seeding produces varied but reproducible random sequences."""

    def test_different_entries_get_different_seeds(self) -> None:
        """Two entries with different audio_filepath should produce different random sequences."""
        stage = PrepareModuleSegmentsStage(module="asr", min_duration=5, max_duration=20)
        seeds_used: list[int] = []

        orig_seed = stage._rng.seed

        def capture_seed(s: int) -> None:
            seeds_used.append(s)
            orig_seed(s)

        entry_a = {"audio_filepath": "file_a.wav", "segments": []}
        entry_b = {"audio_filepath": "file_b.wav", "segments": []}

        with patch.object(stage._rng, "seed", side_effect=capture_seed):
            stage.process(AudioTask(data=entry_a))
            stage.process(AudioTask(data=entry_b))

        assert len(seeds_used) == 2
        assert seeds_used[0] != seeds_used[1], "Different entries must get different random seeds"

    def test_same_entry_is_reproducible(self) -> None:
        """Same audio_filepath always produces the same seed (reproducibility)."""
        stage = PrepareModuleSegmentsStage(module="asr", min_duration=5, max_duration=20)
        seeds_used: list[int] = []

        orig_seed = stage._rng.seed

        def capture_seed(s: int) -> None:
            seeds_used.append(s)
            orig_seed(s)

        entry = {"audio_filepath": "file_a.wav", "segments": []}

        with patch.object(stage._rng, "seed", side_effect=capture_seed):
            stage.process(AudioTask(data=entry))
            stage.process(AudioTask(data=entry))

        assert len(seeds_used) == 2
        assert seeds_used[0] == seeds_used[1], "Same entry must always get the same seed"


class TestPrepareModuleSegmentsReviewFixes:
    """Regressions for the PR #2339 re-review safety fixes."""

    def test_renamed_alignment_key_is_read_and_wordless_rows_match_default_keys(self) -> None:
        # The renamed-key chain must produce exactly what the default-key chain produces:
        # words under ``my_alignment`` are consumed, and a row with no aligned words leaves
        # with ``segments == []`` (the pre-conversion behaviour) rather than keeping the raw
        # diarization turns, which are neither split nor scored.
        words = [
            {"word": "hello", "start": 0.2, "end": 0.6},
            {"word": "world.", "start": 0.7, "end": 1.1},
        ]
        renamed = PrepareModuleSegmentsStage(
            module="tts", alignment_key="my_alignment", overlap_segments_key="my_overlap"
        )
        default = PrepareModuleSegmentsStage(module="tts")
        renamed_entry = {
            "segments": [{"speaker": "speaker1", "start": 0.0, "end": 3.0}],
            "duration": 3.0,
            "my_alignment": words,
        }
        default_entry = {
            "segments": [{"speaker": "speaker1", "start": 0.0, "end": 3.0}],
            "duration": 3.0,
            "alignment": words,
        }
        renamed_result = renamed.process(AudioTask(data=renamed_entry)).data["segments"]
        default_result = default.process(AudioTask(data=default_entry)).data["segments"]
        assert renamed_result, "words under the configured alignment_key must be consumed"
        assert renamed_result == default_result

        wordless = renamed.process(
            AudioTask(data={"segments": [{"speaker": "speaker1", "start": 0.0, "end": 3.0}], "duration": 3.0})
        )
        assert wordless.data["segments"] == [], "a row with no aligned words leaves with an empty prepared list"

    def test_all_rejected_segments_yield_an_empty_list_not_the_raw_input(self) -> None:
        # One 25 s word with max_duration=20: nothing survives preparation, so the output must
        # be ``[]`` -- the raw diarization turn (and the synthetic no-speaker turn) must not
        # pass through as if it were a prepared segment.
        stage = PrepareModuleSegmentsStage(module="tts", max_duration=20.0)
        entry = {
            "segments": [{"speaker": "spk0", "start": 0.0, "end": 25.0}],
            "duration": 25.0,
            "alignment": [{"word": "loooong.", "start": 0.0, "end": 25.0}],
        }
        assert stage.process(AudioTask(data=entry)).data["segments"] == []

    def test_renamed_identity_key_yields_distinct_seeds(self) -> None:
        stage = PrepareModuleSegmentsStage(module="asr", audio_filepath_key="my_path")
        seeds_used: list[int] = []
        orig_seed = stage._rng.seed

        def capture_seed(s: int) -> None:
            seeds_used.append(s)
            orig_seed(s)

        with patch.object(stage._rng, "seed", side_effect=capture_seed):
            stage.process(AudioTask(data={"my_path": "file_a.wav", "segments": []}))
            stage.process(AudioTask(data={"my_path": "file_b.wav", "segments": []}))

        assert len(seeds_used) == 2
        assert seeds_used[0] != seeds_used[1], "distinct renamed identity values must seed differently"

    def test_legacy_positional_signature_still_binds(self) -> None:
        stage = PrepareModuleSegmentsStage("asr", 3.0, 25.0, 1.5, "t", "w", ".", 0.5, True, "MyName")
        assert stage.module == "asr"
        assert stage.min_duration == 3.0
        assert stage.max_duration == 25.0
        assert stage.max_pause == 1.5
        assert stage.text_key == "t"
        assert stage.words_key == "w"
        assert stage.terminal_punct_marks == "."
        assert stage.full_utterance_ratio == 0.5
        assert stage.punctuation_split_only is True
        assert stage.name == "MyName"


def test_prepare_module_segments_is_agent_ready() -> None:
    """Conformance: contract shape/roles/serialization hold and declared writes appear at runtime."""
    stage = PrepareModuleSegmentsStage(module="tts", min_duration=1.0, max_duration=20.0)

    def fixture() -> AudioTask:
        return AudioTask(
            data={
                "segments": [
                    {
                        "speaker": "s1",
                        "start": 0.0,
                        "end": 3.0,
                        "text": "hi there",
                        "words": [
                            {"word": "hi", "start": 0.0, "end": 1.0, "speaker": "s1"},
                            {"word": "there", "start": 1.0, "end": 2.5, "speaker": "s1"},
                        ],
                    }
                ],
                "overlap_segments": [],
                "duration": 3.0,
            }
        )

    contract = assert_agent_ready(stage, fixture, available_keys={"segments", "duration"})
    assert contract.reads.data_keys == ["segments", "duration"]
    assert contract.writes.data_keys == ["segments"]
    assert set(contract.optional_reads.data_keys) == {
        "alignment",
        "overlap_segments",
        "audio_filepath",
        "audio_item_id",
    }
