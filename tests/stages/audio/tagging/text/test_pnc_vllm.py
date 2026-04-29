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

"""Unit tests for PNCwithvLLMInferenceStage and CleanLLMOutputStage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

from nemo_curator.stages.audio.tagging.text.pnc import (
    CleanLLMOutputStage,
    PNCwithvLLMInferenceStage,
)
from nemo_curator.tasks import AudioTask


def _make_vllm_output(text: str) -> SimpleNamespace:
    """Mimic a single vllm RequestOutput with one CompletionOutput."""
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


class TestPNCwithvLLMInferenceStage:
    """Tests for PNCwithvLLMInferenceStage with mocked VLLMInference."""

    @pytest.fixture
    def stage(self) -> PNCwithvLLMInferenceStage:
        """Create a stage with a fully mocked VLLMInference backend."""
        with patch("nemo_curator.stages.audio.tagging.text.pnc.VLLMInference") as MockVLLM:  # noqa: N806
            mock_vllm = MagicMock()
            MockVLLM.return_value = mock_vllm
            s = PNCwithvLLMInferenceStage(
                prompt={"user": "Punctuate: {text}"},
                model_params={"model": "mock-model"},
            )
            s._vllm = mock_vllm
            yield s

    def test_process_segments(self, stage: PNCwithvLLMInferenceStage, audio_task: Callable[..., AudioTask]) -> None:
        stage._vllm.get_entry_prompt.side_effect = lambda seg: f"prompt:{seg['text']}"
        stage._vllm.process_batch.return_value = [
            _make_vllm_output("Hello world."),
            _make_vllm_output("How are you?"),
        ]

        task = audio_task(
            segments=[
                {"text": "hello world", "start": 0.0, "end": 1.0},
                {"text": "how are you", "start": 1.0, "end": 2.0},
            ],
        )
        result = stage.process(task)

        assert result.data["segments"][0]["generation"] == "Hello world."
        assert result.data["segments"][1]["generation"] == "How are you?"

    def test_process_top_level_text(self, stage: PNCwithvLLMInferenceStage) -> None:
        stage._vllm.get_entry_prompt.return_value = "prompt:hello"
        stage._vllm.process_batch.return_value = [_make_vllm_output("Hello.")]

        task = AudioTask(data={"text": "hello"})
        result = stage.process(task)

        assert result.data["generation"] == "Hello."

    def test_process_skips_empty_text(
        self, stage: PNCwithvLLMInferenceStage, audio_task: Callable[..., AudioTask]
    ) -> None:
        task = audio_task(
            segments=[
                {"text": "", "start": 0.0, "end": 1.0},
                {"start": 1.0, "end": 2.0},
            ],
        )
        result = stage.process(task)

        stage._vllm.process_batch.assert_not_called()
        assert "generation" not in result.data["segments"][0]

    def test_process_no_segments_no_text(self, stage: PNCwithvLLMInferenceStage) -> None:
        task = AudioTask(data={"audio_filepath": "/a.wav"})
        result = stage.process(task)

        stage._vllm.process_batch.assert_not_called()
        assert "generation" not in result.data

    def test_process_batch_multiple_tasks(self, stage: PNCwithvLLMInferenceStage) -> None:
        stage._vllm.get_entry_prompt.side_effect = lambda seg: f"prompt:{seg['text']}"
        stage._vllm.process_batch.return_value = [
            _make_vllm_output("Hello."),
            _make_vllm_output("World."),
            _make_vllm_output("Bye."),
        ]

        tasks = [
            AudioTask(
                data={
                    "segments": [
                        {"text": "hello", "start": 0.0, "end": 1.0},
                        {"text": "world", "start": 1.0, "end": 2.0},
                    ],
                }
            ),
            AudioTask(data={"text": "bye"}),
        ]
        results = stage.process_batch(tasks)

        assert len(results) == 2
        assert results[0].data["segments"][0]["generation"] == "Hello."
        assert results[0].data["segments"][1]["generation"] == "World."
        assert results[1].data["generation"] == "Bye."

    def test_process_batch_empty(self, stage: PNCwithvLLMInferenceStage) -> None:
        assert stage.process_batch([]) == []

    def test_custom_generation_field(self, audio_task: Callable[..., AudioTask]) -> None:
        with patch("nemo_curator.stages.audio.tagging.text.pnc.VLLMInference") as MockVLLM:  # noqa: N806
            mock_vllm = MagicMock()
            MockVLLM.return_value = mock_vllm
            stage = PNCwithvLLMInferenceStage(
                prompt={"user": "Punctuate: {text}"},
                model_params={"model": "mock-model"},
                generation_field="pnc_output",
            )
            stage._vllm = mock_vllm

        mock_vllm.get_entry_prompt.return_value = "prompt"
        mock_vllm.process_batch.return_value = [_make_vllm_output("Result.")]

        task = audio_task(text="test")
        result = stage.process(task)

        assert result.data["pnc_output"] == "Result."

    def test_setup_delegates_to_vllm(self, stage: PNCwithvLLMInferenceStage) -> None:
        stage.setup()
        stage._vllm.setup.assert_called_once()

    def test_setup_on_node_delegates_to_vllm(self, stage: PNCwithvLLMInferenceStage) -> None:
        stage.setup_on_node()
        stage._vllm.setup_on_node.assert_called_once()

    def test_teardown_cleans_up(self, stage: PNCwithvLLMInferenceStage) -> None:
        vllm_mock = stage._vllm
        stage.teardown()
        vllm_mock.clean_up.assert_called_once()

    def test_teardown_handles_none(self) -> None:
        with patch("nemo_curator.stages.audio.tagging.text.pnc.VLLMInference"):
            stage = PNCwithvLLMInferenceStage(
                prompt={"user": "{text}"},
                model_params={"model": "m"},
            )
        stage._vllm = None
        stage.teardown()


class TestCleanLLMOutputStageUtils:
    def test_remove_pncs(self) -> None:
        result = CleanLLMOutputStage.remove_pncs("Hello, World.", ".,?")
        assert result == "hello world"

    def test_remove_pncs_multilingual(self) -> None:
        result = CleanLLMOutputStage.remove_pncs("مرحبا، كيف؟", ".,?")
        assert "،" not in result
        assert "؟" not in result

    def test_clean_llm_output_basic(self) -> None:
        result = CleanLLMOutputStage.clean_llm_output('Hello "world"!', ".,?")
        assert result == "Hello world !"

    def test_clean_llm_output_removes_special(self) -> None:
        text = "Hello—world\nno_think The input text is good"
        result = CleanLLMOutputStage.clean_llm_output(text, ".,?")
        assert "—" not in result
        assert "no_think" not in result
        assert "The input text is" not in result
        assert "good" in result

    def test_clean_llm_output_collapses_repeated_punct(self) -> None:
        result = CleanLLMOutputStage.clean_llm_output("hello...world", ".,?")
        assert ".." not in result

    def test_is_valid_text_true(self) -> None:
        vocab = set("abcdefghijklmnopqrstuvwxyz .,?")
        assert CleanLLMOutputStage.is_valid_text("hello, world.", vocab)

    def test_is_valid_text_false_digit(self) -> None:
        vocab = set("abcdefghijklmnopqrstuvwxyz ")
        assert not CleanLLMOutputStage.is_valid_text("hello 123", vocab)


class TestCleanLLMOutputStageProcess:
    def test_clean_matching_text(self) -> None:
        stage = CleanLLMOutputStage()
        task = AudioTask(
            data={
                "text": "hello world",
                "generation": "Hello, world.",
            }
        )
        result = stage.process(task)
        out = result.data

        assert out["use_bert_pnc"] is False
        assert out["generation_cleaned"] == "Hello, world."

    def test_missing_generation_field(self) -> None:
        stage = CleanLLMOutputStage()
        task = AudioTask(data={"text": "hello"})
        result = stage.process(task)

        assert result.data["use_bert_pnc"] is False
        assert "generation_cleaned" not in result.data

    def test_high_cer_triggers_bert_fallback(self) -> None:
        stage = CleanLLMOutputStage(cer_threshold=0.01)
        task = AudioTask(
            data={
                "text": "hello world",
                "generation": "completely different text here",
            }
        )
        result = stage.process(task)

        assert result.data["use_bert_pnc"] is True
        assert result.data["generation_cleaned"] == "hello world"

    def test_digit_triggers_bert_fallback(self) -> None:
        stage = CleanLLMOutputStage(cer_threshold=1.0)
        task = AudioTask(
            data={
                "text": "hello",
                "generation": "Hello 123.",
            }
        )
        result = stage.process(task)

        assert result.data["use_bert_pnc"] is True

    def test_invalid_chars_trigger_bert_fallback(self) -> None:
        stage = CleanLLMOutputStage(
            vocab_set="abcdefghijklmnopqrstuvwxyz ",
            cer_threshold=1.0,
        )
        task = AudioTask(
            data={
                "text": "hello",
                "generation": "Héllo.",
            }
        )
        result = stage.process(task)

        assert result.data["use_bert_pnc"] is True

    def test_segments_processed_individually(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = CleanLLMOutputStage()
        task = audio_task(
            segments=[
                {"text": "hello", "generation": "Hello."},
                {"text": "world", "generation": "World."},
            ],
        )
        result = stage.process(task)

        for seg in result.data["segments"]:
            assert "use_bert_pnc" in seg
            assert "generation_cleaned" in seg

    def test_segment_without_generation(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = CleanLLMOutputStage()
        task = audio_task(
            segments=[
                {"text": "hello"},
            ],
        )
        result = stage.process(task)

        assert result.data["segments"][0]["use_bert_pnc"] is False
        assert "generation_cleaned" not in result.data["segments"][0]

    def test_vocab_file(self, tmp_path: Path) -> None:
        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n \n")

        stage = CleanLLMOutputStage(vocab_file=str(vocab_path))
        assert " " in stage._full_vocab_set
        assert "a" in stage._full_vocab_set
        assert "." in stage._full_vocab_set  # from punct_marks

    def test_vocab_file_overrides_vocab_set(self, tmp_path: Path) -> None:
        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("xyz")

        stage = CleanLLMOutputStage(
            vocab_set="abcdefghijklmnopqrstuvwxyz ",
            vocab_file=str(vocab_path),
        )
        assert "a" not in stage._full_vocab_set
        assert "x" in stage._full_vocab_set


class TestCleanLLMOutputSecondPassASR:
    """Test PNC cleaning after 2nd-pass ASR.

    2nd-pass ASR produces entries with ``segments`` containing per-segment
    text (no word-level alignment).  PNC processes each segment individually.
    ``update_alignment`` is False (default).
    """

    def test_segments_cleaned_individually(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = CleanLLMOutputStage(cer_threshold=0.01)
        task = audio_task(
            segments=[
                {"text": "hello world", "generation": "Hello, world."},
                {"text": "good morning", "generation": "Good morning."},
            ],
        )
        result = stage.process(task)

        for seg in result.data["segments"]:
            assert seg["use_bert_pnc"] is False
            assert "generation_cleaned" in seg

    def test_high_cer_segment_falls_back(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = CleanLLMOutputStage(cer_threshold=0.01)
        task = audio_task(
            segments=[
                {"text": "hello world", "generation": "Hello, world."},
                {"text": "good morning", "generation": "something completely different"},
            ],
        )
        result = stage.process(task)

        assert result.data["segments"][0]["use_bert_pnc"] is False
        assert result.data["segments"][1]["use_bert_pnc"] is True
        assert result.data["segments"][1]["generation_cleaned"] == "good morning"

    def test_no_alignment_update_by_default(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = CleanLLMOutputStage(cer_threshold=0.01)
        task = audio_task(
            segments=[
                {
                    "text": "hello world",
                    "generation": "Hello, world.",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5},
                        {"word": "world", "start": 0.5, "end": 1.0},
                    ],
                },
            ],
        )
        result = stage.process(task)

        words = result.data["segments"][0]["words"]
        assert words[0]["word"] == "hello"
        assert words[1]["word"] == "world"

    def test_mixed_segments_some_missing_generation(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = CleanLLMOutputStage()
        task = audio_task(
            segments=[
                {"text": "hello", "generation": "Hello."},
                {"text": "", "start": 1.0, "end": 2.0},
                {"text": "world", "generation": "World."},
            ],
        )
        result = stage.process(task)

        assert result.data["segments"][0]["use_bert_pnc"] is False
        assert result.data["segments"][1]["use_bert_pnc"] is False
        assert result.data["segments"][2]["use_bert_pnc"] is False

    def test_batch_pnc_stage_then_clean(self, audio_task: Callable[..., AudioTask]) -> None:
        """Full flow: PNCwithvLLMInferenceStage -> CleanLLMOutputStage for 2nd pass."""
        with patch("nemo_curator.stages.audio.tagging.text.pnc.VLLMInference") as MockVLLM:  # noqa: N806
            mock_vllm = MagicMock()
            MockVLLM.return_value = mock_vllm
            pnc_stage = PNCwithvLLMInferenceStage(
                prompt={"user": "Punctuate: {text}"},
                model_params={"model": "mock-model"},
                generation_field="text_pnc",
            )
            pnc_stage._vllm = mock_vllm

        mock_vllm.get_entry_prompt.side_effect = lambda seg: f"prompt:{seg['text']}"
        mock_vllm.process_batch.return_value = [
            _make_vllm_output("Hello, world."),
            _make_vllm_output("Good morning."),
        ]

        task = audio_task(
            segments=[
                {"text": "hello world", "start": 0.0, "end": 1.0},
                {"text": "good morning", "start": 1.0, "end": 2.0},
            ],
        )
        task = pnc_stage.process(task)

        assert task.data["segments"][0]["text_pnc"] == "Hello, world."
        assert task.data["segments"][1]["text_pnc"] == "Good morning."

        clean_stage = CleanLLMOutputStage(
            generation_field="text_pnc",
            asr_pred_text_key="text",
            cer_threshold=0.01,
        )
        result = clean_stage.process(task)

        for seg in result.data["segments"]:
            assert seg["use_bert_pnc"] is False
            assert "text_pnc_cleaned" in seg


class TestCleanLLMOutputFirstPassASR:
    """Test PNC cleaning after 1st-pass ASR.

    1st-pass ASR produces entries with ``segments`` (per-speaker chunks with
    ``words``), plus a TOP-LEVEL ``text`` and ``alignment`` that concatenate
    all segment words.  PNC runs on the top-level text, and the cleaned
    output is written back to the top-level alignment words.

    To process the top-level entry rather than individual segments, set
    ``segments_key`` to a key that doesn't exist (e.g. ``"__none__"``).
    """

    @staticmethod
    def _first_pass_entry() -> dict:
        """Return a manifest entry mimicking 1st-pass ASR output."""
        return {
            "audio_filepath": "/test/audio.wav",
            "segments": [
                {
                    "text": "hello world",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
                        {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.8},
                    ],
                },
                {
                    "text": "good morning everyone",
                    "start": 1.0,
                    "end": 2.5,
                    "words": [
                        {"word": "good", "start": 1.0, "end": 1.3, "confidence": 0.9},
                        {"word": "morning", "start": 1.3, "end": 1.7, "confidence": 0.8},
                        {"word": "everyone", "start": 1.7, "end": 2.5, "confidence": 0.7},
                    ],
                },
            ],
            "text": "hello world good morning everyone",
            "alignment": [
                {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
                {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.8},
                {"word": "good", "start": 1.0, "end": 1.3, "confidence": 0.9},
                {"word": "morning", "start": 1.3, "end": 1.7, "confidence": 0.8},
                {"word": "everyone", "start": 1.7, "end": 2.5, "confidence": 0.7},
            ],
        }

    def test_alignment_updated_on_exact_match(self) -> None:
        """When chars match exactly, alignment words get punctuated text."""
        data = self._first_pass_entry()
        data["generation"] = "Hello, world. Good morning, everyone."

        stage = CleanLLMOutputStage(
            cer_threshold=0,
            update_alignment=True,
            alignment_key="alignment",
            segments_key="__none__",
        )
        result = stage.process(AudioTask(data=data))

        assert result.data["use_bert_pnc"] is False
        words = [w["word"] for w in result.data["alignment"]]
        assert words == ["Hello,", "world.", "Good", "morning,", "everyone."]
        assert result.data["alignment"][0]["start"] == 0.0
        assert result.data["alignment"][0]["confidence"] == 0.9

    def test_alignment_not_updated_when_disabled(self) -> None:
        data = self._first_pass_entry()
        data["generation"] = "Hello, world. Good morning, everyone."

        stage = CleanLLMOutputStage(
            cer_threshold=0,
            update_alignment=False,
            alignment_key="alignment",
            segments_key="__none__",
        )
        result = stage.process(AudioTask(data=data))

        words = [w["word"] for w in result.data["alignment"]]
        assert words == ["hello", "world", "good", "morning", "everyone"]

    def test_cer_mismatch_falls_back_no_alignment_update(self) -> None:
        """When LLM output is too different, fall back to ASR text."""
        data = self._first_pass_entry()
        data["generation"] = "completely different text here and more"

        stage = CleanLLMOutputStage(
            cer_threshold=0,
            update_alignment=True,
            alignment_key="alignment",
            segments_key="__none__",
        )
        result = stage.process(AudioTask(data=data))

        assert result.data["use_bert_pnc"] is True
        assert result.data["generation_cleaned"] == "hello world good morning everyone"
        words = [w["word"] for w in result.data["alignment"]]
        assert words == ["hello", "world", "good", "morning", "everyone"]

    def test_raises_on_char_mismatch_with_alignment(self) -> None:
        """When CER within threshold but chars differ, raise ValueError."""
        data = self._first_pass_entry()
        data["generation"] = "Hello, worlds. Good morning, everyone."

        stage = CleanLLMOutputStage(
            cer_threshold=1.0,
            update_alignment=True,
            alignment_key="alignment",
            segments_key="__none__",
        )
        with pytest.raises(ValueError, match="Cannot update alignment"):
            stage.process(AudioTask(data=data))

    def test_missing_alignment_key_skips_update(self) -> None:
        """When alignment key not present, skip silently."""
        data = {
            "text": "hello world",
            "generation": "Hello, world.",
        }

        stage = CleanLLMOutputStage(
            cer_threshold=0,
            update_alignment=True,
            alignment_key="alignment",
            segments_key="__none__",
        )
        result = stage.process(AudioTask(data=data))

        assert result.data["use_bert_pnc"] is False
        assert "alignment" not in result.data

    def test_full_flow_pnc_then_clean_with_alignment(self) -> None:
        """Full 1st-pass flow: PNC on top-level text -> CleanLLMOutput with alignment update."""
        with patch("nemo_curator.stages.audio.tagging.text.pnc.VLLMInference") as MockVLLM:  # noqa: N806
            mock_vllm = MagicMock()
            MockVLLM.return_value = mock_vllm
            pnc_stage = PNCwithvLLMInferenceStage(
                prompt={"user": "Punctuate: {text}"},
                model_params={"model": "mock-model"},
                generation_field="text_pnc",
                segments_key="__none__",
            )
            pnc_stage._vllm = mock_vllm

        mock_vllm.get_entry_prompt.return_value = "prompt"
        mock_vllm.process_batch.return_value = [
            _make_vllm_output("Hello, world. Good morning, everyone."),
        ]

        data = self._first_pass_entry()
        task = AudioTask(data=data)
        task = pnc_stage.process(task)

        assert task.data["text_pnc"] == "Hello, world. Good morning, everyone."

        clean_stage = CleanLLMOutputStage(
            generation_field="text_pnc",
            asr_pred_text_key="text",
            cer_threshold=0,
            update_alignment=True,
            alignment_key="alignment",
            segments_key="__none__",
        )
        result = clean_stage.process(task)

        assert result.data["use_bert_pnc"] is False
        words = [w["word"] for w in result.data["alignment"]]
        assert words == ["Hello,", "world.", "Good", "morning,", "everyone."]

    def test_alignment_with_empty_word_entries(self) -> None:
        """Alignment entries with empty words should be skipped during update."""
        data = {
            "text": "hello world",
            "generation": "Hello, world.",
            "alignment": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "", "start": 0.5, "end": 0.6},
                {"word": "world", "start": 0.6, "end": 1.0},
            ],
        }

        stage = CleanLLMOutputStage(
            cer_threshold=0,
            update_alignment=True,
            alignment_key="alignment",
            segments_key="__none__",
        )
        result = stage.process(AudioTask(data=data))

        assert result.data["alignment"][0]["word"] == "Hello,"
        assert result.data["alignment"][1]["word"] == ""
        assert result.data["alignment"][2]["word"] == "world."

    def test_word_count_mismatch_skips_alignment(self) -> None:
        """When PNC produces different word count, alignment update is skipped.

        The LLM merges "hello world" into "Helloworld," giving 2 cleaned
        words vs 3 alignment entries.  Characters still match (both are
        "helloworldtest") so the code reaches _update_alignment_words,
        which detects the count mismatch and leaves alignment untouched.
        """
        data = {
            "text": "hello world test",
            "generation": "Helloworld, test.",
            "alignment": [
                {"word": "hello", "start": 0.0, "end": 0.3},
                {"word": "world", "start": 0.3, "end": 0.6},
                {"word": "test", "start": 0.6, "end": 1.0},
            ],
        }

        stage = CleanLLMOutputStage(
            cer_threshold=0,
            update_alignment=True,
            alignment_key="alignment",
            segments_key="__none__",
        )
        result = stage.process(AudioTask(data=data))

        words = [w["word"] for w in result.data["alignment"]]
        assert words == ["hello", "world", "test"]
