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

"""Comprehensive tests for ChatterboxTTSStage."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio.tts.chatterbox_tts import (
    SUPPORTED_LANGUAGES,
    ChatterboxTTSStage,
)
from nemo_curator.tasks import AudioTask

MODULE = "nemo_curator.stages.audio.tts.chatterbox_tts"


@pytest.fixture
def ref_dataset(tmp_path: Path) -> str:
    """Create a reference voices dataset with wavs/ and rttms/ layout."""
    wavs_dir = tmp_path / "wavs" / "dialog001"
    wavs_dir.mkdir(parents=True)
    rttms_dir = tmp_path / "rttms" / "dialog001"
    rttms_dir.mkdir(parents=True)

    sr = 16000
    rng = np.random.default_rng()
    for spk in ("spk_A", "spk_B", "spk_C"):
        audio = rng.standard_normal(sr * 5).astype(np.float32)
        sf.write(str(wavs_dir / f"{spk}.wav"), audio, sr)

        rttm_path = rttms_dir / f"{spk}.rttm"
        rttm_path.write_text(
            f"SPEAKER dialog001 1 0.0 2.0 <NA> <NA> {spk} <NA> <NA>\n"
            f"SPEAKER dialog001 1 3.0 1.5 <NA> <NA> {spk} <NA> <NA>\n"
        )

    return str(tmp_path)


@pytest.fixture
def ref_dataset_mls(tmp_path: Path) -> str:
    """Create a reference voices dataset in MLS layout."""
    sr = 16000
    rng = np.random.default_rng()
    for spk_id in ("1234", "5678", "9012"):
        book_dir = tmp_path / spk_id / "book01"
        book_dir.mkdir(parents=True)
        for seg in range(3):
            audio = rng.standard_normal(sr * 2).astype(np.float32)
            sf.write(str(book_dir / f"{spk_id}_book01_{seg:04d}.flac"), audio, sr)

    return str(tmp_path)


@pytest.fixture
def output_dir(tmp_path: Path) -> str:
    return str(tmp_path / "tts_output")


def _make_task(
    text: str = "Hello world",
    speaker: str = "Alice",
    conversation_id: str = "conv001",
    task_id: str = "t1",
    **extra_fields,
) -> AudioTask:
    data = {
        "utterance": text,
        "speaker": speaker,
        "conversation_id": conversation_id,
        **extra_fields,
    }
    return AudioTask(data=data, task_id=task_id, dataset_name="test")


def _fake_model(sample_rate: int = 24000) -> MagicMock:
    """Return a mock Chatterbox model that produces a sine wave."""
    model = MagicMock()

    def _generate(text: str, **_kwargs: object) -> torch.Tensor:
        duration_sec = max(0.5, len(text) * 0.02)
        n_samples = int(sample_rate * duration_sec)
        t = torch.linspace(0, duration_sec, n_samples)
        return 0.3 * torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

    model.generate.side_effect = _generate
    return model


def _build_stage(
    output_dir: str,
    ref_dataset: str,
    language: str | None = None,
    **overrides,
) -> ChatterboxTTSStage:
    """Build a stage without loading a real model."""
    kwargs = {
        "output_audio_dir": output_dir,
        "reference_voices_dataset": ref_dataset,
        "language": language,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return ChatterboxTTSStage(**kwargs)


def _setup_stage_with_mock(
    stage: ChatterboxTTSStage,
    sample_rate: int = 24000,
) -> ChatterboxTTSStage:
    """Inject a fake model and discover reference files."""
    os.makedirs(stage.output_audio_dir, exist_ok=True)
    stage._init_temp_dir()
    stage.model = _fake_model(sample_rate)
    stage._load_reference_audio_files()
    return stage


class TestConstruction:
    def test_english_defaults(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        assert stage.language is None
        assert stage.repetition_penalty == 1.2
        assert stage.exaggeration_range is None
        assert stage.exaggeration == 0.5

    def test_multilingual_defaults(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, language="ru")
        assert stage.language == "ru"
        assert stage.repetition_penalty == 2.0

    def test_custom_repetition_penalty(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, language="de", repetition_penalty=1.5)
        assert stage.repetition_penalty == 1.5

    def test_exaggeration_range(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, exaggeration=[0.2, 0.8])
        assert stage.exaggeration_range == (0.2, 0.8)
        assert stage.exaggeration == 0.2

    def test_exaggeration_scalar(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, exaggeration=0.7)
        assert stage.exaggeration_range is None
        assert stage.exaggeration == 0.7

    def test_uppercase_language_normalized(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, language="RU")
        assert stage.language == "ru"

    def test_invalid_language_raises(self, output_dir: str, ref_dataset: str) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            _build_stage(output_dir, ref_dataset, language="xx")

    def test_gpu_resources(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        assert stage.resources.gpus == 1


class TestModelLoading:
    @patch(f"{MODULE}.ChatterboxTTS")
    def test_loads_english_model(self, mock_cls: MagicMock, output_dir: str, ref_dataset: str) -> None:
        mock_cls.from_pretrained.return_value = MagicMock()
        stage = _build_stage(output_dir, ref_dataset)
        stage._load_model()
        mock_cls.from_pretrained.assert_called_once_with(device="cpu")
        assert stage.model is not None

    @patch(f"{MODULE}.ChatterboxMultilingualTTS")
    def test_loads_multilingual_model(self, mock_cls: MagicMock, output_dir: str, ref_dataset: str) -> None:
        mock_cls.from_pretrained.return_value = MagicMock()
        stage = _build_stage(output_dir, ref_dataset, language="fr")
        stage._load_model()
        mock_cls.from_pretrained.assert_called_once_with(device="cpu")
        assert stage.model is not None


class TestReferenceDiscovery:
    def test_wavs_layout(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        stage._load_reference_audio_files()
        assert stage._reference_layout == "wavs"
        assert len(stage.reference_wavs_list) == 3

    def test_mls_layout(self, output_dir: str, ref_dataset_mls: str) -> None:
        stage = _build_stage(output_dir, ref_dataset_mls)
        stage._load_reference_audio_files()
        assert stage._reference_layout == "mls"
        assert len(stage._speaker_audio_map) == 3
        assert len(stage.reference_wavs_list) == 9

    def test_no_files_raises(self, output_dir: str, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        stage = _build_stage(output_dir, str(empty_dir))
        with pytest.raises(ValueError, match="No reference audio found"):
            stage._load_reference_audio_files()


class TestRTTMProcessing:
    def test_strips_silence(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        stage._init_temp_dir()

        wav_path = os.path.join(ref_dataset, "wavs", "dialog001", "spk_A.wav")
        rttm_path = os.path.join(ref_dataset, "rttms", "dialog001", "spk_A.rttm")

        result = stage._process_audio_with_rttm(wav_path, rttm_path)
        assert result != wav_path
        assert os.path.exists(result)
        info = sf.info(result)
        assert info.duration < sf.info(wav_path).duration + 0.1

    def test_missing_rttm_returns_original(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        stage._init_temp_dir()

        wav_path = os.path.join(ref_dataset, "wavs", "dialog001", "spk_A.wav")
        result = stage._process_audio_with_rttm(wav_path, "/nonexistent.rttm")
        assert result == wav_path

    def test_respects_max_duration(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, max_reference_duration=1.0)
        stage._init_temp_dir()

        wav_path = os.path.join(ref_dataset, "wavs", "dialog001", "spk_A.wav")
        rttm_path = os.path.join(ref_dataset, "rttms", "dialog001", "spk_A.rttm")

        result = stage._process_audio_with_rttm(wav_path, rttm_path)
        info = sf.info(result)
        assert info.duration <= 1.1


class TestSpeakerAssignment:
    def test_consistent_within_conversation(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        path1, id1 = stage._assign_reference("Alice", "conv001")
        path2, id2 = stage._assign_reference("Alice", "conv001")
        assert path1 == path2
        assert id1 == id2

    def test_different_speakers_get_different_refs(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        path_a, _id_a = stage._assign_reference("Alice", "conv001")
        path_b, _id_b = stage._assign_reference("Bob", "conv001")
        assert path_a != path_b

    def test_different_conversations_independent(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        stage._assign_reference("Alice", "conv001")
        stage._assign_reference("Alice", "conv002")
        assert "conv001_Alice" in stage.speaker_to_reference
        assert "conv002_Alice" in stage.speaker_to_reference

    def test_mls_assignment(self, output_dir: str, ref_dataset_mls: str) -> None:
        stage = _build_stage(output_dir, ref_dataset_mls)
        _setup_stage_with_mock(stage)

        path_a, _id_a = stage._assign_reference("Alice", "conv001")
        path_b, _id_b = stage._assign_reference("Bob", "conv001")
        assert path_a != path_b
        assert "conv001_Alice" in stage.speaker_to_ref_id
        assert "conv001_Bob" in stage.speaker_to_ref_id
        assert stage.speaker_to_ref_id["conv001_Alice"] != stage.speaker_to_ref_id["conv001_Bob"]

    def test_wavs_ref_id_is_dialog_speaker(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        _, ref_id = stage._assign_reference("Alice", "conv001")
        assert "/" in ref_id
        assert "chatterbox_ref_" not in ref_id


class TestExaggeration:
    def test_fixed_exaggeration(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, exaggeration=0.6)
        assert stage._get_exaggeration("conv001") == 0.6
        assert stage._get_exaggeration("conv002") == 0.6

    def test_random_exaggeration_per_conversation(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, exaggeration=[0.1, 0.9])
        val1 = stage._get_exaggeration("conv001")
        val2 = stage._get_exaggeration("conv002")
        assert 0.1 <= val1 <= 0.9
        assert 0.1 <= val2 <= 0.9

        assert stage._get_exaggeration("conv001") == val1

    def test_random_exaggeration_consistent_within_conversation(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, exaggeration=[0.0, 1.0])
        val = stage._get_exaggeration("conv_x")
        for _ in range(10):
            assert stage._get_exaggeration("conv_x") == val


class TestNormalization:
    def test_normalizes_to_target_level(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, normalize_level=-20.0)
        loud = torch.randn(1, 24000) * 0.01
        normalised = stage._normalize_audio(loud)
        rms = torch.sqrt(torch.mean(normalised ** 2))
        db = 20 * torch.log10(rms + 1e-8)
        assert abs(db.item() - (-20.0)) < 2.0

    def test_silent_audio_unchanged(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        silent = torch.zeros(1, 24000)
        result = stage._normalize_audio(silent)
        assert torch.allclose(result, silent)

    def test_clips_to_safe_range(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, normalize_level=0.0)
        quiet = torch.randn(1, 24000) * 0.001
        normalised = stage._normalize_audio(quiet)
        assert torch.max(torch.abs(normalised)).item() <= 1.0


class TestOutputFilename:
    def test_deterministic(self):
        name1 = ChatterboxTTSStage._output_filename("conv123", "Alice", "Hello")
        name2 = ChatterboxTTSStage._output_filename("conv123", "Alice", "Hello")
        assert name1 == name2

    def test_different_text_different_name(self):
        name1 = ChatterboxTTSStage._output_filename("conv123", "Alice", "Hello")
        name2 = ChatterboxTTSStage._output_filename("conv123", "Alice", "Goodbye")
        assert name1 != name2

    def test_long_conversation_id_hashed(self):
        long_id = "a" * 50
        name = ChatterboxTTSStage._output_filename(long_id, "Bob", "Hi")
        parts = name.split("_", 1)
        assert len(parts[0]) == 12
        assert parts[0] != long_id[:12]

    def test_similar_conversation_ids_differ(self):
        name1 = ChatterboxTTSStage._output_filename("session1_conv001", "Alice", "Hi")
        name2 = ChatterboxTTSStage._output_filename("session1_conv002", "Alice", "Hi")
        assert name1 != name2

    def test_ends_with_wav(self):
        name = ChatterboxTTSStage._output_filename("c", "s", "t")
        assert name.endswith(".wav")


class TestTurnAudioGeneration:
    def test_english_generate_call(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        stage.model = _fake_model()

        ref_wav = os.path.join(ref_dataset, "wavs", "dialog001", "spk_A.wav")
        audio = stage._generate_turn_audio("Hello world", ref_wav, "conv001")

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        stage.model.generate.assert_called_once()
        call_kwargs = stage.model.generate.call_args
        assert "language_id" not in call_kwargs.kwargs

    def test_multilingual_generate_call(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, language="de")
        stage.model = _fake_model()

        ref_wav = os.path.join(ref_dataset, "wavs", "dialog001", "spk_A.wav")
        audio = stage._generate_turn_audio("Hallo Welt", ref_wav, "conv001")

        assert isinstance(audio, np.ndarray)
        call_kwargs = stage.model.generate.call_args
        assert call_kwargs.kwargs["language_id"] == "de"

    def test_exception_returns_silence(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        stage.model = MagicMock()
        stage.model.generate.side_effect = RuntimeError("GPU OOM")

        ref_wav = os.path.join(ref_dataset, "wavs", "dialog001", "spk_A.wav")
        audio = stage._generate_turn_audio("text", ref_wav, "conv001")
        assert isinstance(audio, np.ndarray)
        assert len(audio) == stage.sample_rate * 2
        assert np.allclose(audio, 0.0)


class TestProcess:
    def test_single_task(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        task = _make_task("Hello world", "Alice", "conv001")
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert "audio_filepath" in result.data
        assert "duration" in result.data
        assert result.data["duration"] > 0
        assert os.path.exists(result.data["audio_filepath"])
        assert result.data["speaker"] == "Alice"
        assert result.data["conversation_id"] == "conv001"

    def test_empty_text_passthrough(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        task = _make_task("", "Alice", "conv001")
        result = stage.process(task)
        assert "audio_filepath" not in result.data

    def test_text_field_fallback(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        task = AudioTask(
            data={"text": "Fallback text", "speaker": "Bob", "conversation_id": "c1"},
            task_id="t1",
            dataset_name="test",
        )
        result = stage.process(task)
        assert "audio_filepath" in result.data

    def test_preserves_extra_fields(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        task = _make_task(
            "Test", "Alice", "conv001",
            overlap=0.3, topic="weather", turn_index=2,
        )
        result = stage.process(task)
        assert result.data["overlap"] == 0.3
        assert result.data["topic"] == "weather"
        assert result.data["turn_index"] == 2


class TestProcessBatch:
    def test_multi_turn_conversation(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        tasks = [
            _make_task("Hello Bob", "Alice", "conv001", task_id="t1"),
            _make_task("Hi Alice", "Bob", "conv001", task_id="t2"),
            _make_task("How are you?", "Alice", "conv001", task_id="t3"),
        ]
        results = stage.process_batch(tasks)

        assert len(results) == 3
        for r in results:
            assert "audio_filepath" in r.data
            assert os.path.exists(r.data["audio_filepath"])
            assert r.data["duration"] > 0

    def test_empty_batch(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        results = stage.process_batch([])
        assert results == []

    def test_consistent_speaker_voices(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        tasks = [
            _make_task("Turn 1", "Alice", "conv001", task_id="t1"),
            _make_task("Turn 2", "Bob", "conv001", task_id="t2"),
            _make_task("Turn 3", "Alice", "conv001", task_id="t3"),
            _make_task("Turn 4", "Bob", "conv001", task_id="t4"),
        ]
        results = stage.process_batch(tasks)

        assert results[0].data["reference_voice"] == results[2].data["reference_voice"]
        assert results[1].data["reference_voice"] == results[3].data["reference_voice"]

    def test_idempotent_generation(self, output_dir: str, ref_dataset: str) -> None:
        """Pre-existing audio files are reused, not regenerated."""
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        task = _make_task("Idempotent test", "Alice", "conv001")
        result1 = stage.process(task)
        path1 = result1.data["audio_filepath"]

        call_count_before = stage.model.generate.call_count
        result2 = stage.process(task)
        assert stage.model.generate.call_count == call_count_before
        assert result2.data["audio_filepath"] == path1

    def test_task_id_preserved(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        task = _make_task("Hi", task_id="my_custom_id")
        result = stage.process(task)
        assert result.task_id == "my_custom_id"

    def test_dataset_name_preserved(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        task = AudioTask(
            data={"utterance": "Hi", "speaker": "A", "conversation_id": "c1"},
            task_id="t1",
            dataset_name="my_dataset",
        )
        result = stage.process(task)
        assert result.dataset_name == "my_dataset"


class TestMultilingual:
    def test_multilingual_with_mls_refs(self, output_dir: str, ref_dataset_mls: str) -> None:
        stage = _build_stage(output_dir, ref_dataset_mls, language="ru")
        _setup_stage_with_mock(stage)

        tasks = [
            _make_task("Привет Боб", "Alice", "conv001", task_id="t1"),
            _make_task("Привет Алиса", "Bob", "conv001", task_id="t2"),
        ]
        results = stage.process_batch(tasks)
        assert len(results) == 2
        for r in results:
            assert "audio_filepath" in r.data

    def test_language_passed_to_generate(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, language="es")
        _setup_stage_with_mock(stage)

        task = _make_task("Hola mundo")
        stage.process(task)

        call_kwargs = stage.model.generate.call_args.kwargs
        assert call_kwargs["language_id"] == "es"

    def test_all_supported_languages_accepted(self, output_dir: str, ref_dataset: str) -> None:
        for lang in SUPPORTED_LANGUAGES:
            stage = _build_stage(output_dir, ref_dataset, language=lang)
            assert stage.language == lang

    def test_exaggeration_range_with_multilingual(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(
            output_dir, ref_dataset, language="de", exaggeration=[0.3, 0.7]
        )
        _setup_stage_with_mock(stage)

        tasks = [
            _make_task("Hallo", "Alice", "conv001", task_id="t1"),
            _make_task("Tschüss", "Alice", "conv002", task_id="t2"),
        ]
        results = stage.process_batch(tasks)
        assert len(results) == 2

        exag1 = stage.conversation_exaggeration["conv001"]
        exag2 = stage.conversation_exaggeration["conv002"]
        assert 0.3 <= exag1 <= 0.7
        assert 0.3 <= exag2 <= 0.7


class TestLifecycle:
    @patch(f"{MODULE}.ChatterboxTTS")
    def test_setup_creates_temp_dir(self, mock_cls: MagicMock, output_dir: str, ref_dataset: str) -> None:
        mock_cls.from_pretrained.return_value = MagicMock()
        stage = _build_stage(output_dir, ref_dataset)

        assert stage.temp_dir is None
        stage.setup()

        assert stage.temp_dir is not None
        assert os.path.isdir(stage.temp_dir)

    @patch(f"{MODULE}.ChatterboxTTS")
    def test_teardown_cleans_up(self, mock_cls: MagicMock, output_dir: str, ref_dataset: str) -> None:
        mock_cls.from_pretrained.return_value = MagicMock()
        stage = _build_stage(output_dir, ref_dataset)
        stage.setup()

        temp_dir = stage.temp_dir
        assert os.path.isdir(temp_dir)

        stage.teardown()
        assert stage.model is None
        assert not os.path.exists(temp_dir)

    @patch(f"{MODULE}.ChatterboxTTS")
    def test_teardown_clears_speaker_state(self, mock_cls: MagicMock, output_dir: str, ref_dataset: str) -> None:
        mock_cls.from_pretrained.return_value = _fake_model()
        stage = _build_stage(output_dir, ref_dataset)
        stage.setup()

        stage._assign_reference("Alice", "conv001")
        assert len(stage.speaker_to_reference) > 0

        stage.teardown()
        assert len(stage.speaker_to_reference) == 0
        assert len(stage.speaker_to_ref_id) == 0
        assert len(stage.conversation_exaggeration) == 0

    @patch(f"{MODULE}.ChatterboxTTS")
    def test_teardown_setup_lifecycle(self, mock_cls: MagicMock, output_dir: str, ref_dataset: str) -> None:
        """After teardown + re-setup, reference paths are valid."""
        mock_cls.from_pretrained.return_value = _fake_model()
        stage = _build_stage(output_dir, ref_dataset)
        stage.setup()

        stage._assign_reference("Alice", "conv001")
        stage.teardown()
        stage.setup()

        ref_path, _ref_id = stage._assign_reference("Alice", "conv001")
        assert os.path.exists(ref_path)

class TestEdgeCases:
    def test_whitespace_only_text(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        task = _make_task("   ", "Alice", "conv001")
        result = stage.process(task)
        assert "audio_filepath" not in result.data

    def test_mixed_empty_and_valid(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        tasks = [
            _make_task("Valid text", "Alice", "conv001", task_id="t1"),
            _make_task("", "Bob", "conv001", task_id="t2"),
            _make_task("Also valid", "Charlie", "conv001", task_id="t3"),
        ]
        results = stage.process_batch(tasks)
        assert len(results) == 3

        assert "audio_filepath" in results[0].data
        assert "audio_filepath" not in results[1].data
        assert "audio_filepath" in results[2].data

    def test_many_speakers_exhaust_reference_pool(self, output_dir: str, ref_dataset: str) -> None:
        """More speakers than reference files still works (reuses pool)."""
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        tasks = [
            _make_task(f"Turn {i}", f"Speaker_{i}", "conv001", task_id=f"t{i}")
            for i in range(10)
        ]
        results = stage.process_batch(tasks)
        assert len(results) == 10
        assert all("audio_filepath" in r.data for r in results)

    def test_multiple_conversations_in_batch(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)

        tasks = [
            _make_task("Hello", "Alice", "conv001", task_id="t1"),
            _make_task("Bonjour", "Alice", "conv002", task_id="t2"),
        ]
        results = stage.process_batch(tasks)
        assert len(results) == 2
        assert "conv001_Alice" in stage.speaker_to_reference
        assert "conv002_Alice" in stage.speaker_to_reference

    def test_generation_failure_produces_silence(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        _setup_stage_with_mock(stage)
        stage.model.generate.side_effect = RuntimeError("GPU OOM")

        task = _make_task("This will fail")
        result = stage.process(task)
        assert "audio_filepath" in result.data
        audio, _sr = sf.read(result.data["audio_filepath"])
        assert np.allclose(audio, 0.0)
