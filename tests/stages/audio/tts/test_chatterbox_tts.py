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

"""Tests for ChatterboxTTSStage."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio.tts.chatterbox_tts import ChatterboxTTSStage
from nemo_curator.tasks import AudioTask

MODULE = "nemo_curator.stages.audio.tts.chatterbox_tts"


@pytest.fixture
def ref_dataset(tmp_path: Path) -> str:
    """Reference voices in wavs/ layout with optional rttms/."""
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
    """Reference voices in MLS layout."""
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


def _fake_model(sample_rate: int = 24000) -> MagicMock:
    model = MagicMock()

    def _generate(text: str, **_kwargs: object) -> torch.Tensor:
        duration_sec = max(0.5, len(text) * 0.02)
        n_samples = int(sample_rate * duration_sec)
        t = torch.linspace(0, duration_sec, n_samples)
        return 0.3 * torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

    model.generate.side_effect = _generate
    return model


def _make_task(
    text: str = "Hello world",
    speaker: str = "Alice",
    conversation_id: str = "conv001",
    task_id: str = "t1",
    **extra_fields: object,
) -> AudioTask:
    data = {
        "utterance": text,
        "speaker": speaker,
        "conversation_id": conversation_id,
        **extra_fields,
    }
    return AudioTask(data=data, task_id=task_id, dataset_name="test")


def _build_stage(
    output_dir: str,
    ref_dataset: str,
    language: str | None = None,
    **overrides: object,
) -> ChatterboxTTSStage:
    kwargs = {
        "output_audio_dir": output_dir,
        "reference_voices_dataset": ref_dataset,
        "language": language,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return ChatterboxTTSStage(**kwargs)


def _inject_model(stage: ChatterboxTTSStage) -> None:
    stage.model = _fake_model()


class TestChatterboxTTSStage:
    """Test suite for ChatterboxTTSStage."""

    def test_stage_properties(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        assert stage.name == "ChatterboxTTSStage"
        assert stage.resources.gpus == 1

    def test_invalid_language_raises(self, output_dir: str, ref_dataset: str) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            _build_stage(output_dir, ref_dataset, language="xx")

    def test_process_raises_not_implemented(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        with pytest.raises(NotImplementedError, match="process_batch"):
            stage.process(_make_task())

    def test_setup_raises_when_no_reference_audio(
        self, output_dir: str, tmp_path: Path
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        stage = _build_stage(output_dir, str(empty_dir))
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            with pytest.raises(ValueError, match="No reference audio found"):
                stage.setup()

    @patch(f"{MODULE}.ChatterboxTTS")
    def test_setup_loads_english_model(
        self, mock_cls: MagicMock, output_dir: str, ref_dataset: str
    ) -> None:
        mock_cls.from_pretrained.return_value = _fake_model()
        stage = _build_stage(output_dir, ref_dataset)
        stage.setup()
        mock_cls.from_pretrained.assert_called_once_with(device="cpu")
        assert stage.model is not None
        assert stage.reference_wavs_list is not None
        stage.teardown()
        assert stage.model is None

    @patch(f"{MODULE}.ChatterboxMultilingualTTS")
    def test_setup_loads_multilingual_model(
        self, mock_cls: MagicMock, output_dir: str, ref_dataset: str
    ) -> None:
        mock_cls.from_pretrained.return_value = _fake_model()
        stage = _build_stage(output_dir, ref_dataset, language="fr")
        stage.setup()
        mock_cls.from_pretrained.assert_called_once_with(device="cpu")
        assert stage.language == "fr"

    def test_process_batch_empty(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage.setup()
        assert stage.process_batch([]) == []

    def test_process_batch_single_entry(self, output_dir: str, ref_dataset: str) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            task = _make_task("Hello world", "Alice", "conv001")
            results = stage.process_batch([task])

        assert len(results) == 1
        out = results[0].data
        assert out["speaker"] == "Alice"
        assert out["conversation_id"] == "conv001"
        assert "audio_filepath" in out
        assert "duration" in out
        assert "reference_voice" in out
        assert out["duration"] > 0
        assert os.path.exists(out["audio_filepath"])

    def test_process_batch_multi_turn(self, output_dir: str, ref_dataset: str) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            tasks = [
                _make_task("Hello Bob", "Alice", "conv001", task_id="t1"),
                _make_task("Hi Alice", "Bob", "conv001", task_id="t2"),
                _make_task("How are you?", "Alice", "conv001", task_id="t3"),
            ]
            results = stage.process_batch(tasks)

        assert len(results) == 3
        assert results[0].data["reference_voice"] == results[2].data["reference_voice"]
        assert results[1].data["reference_voice"] != results[0].data["reference_voice"]
        for r in results:
            assert os.path.exists(r.data["audio_filepath"])

    def test_process_batch_skips_empty_text(self, output_dir: str, ref_dataset: str) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            tasks = [
                _make_task("Valid", "Alice", "conv001", task_id="t1"),
                _make_task("", "Bob", "conv001", task_id="t2"),
                _make_task("   ", "Charlie", "conv001", task_id="t3"),
            ]
            results = stage.process_batch(tasks)

        assert "audio_filepath" in results[0].data
        assert "audio_filepath" not in results[1].data
        assert "audio_filepath" not in results[2].data

    def test_process_batch_text_field_fallback(self, output_dir: str, ref_dataset: str) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            task = AudioTask(
                data={"text": "Fallback text", "speaker": "Bob", "conversation_id": "c1"},
                task_id="t1",
                dataset_name="test",
            )
            results = stage.process_batch([task])

        assert "audio_filepath" in results[0].data

    def test_process_batch_preserves_task_metadata_and_fields(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            task = _make_task(
                "Test",
                task_id="my_id",
                overlap=0.3,
                topic="weather",
            )
            task.dataset_name = "my_dataset"
            results = stage.process_batch([task])

        result = results[0]
        assert result.task_id == "my_id"
        assert result.dataset_name == "my_dataset"
        assert result.data["overlap"] == 0.3
        assert result.data["topic"] == "weather"

    def test_process_batch_idempotent(self, output_dir: str, ref_dataset: str) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            task = _make_task("Idempotent test", "Alice", "conv001")
            path1 = stage.process_batch([task])[0].data["audio_filepath"]
            calls_before = stage.model.generate.call_count
            path2 = stage.process_batch([task])[0].data["audio_filepath"]

        assert path1 == path2
        assert stage.model.generate.call_count == calls_before

    def test_process_batch_multilingual(
        self, output_dir: str, ref_dataset_mls: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset_mls, language="es")
            stage.setup()

            task = _make_task("Hola mundo")
            stage.process_batch([task])

        assert stage.model.generate.call_args.kwargs["language_id"] == "es"

    def test_process_batch_mls_reference_layout(
        self, output_dir: str, ref_dataset_mls: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset_mls, language="ru")
            stage.setup()

            tasks = [
                _make_task("Привет", "Alice", "conv001", task_id="t1"),
                _make_task("Пока", "Bob", "conv001", task_id="t2"),
            ]
            results = stage.process_batch(tasks)

        assert len(results) == 2
        assert results[0].data["reference_voice"] != results[1].data["reference_voice"]

    def test_process_batch_generation_failure_produces_silence(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()
            stage.model.generate.side_effect = RuntimeError("GPU OOM")

            result = stage.process_batch([_make_task("This will fail")])[0]

        audio, _sr = sf.read(result.data["audio_filepath"])
        assert np.allclose(audio, 0.0)
