# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the per-utterance adapter through native generation and cache I/O."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.tasks import AudioTask
from tutorials.audio.tts_diarization.stages import PaperChatterboxTTSStage


@pytest.fixture
def stage(tmp_path: Path) -> PaperChatterboxTTSStage:
    output = tmp_path / "tts"
    output.mkdir()
    tts = PaperChatterboxTTSStage(output_audio_dir=str(output), exaggeration=[0.5, 1.0], device="cpu")
    tts.model = Mock(sr=24000)
    tts.model.generate.return_value = torch.sin(torch.arange(24000).reshape(1, -1) * 0.1) * 0.2
    return tts


def job(tmp_path: Path, utterance: str = "first") -> AudioTask:
    reference = tmp_path / "reference.wav"
    if not reference.exists():
        sf.write(reference, np.zeros(16000 * 5), 16000)
    return AudioTask(
        data={
            "utterance_id": utterance,
            "speaker": "voice",
            "text": "Hello world",
            "reference_audio": str(reference),
            "chapter_id": "chapter",
            "source_duration": 12,
            "audio_filepath": "/original/source.wav",
            "duration": 12,
        }
    )


def test_preassigned_reference_and_verified_cache_preserve_provenance(stage: PaperChatterboxTTSStage, tmp_path: Path):
    result = stage.process_batch([job(tmp_path)])[0]
    assert result.data["duration"] == 1
    assert result.data["source_duration"] == 12
    assert result.data["chapter_id"] == "chapter"
    assert "conversation_id" not in result.data
    kwargs = stage.model.generate.call_args.kwargs
    assert kwargs["audio_prompt_path"] == str(tmp_path / "reference.wav")
    assert 0.5 <= kwargs["exaggeration"] <= 1.0
    assert "language_id" not in kwargs
    waveform, sample_rate = sf.read(result.data["audio_filepath"])
    assert sample_rate == 24000
    assert 20 * np.log10(np.sqrt(np.mean(waveform**2))) == pytest.approx(-20, abs=0.01)
    cached = stage.process_batch([job(tmp_path)])[0]
    assert stage.model.generate.call_count == 1
    assert cached.data["audio_filepath"] == result.data["audio_filepath"]
    sf.write(tmp_path / "reference.wav", np.ones(16000 * 5) * 0.01, 16000)
    changed = stage.process_batch([job(tmp_path)])[0]
    assert stage.model.generate.call_count == 2
    assert changed.data["audio_filepath"] != result.data["audio_filepath"]


def test_each_utterance_gets_its_own_stable_exaggeration(stage: PaperChatterboxTTSStage, tmp_path: Path):
    stage.process_batch([job(tmp_path, "first"), job(tmp_path, "second")])
    draws = [call.kwargs["exaggeration"] for call in stage.model.generate.call_args_list]
    assert draws[0] != draws[1]
    assert draws == [stage._get_exaggeration("first"), stage._get_exaggeration("second")]


def test_failed_generation_cannot_return_original_audio_or_silence(stage: PaperChatterboxTTSStage, tmp_path: Path):
    stage.model.generate.side_effect = RuntimeError("synthesis failed")
    task = job(tmp_path)
    with pytest.raises(RuntimeError, match="TTS failed"):
        stage.process_batch([task])
    assert "audio_filepath" not in task.data
    assert not list((tmp_path / "tts").glob("*.wav"))


def test_short_reference_is_rejected_before_generation(stage: PaperChatterboxTTSStage, tmp_path: Path):
    task = job(tmp_path)
    sf.write(tmp_path / "reference.wav", np.zeros(16000 * 4), 16000)
    with pytest.raises(ValueError, match="shorter than"):
        stage.process_batch([task])
    stage.model.generate.assert_not_called()
