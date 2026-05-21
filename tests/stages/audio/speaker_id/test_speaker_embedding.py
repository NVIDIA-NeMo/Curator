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

"""Unit tests for speaker embedding stages (models mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import torch

from nemo_curator.stages.audio.speaker_id.speaker_embedding_audiotask import (
    SpeakerEmbeddingAudioTaskStage,
)
from nemo_curator.tasks import AudioTask


def _make_audio_task(
    duration_s: float = 1.0,
    sr: int = 16000,
    task_id: str = "test",
    audio_filepath: str = "audio_0000.wav",
) -> AudioTask:
    n_samples = int(duration_s * sr)
    return AudioTask(
        task_id=task_id,
        dataset_name="test",
        data={
            "waveform": np.random.randn(n_samples).astype(np.float32),
            "sample_rate": sr,
            "audio_filepath": audio_filepath,
        },
    )


def _mock_speaker_model(emb_dim: int = 192) -> mock.MagicMock:
    model = mock.MagicMock()
    model.device = torch.device("cpu")

    def fake_forward(input_signal, input_signal_length):
        batch = input_signal.shape[0]
        logits = torch.zeros(batch, 1)
        embs = torch.randn(batch, emb_dim)
        return logits, embs

    model.forward = fake_forward
    model.eval = mock.MagicMock()
    return model


class TestSpeakerEmbeddingAudioTaskStage:
    def test_process_adds_embedding(self) -> None:
        stage = SpeakerEmbeddingAudioTaskStage(
            speaker_model=_mock_speaker_model(),
        )

        task = _make_audio_task()
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert "embedding" in result.data
        assert isinstance(result.data["embedding"], np.ndarray)
        assert result.data["embedding"].shape == (192,)

    def test_process_accumulates_ids(self) -> None:
        stage = SpeakerEmbeddingAudioTaskStage(
            speaker_model=_mock_speaker_model(),
        )

        stage.process(_make_audio_task(audio_filepath="a.wav"))
        stage.process(_make_audio_task(audio_filepath="b.wav"))

        assert len(stage._accumulated_ids) == 2
        assert stage._accumulated_ids == ["a.wav", "b.wav"]

    def test_flush_npz_writes_file(self, tmp_path: Path) -> None:
        stage = SpeakerEmbeddingAudioTaskStage(
            speaker_model=_mock_speaker_model(),
            output_dir=str(tmp_path),
        )

        stage.process(_make_audio_task(audio_filepath="a.wav"))
        stage.process(_make_audio_task(audio_filepath="b.wav"))
        stage._flush_npz()

        npz_files = list(tmp_path.glob("embeddings_*.npz"))
        assert len(npz_files) == 1

        data = np.load(str(npz_files[0]), allow_pickle=True)
        assert len(data["cut_ids"]) == 2
        assert data["embeddings"].shape == (2, 192)

    def test_flush_clears_buffer(self, tmp_path: Path) -> None:
        stage = SpeakerEmbeddingAudioTaskStage(
            speaker_model=_mock_speaker_model(),
            output_dir=str(tmp_path),
        )

        stage.process(_make_audio_task())
        stage._flush_npz()

        assert len(stage._accumulated_ids) == 0
        assert len(stage._accumulated_embs) == 0

    def test_no_output_dir_skips_flush(self) -> None:
        stage = SpeakerEmbeddingAudioTaskStage(
            speaker_model=_mock_speaker_model(),
            output_dir="",
        )

        stage.process(_make_audio_task())
        stage._flush_npz()  # should not raise

    def test_custom_embedding_key(self) -> None:
        stage = SpeakerEmbeddingAudioTaskStage(
            speaker_model=_mock_speaker_model(),
            embedding_key="spk_emb",
        )

        result = stage.process(_make_audio_task())
        assert "spk_emb" in result.data

    def test_validation_requires_model(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="model_name or speaker_model"):
            SpeakerEmbeddingAudioTaskStage(model_name="", speaker_model=None)

    def test_process_batch_flushes(self, tmp_path: Path) -> None:
        stage = SpeakerEmbeddingAudioTaskStage(
            speaker_model=_mock_speaker_model(),
            output_dir=str(tmp_path),
        )

        tasks = [_make_audio_task(audio_filepath=f"f{i}.wav") for i in range(3)]
        results = stage.process_batch(tasks)

        assert len(results) == 3
        npz_files = list(tmp_path.glob("embeddings_*.npz"))
        assert len(npz_files) == 1
        assert len(stage._accumulated_ids) == 0
