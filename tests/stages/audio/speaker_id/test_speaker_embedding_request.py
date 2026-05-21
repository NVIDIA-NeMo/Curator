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

"""Unit tests for SpeakerEmbeddingRequestStage (model mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio.speaker_id.speaker_embedding_request import (
    SpeakerEmbeddingRequestStage,
)
from nemo_curator.tasks import DocumentBatch


def _write_sine_wav(path: Path, sr: int = 16000, duration_s: float = 1.0) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * 440 * t), sr)


def _mock_speaker_model(emb_dim: int = 192) -> mock.MagicMock:
    model = mock.MagicMock()
    model.device = torch.device("cpu")

    def fake_forward(input_signal, input_signal_length):
        batch = input_signal.shape[0]
        return torch.zeros(batch, 1), torch.randn(batch, emb_dim)

    model.forward = fake_forward
    model.eval = mock.MagicMock()
    return model


def _make_batch(entries: list[dict]) -> DocumentBatch:
    return DocumentBatch(
        task_id="test",
        dataset_name="test",
        data=pd.DataFrame(entries),
    )


class TestSpeakerEmbeddingRequestStage:
    def test_process_adds_embedding_column(self, tmp_path: Path) -> None:
        wav = tmp_path / "audio.wav"
        _write_sine_wav(wav)

        stage = SpeakerEmbeddingRequestStage(speaker_model=_mock_speaker_model())
        batch = _make_batch([{"audio_filepath": str(wav)}])
        result = stage.process(batch)

        assert isinstance(result, DocumentBatch)
        df = result.to_pandas()
        assert "embedding" in df.columns
        emb = df["embedding"].iloc[0]
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (192,)

    def test_missing_file_returns_none_embedding(self, tmp_path: Path) -> None:
        stage = SpeakerEmbeddingRequestStage(speaker_model=_mock_speaker_model())
        batch = _make_batch([{"audio_filepath": str(tmp_path / "missing.wav")}])
        result = stage.process(batch)

        df = result.to_pandas()
        assert df["embedding"].iloc[0] is None

    def test_multiple_rows(self, tmp_path: Path) -> None:
        entries = []
        for i in range(3):
            wav = tmp_path / f"audio_{i}.wav"
            _write_sine_wav(wav)
            entries.append({"audio_filepath": str(wav)})

        stage = SpeakerEmbeddingRequestStage(speaker_model=_mock_speaker_model())
        result = stage.process(_make_batch(entries))

        df = result.to_pandas()
        assert len(df) == 3
        assert all(isinstance(e, np.ndarray) for e in df["embedding"])

    def test_saves_npz_when_output_path_set(self, tmp_path: Path) -> None:
        wav = tmp_path / "audio.wav"
        _write_sine_wav(wav)
        out = str(tmp_path / "embeddings.npz")

        stage = SpeakerEmbeddingRequestStage(
            speaker_model=_mock_speaker_model(),
            output_path=out,
            output_format="npz",
        )
        stage.process(_make_batch([{"audio_filepath": str(wav)}]))

        assert Path(out).exists()
        data = np.load(out, allow_pickle=True)
        assert len(data["cut_ids"]) == 1
        assert data["embeddings"].shape == (1, 192)

    def test_saves_pt_when_output_format_pt(self, tmp_path: Path) -> None:
        wav = tmp_path / "audio.wav"
        _write_sine_wav(wav)
        out = str(tmp_path / "embeddings.pt")

        stage = SpeakerEmbeddingRequestStage(
            speaker_model=_mock_speaker_model(),
            output_path=out,
            output_format="pt",
        )
        stage.process(_make_batch([{"audio_filepath": str(wav)}]))

        assert Path(out).exists()
        data = torch.load(out, weights_only=False)
        assert len(data["cut_ids"]) == 1

    def test_validation_requires_model(self) -> None:
        with pytest.raises(ValueError, match="model_name or speaker_model"):
            SpeakerEmbeddingRequestStage(model_name="", speaker_model=None)

    def test_preserves_existing_columns(self, tmp_path: Path) -> None:
        wav = tmp_path / "audio.wav"
        _write_sine_wav(wav)

        stage = SpeakerEmbeddingRequestStage(speaker_model=_mock_speaker_model())
        batch = _make_batch([{"audio_filepath": str(wav), "text": "hello", "duration": 1.0}])
        result = stage.process(batch)

        df = result.to_pandas()
        assert df["text"].iloc[0] == "hello"
        assert df["duration"].iloc[0] == 1.0
