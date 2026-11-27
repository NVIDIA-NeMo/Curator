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

"""Unit tests for WhisperXVADAdapter (the WhisperX model is faked)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nemo_curator.models.audio.vad import whisperx as module
from nemo_curator.models.audio.vad.base import VADResult, VADSegment
from nemo_curator.models.audio.vad.whisperx import WhisperXVADAdapter
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


def test_detect_batch_maps_options_and_drops_short_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_segments(_model: object, audio: np.ndarray, sample_rate: int, **kwargs: Any) -> list[dict[str, float]]:  # noqa: ANN401
        calls.append((audio.shape, sample_rate, kwargs))
        if sample_rate == 0:
            msg = "bad item"
            raise ValueError(msg)
        return [{"start": 0.0, "end": 0.5}, {"start": 1.0, "end": 3.0}]

    monkeypatch.setattr(module, "whisperx_vad_segments", fake_segments)
    adapter = WhisperXVADAdapter(threshold=0.6, min_duration_sec=1.0, max_duration_sec=30.0)
    adapter._model = object()

    results = adapter.detect_batch(
        [
            {"waveform": np.zeros(8, dtype=np.float32), "sample_rate": 16000},
            {"waveform": np.zeros(4), "sample_rate": 0},
        ]
    )

    assert calls[0] == ((1, 8), 16000, {"merge_max_length": 30.0, "vad_onset": 0.6})
    assert results[0] == VADResult(segments=[VADSegment(start=1.0, end=3.0)])
    assert results[1].segments == []
    assert "bad item" in results[1].error


def test_load_model_selects_device_and_unload_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = []
    monkeypatch.setattr(module, "load_whisperx_vad_model", lambda device, **kwargs: loads.append((device, kwargs)))
    adapter = WhisperXVADAdapter(threshold=0.4, vad_offset=0.3)

    adapter.download_weights_on_node()
    adapter.load_model(num_gpus=0)

    assert loads == [("cpu", {"vad_onset": 0.4, "vad_offset": 0.3, "hf_token": None})] * 2
    adapter.unload_model()
    with pytest.raises(RuntimeError, match="load_model"):
        adapter.detect_batch([])


def test_vad_segmentation_stage_runs_whisperx_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "whisperx_vad_segments", lambda *_args, **_kwargs: [{"start": 0.1, "end": 0.4}])
    stage = VADSegmentationStage(
        adapter_target="nemo_curator.models.audio.vad.whisperx.WhisperXVADAdapter",
        min_duration_sec=0.0,
        nested=True,
        resources=Resources(cpus=1.0),
    )
    adapter = stage._create_adapter()
    adapter._model = object()
    stage._adapter = adapter
    task = AudioTask(data={"waveform": np.ones((1, 10), dtype=np.float32), "sample_rate": 10})

    (result,) = stage.process_batch([task])

    assert [(s["start_ms"], s["end_ms"]) for s in result.data["segments"]] == [(100, 400)]
