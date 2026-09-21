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

import numpy as np
import torch

from nemo_curator.stages.audio.tts.bandwidth import BandwidthAnnotationStage, estimate_bandwidth
from nemo_curator.tasks import AudioTask


def test_estimate_bandwidth_on_tone() -> None:
    sr = 16000
    freq = 4000.0
    t = np.arange(sr, dtype=np.float32) / sr
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    bandwidth = estimate_bandwidth(audio, sr, n_fft=512)
    assert 3000 <= bandwidth <= 5000


def test_annotates_waveform_without_dropping() -> None:
    sr = 16000
    waveform = torch.zeros(1, sr)
    task = AudioTask(data={"waveform": waveform, "sample_rate": sr, "tn_raw": "x"})
    result = BandwidthAnnotationStage().process(task)
    assert result is task
    assert result.data["bandwidth"]["error"] is None
    assert result.data["bandwidth"]["bandwidth"] is not None
    assert result.data["bandwidth"]["bandwidth_sample_rate"] == sr
    assert result.data["tn_raw"] == "x"


def test_missing_audio_writes_error() -> None:
    task = AudioTask(data={"tn_raw": "hello"})
    result = BandwidthAnnotationStage().process(task)
    assert result.data["bandwidth"] == {
        "bandwidth": None,
        "bandwidth_sample_rate": None,
        "error": "missing_audio",
    }
