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
import pytest
import torch

from nemo_curator.stages.audio.metrics.squim import TorchSquimQualityMetricsStage
from nemo_curator.tasks import AudioTask


@pytest.mark.parametrize("residency", ["waveform", "auto"])
@pytest.mark.parametrize("nested", [False, True], ids=["task", "segment"])
@pytest.mark.parametrize("reversed_view", [False, True], ids=["contiguous", "reversed"])
def test_resident_waveform_layout_preserves_samples_through_batch(
    residency: str, nested: bool, reversed_view: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    waveform = np.linspace(-0.25, 0.25, 16000, dtype=np.float32)
    if reversed_view:
        waveform = waveform[::-1]
    original = waveform.copy()
    stage = TorchSquimQualityMetricsStage(input_residency=residency, target_sr=16000)
    data = {"waveform": waveform, "sample_rate": 16000, "retained": "metadata"}
    if residency == "auto":
        data[stage.audio_filepath_key] = "/unused/fallback.wav"
    if nested:
        data[stage.segments_key] = [
            {"start": 0.25, "end": 0.75, "speaker": "speaker_0", "text": "speech", "metrics": {"prior": 4}}
        ]
    else:
        data[stage.metrics_key] = {"prior": 4}
    task = AudioTask(data=data, _metadata={"source": "retained"})
    observed = []

    def compute(waveforms: list[torch.Tensor]) -> list[tuple[float, float, float]]:
        observed.extend(waveforms)
        return [(1.0, 0.5, 2.0) for _ in waveforms]

    monkeypatch.setattr(stage, "_compute_metrics_batched", compute)

    assert stage.process_batch([task]) == [task]
    assert len(observed) == 1
    expected = original[4000:12000] if nested else original
    torch.testing.assert_close(observed[0], torch.from_numpy(expected), rtol=0, atol=0)
    assert observed[0].is_contiguous()
    assert observed[0].dtype == torch.float32
    assert task.data["waveform"] is waveform
    np.testing.assert_array_equal(waveform, original)
    assert task.data["retained"] == "metadata"
    assert task._metadata == {"source": "retained"}
    output = task.data[stage.segments_key][0] if nested else task.data
    assert output[stage.metrics_key] == {
        "prior": 4,
        "pesq_squim": 1.0,
        "stoi_squim": 0.5,
        "sisdr_squim": 2.0,
    }
