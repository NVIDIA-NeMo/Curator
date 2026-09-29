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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_curator.models.audio.lid.ambernet import AmberNetLIDAdapter


class _AmberNet:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(train_ds={"labels": ["en", "hi", "ta"]})
        self.batch_shape: tuple[int, ...] | None = None
        self.lengths: torch.Tensor | None = None

    def forward(
        self,
        *,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        self.batch_shape = tuple(input_signal.shape)
        self.lengths = input_signal_length.cpu()
        logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]], device=input_signal.device)
        return logits[: input_signal.shape[0]], None


def test_identify_batch_softmaxes_logits_and_preserves_empty_order() -> None:
    adapter = AmberNetLIDAdapter()
    model = _AmberNet()
    adapter._model = model
    adapter._device = torch.device("cpu")

    results = adapter.identify_batch(
        [
            {"waveform": np.ones(2, dtype=np.float32)},
            {"waveform": np.array([], dtype=np.float32)},
            {"waveform": np.ones(4, dtype=np.float32)},
        ]
    )

    assert [result.language for result in results] == ["hi", "", "en"]
    expected = torch.softmax(torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]]), dim=-1).amax(dim=-1)
    assert [results[0].confidence, results[2].confidence] == pytest.approx(expected.tolist())
    assert model.batch_shape == (2, 4)
    torch.testing.assert_close(model.lengths, torch.tensor([2, 4]))


def test_missing_labels_fall_back_to_class_indices() -> None:
    adapter = AmberNetLIDAdapter()
    model = _AmberNet()
    model.cfg = {}
    adapter._model = model
    adapter._device = torch.device("cpu")

    result = adapter.identify_batch([{"waveform": np.ones(2, dtype=np.float32)}])[0]

    assert result.language == "1"


def test_prefetch_uses_return_model_file_without_loading_worker_model() -> None:
    model_type = MagicMock()
    nemo_asr = SimpleNamespace(models=SimpleNamespace(EncDecSpeakerLabelModel=model_type))
    adapter = AmberNetLIDAdapter(model_name="organization/custom-ambernet")

    with patch("nemo_curator.models.audio.lid.ambernet._nemo_asr_module", return_value=nemo_asr):
        adapter.download_weights_on_node()

    model_type.from_pretrained.assert_called_once_with(
        model_name="organization/custom-ambernet",
        return_model_file=True,
    )
    assert adapter._model is None


def test_load_model_uses_requested_cpu_device() -> None:
    model = MagicMock()
    model_type = MagicMock()
    model_type.from_pretrained.return_value = model
    nemo_asr = SimpleNamespace(models=SimpleNamespace(EncDecSpeakerLabelModel=model_type))
    adapter = AmberNetLIDAdapter()

    with patch("nemo_curator.models.audio.lid.ambernet._nemo_asr_module", return_value=nemo_asr):
        adapter.load_model(num_gpus=0)

    model_type.from_pretrained.assert_called_once_with(
        model_name="langid_ambernet",
        map_location=torch.device("cpu"),
    )
    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()


def test_adapter_rejects_non_mono_waveform() -> None:
    adapter = AmberNetLIDAdapter()
    adapter._model = _AmberNet()
    adapter._device = torch.device("cpu")

    with pytest.raises(ValueError, match="mono 1-D"):
        adapter.identify_batch([{"waveform": np.zeros((2, 4), dtype=np.float32)}])
