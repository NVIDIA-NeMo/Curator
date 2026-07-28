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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_curator.models.asr_nemo import NeMoASRAdapter


@patch("nemo_curator.models.asr_nemo.nemo_asr")
def test_local_checkpoint_uses_restore_from(mock_nemo_asr: MagicMock, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.nemo"
    checkpoint.touch()
    model = MagicMock()
    mock_nemo_asr.models.ASRModel.restore_from.return_value = model
    adapter = NeMoASRAdapter(model_path=str(checkpoint), map_location=torch.device("cpu"))

    adapter.download_weights_on_node()
    adapter.setup()

    mock_nemo_asr.models.ASRModel.restore_from.assert_called_once_with(
        restore_path=str(checkpoint),
        map_location=torch.device("cpu"),
    )
    assert adapter.model is model


def test_local_checkpoint_must_exist(tmp_path: Path) -> None:
    adapter = NeMoASRAdapter(model_path=str(tmp_path / "missing.nemo"))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        adapter.download_weights_on_node()


def test_transcription_shapes_are_normalized() -> None:
    hypothesis = MagicMock(text="hello")
    model = MagicMock()
    model.transcribe.return_value = ([[hypothesis]], None)
    adapter = NeMoASRAdapter(model=model)

    assert adapter.transcribe(["audio.wav"]) == ["hello"]
