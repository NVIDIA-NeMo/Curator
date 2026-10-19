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

from nemo_curator.models.asr.nemo_local import LocalNeMoASRAdapter


def test_local_checkpoint_uses_base_adapter_load_hook(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.nemo"
    checkpoint.touch()
    model = MagicMock()
    nemo_asr = MagicMock()
    nemo_asr.models.ASRModel.restore_from.return_value = model
    adapter = LocalNeMoASRAdapter(model_id=str(checkpoint))

    adapter.download_weights_on_node(str(checkpoint))
    with patch("nemo_curator.models.asr.nemo_local._nemo_asr_module", return_value=nemo_asr):
        adapter.load_model(num_gpus=0)

    nemo_asr.models.ASRModel.restore_from.assert_called_once()
    kwargs = nemo_asr.models.ASRModel.restore_from.call_args.kwargs
    assert kwargs["restore_path"] == str(checkpoint)
    assert kwargs["map_location"].type == "cpu"
    assert adapter._model is model


def test_local_checkpoint_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LocalNeMoASRAdapter.download_weights_on_node(str(tmp_path / "missing.nemo"))


def test_local_checkpoint_must_use_nemo_suffix(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()

    with pytest.raises(ValueError, match=r"must end in \.nemo"):
        LocalNeMoASRAdapter.download_weights_on_node(str(checkpoint))
