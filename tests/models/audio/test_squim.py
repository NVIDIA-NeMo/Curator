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

"""Unit tests for TorchSquimAdapter."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_curator.models.audio.squim import TorchSquimAdapter
from nemo_curator.stages.audio.metrics.squim import TorchSquimQualityMetricsStage
from nemo_curator.stages.resources import Resources


def test_score_batch_pads_each_batch_and_keeps_order() -> None:
    model = MagicMock(side_effect=lambda x: (x.sum(dim=1), x.sum(dim=1) + 1, x.sum(dim=1) + 2))
    adapter = TorchSquimAdapter(model=model)

    results = adapter.score_batch([torch.ones(2), torch.ones(3), torch.ones(1)], batch_size=2)

    assert [call.args[0].shape for call in model.call_args_list] == [(2, 3), (1, 1)]
    assert results == [(3.0, 2.0, 4.0), (4.0, 3.0, 5.0), (2.0, 1.0, 3.0)]


def test_stage_setup_and_teardown_use_adapter_on_cpu() -> None:
    stage = TorchSquimQualityMetricsStage(resources=Resources(cpus=1.0))

    with patch("nemo_curator.models.audio.squim.SQUIM_OBJECTIVE") as bundle:
        stage.setup()

    assert isinstance(stage.model, TorchSquimAdapter)
    assert stage.model.model is bundle.get_model.return_value
    stage.teardown()
    assert stage.model is None


def test_gpu_request_without_cuda_fails() -> None:
    stage = TorchSquimQualityMetricsStage(resources=Resources(gpus=1.0))

    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="CUDA is not available"),
    ):
        stage.setup()
