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

"""Unit tests for UTMOSAdapter."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import torch
from loguru import logger

from nemo_curator.models.audio.utmos import UTMOSAdapter

if TYPE_CHECKING:
    from pathlib import Path


def test_failed_github_load_uses_local_hub_cache(tmp_path: Path) -> None:
    cached_repo = tmp_path / "tarepan_SpeechMOS_v1.2.0"
    cached_repo.mkdir()
    (cached_repo / "hubconf.py").write_text(
        "import torch\n"
        "class Predictor(torch.nn.Module):\n"
        "    def forward(self, waveform, sample_rate):\n"
        "        return waveform.mean(dim=-1)\n"
        "def utmos22_strong():\n"
        "    return Predictor()\n"
    )
    load = torch.hub.load

    def load_with_failed_github(repo: str, *args: object, **kwargs: object) -> object:
        if kwargs.get("source") != "local":
            msg = "original checkpoint load failure"
            raise RuntimeError(msg)
        return load(repo, *args, **kwargs)

    messages = []
    sink = logger.add(messages.append)
    try:
        with (
            patch("torch.hub.get_dir", return_value=str(tmp_path)),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.hub.load", side_effect=load_with_failed_github) as load_model,
        ):
            adapter = UTMOSAdapter()
            adapter.load_model(num_gpus=1)
            adapter.load_model(num_gpus=1)

        assert adapter._model is not None
        assert load_model.call_count == 2
        assert load_model.call_args.args[0] == str(cached_repo)
        assert "original checkpoint load failure" in "".join(messages)
    finally:
        logger.remove(sink)


def test_score_resamples_to_model_rate_and_unload_releases_state() -> None:
    model = MagicMock(return_value=torch.tensor([3.25]))
    model.parameters = lambda: iter([torch.tensor([0.0])])
    adapter = UTMOSAdapter()
    adapter._model = model

    assert adapter.score(torch.zeros(1, 8000), 8000) == 3.25
    assert model.call_args.args[0].shape == (1, 16000)
    assert model.call_args.kwargs["sr"] == 16000

    adapter.unload_model()
    assert adapter._model is None
    assert adapter._resamplers == {}
