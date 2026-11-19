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

"""UTMOS (utmos22_strong from tarepan/SpeechMOS) MOS-prediction adapter.

The adapter owns weight caching, model lifecycle, resampling to the model rate,
and inference. ``UTMOSFilterStage`` owns task I/O and the filter decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torchaudio
from loguru import logger

_UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
_UTMOS_ENTRYPOINT = "utmos22_strong"
_UTMOS_TARGET_SR = 16000


def _hub_load(repo: str, **kwargs: Any) -> torch.nn.Module:  # noqa: ANN401
    return torch.hub.load(repo, _UTMOS_ENTRYPOINT, trust_repo=True, skip_validation=True, **kwargs)


@dataclass
class UTMOSAdapter:
    """Predict a mean opinion score for one mono waveform."""

    sample_rate: int = _UTMOS_TARGET_SR
    _model: Any = field(default=None, init=False, repr=False)
    _resamplers: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def download_weights_on_node(self) -> None:
        """Populate the torch.hub cache without keeping the model."""
        _hub_load(_UTMOS_REPO, force_reload=False)

    def load_model(self, *, num_gpus: int) -> None:
        """Load the predictor, falling back to the local torch.hub cache when offline."""
        if self._model is not None:
            return
        use_cuda = num_gpus > 0 and torch.cuda.is_available()
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if use_cuda else "cpu")
        try:
            predictor = _hub_load(_UTMOS_REPO, force_reload=False)
        except Exception:  # noqa: BLE001
            repo, ref = _UTMOS_REPO.rsplit(":", 1)
            cached_repo = Path(torch.hub.get_dir()) / f"{repo.replace('/', '_')}_{ref.replace('/', '_')}"
            logger.opt(exception=True).warning(f"UTMOS initial load failed; trying local cache at {cached_repo}")
            try:
                predictor = _hub_load(str(cached_repo), source="local")
            except Exception as e:
                msg = "UTMOS model failed to load. Check network connectivity and torch.hub cache."
                raise RuntimeError(msg) from e

        predictor = predictor.to(device)
        predictor.eval()
        with torch.no_grad():
            predictor(torch.randn(1, self.sample_rate, device=device), self.sample_rate)
        self._model = predictor
        logger.info(f"UTMOS model loaded on {device}")

    def unload_model(self) -> None:
        """Release the predictor, resamplers, and CUDA cache."""
        self._model = None
        self._resamplers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """Return the MOS for a ``(1, N)`` float waveform at ``sample_rate``."""
        if self._model is None:
            msg = "UTMOSAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)
        device = next(self._model.parameters()).device
        waveform = waveform.to(device)
        if sample_rate != self.sample_rate:
            if sample_rate not in self._resamplers:
                self._resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, self.sample_rate).to(
                    device
                )
            waveform = self._resamplers[sample_rate](waveform)
        with torch.no_grad():
            score = self._model(waveform, sr=self.sample_rate)
        return float(score.item() if torch.is_tensor(score) else score)
