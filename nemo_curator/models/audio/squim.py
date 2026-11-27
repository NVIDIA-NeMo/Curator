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

"""TorchAudio SQUIM objective (PESQ, STOI, SI-SDR) adapter.

The adapter owns weight caching, device placement and batched inference.
``TorchSquimQualityMetricsStage`` owns task I/O and segment bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger
from torchaudio.pipelines import SQUIM_OBJECTIVE


@dataclass
class TorchSquimAdapter:
    """Score 1-D 16 kHz waveforms with TorchAudio's ``SQUIM_OBJECTIVE`` model.

    ``model``/``device`` may wrap an already-loaded SQUIM model; ``load_model`` fills them otherwise.
    """

    model: Any = field(default=None, repr=False)
    device: str = "cpu"

    def download_weights_on_node(self) -> None:
        """Populate the TorchAudio weight cache."""
        SQUIM_OBJECTIVE.get_model()

    def load_model(self, *, num_gpus: int) -> None:
        """Load the model on CUDA when ``num_gpus > 0``; that requires CUDA."""
        if self.model is not None:
            return
        if num_gpus > 0 and not torch.cuda.is_available():
            msg = "TorchSquimAdapter: GPU requested via resources but CUDA is not available."
            raise RuntimeError(msg)
        self.device = "cuda" if num_gpus > 0 else "cpu"
        model = SQUIM_OBJECTIVE.get_model()
        self.model = model.cuda() if self.device == "cuda" else model
        logger.info(f"Initialized SQUIM model on {self.device}")

    def unload_model(self) -> None:
        """Release the model and CUDA cache."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score_batch(self, waveforms: list[torch.Tensor], batch_size: int) -> list[tuple[float, float, float]]:
        """Return ``(pesq, stoi, si_sdr)`` per waveform, rounded to 3 decimals, in input order."""
        if self.model is None:
            msg = "TorchSquimAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)
        results: list[tuple[float, float, float]] = []
        for i in range(0, len(waveforms), batch_size):
            batch = waveforms[i : i + batch_size]
            max_len = max(w.shape[0] for w in batch)
            padded = torch.zeros(len(batch), max_len)
            for j, w in enumerate(batch):
                padded[j, : w.shape[0]] = w
            padded = padded.to(self.device)
            with torch.no_grad():
                stoi, pesq, si_sdr = self.model(padded)
            results.extend(
                (round(pesq[j].item(), 3), round(stoi[j].item(), 3), round(si_sdr[j].item(), 3))
                for j in range(len(batch))
            )
        return results
