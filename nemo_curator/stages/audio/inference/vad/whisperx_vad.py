# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""WhisperX VAD model helper.

Provides :class:`WhisperXVADModel`, the shared inference wrapper around
WhisperX's ``Pyannote.merge_chunks``-driven VAD. Two callers consume it:

* :class:`~nemo_curator.adapters.vad.WhisperXVADAdapter` - the VAD
  adapter used by :class:`~nemo_curator.stages.audio.inference.vad.VADStage`.
* :class:`~nemo_curator.adapters.diarization.PyAnnoteDiarizationAdapter` -
  uses it to micro-split long PyAnnote speaker turns.

The pre-split ``WhisperXVADStage`` lived here too; it was removed in
favour of the SDP-V2 stage-adapter split.
"""

import os
from typing import TYPE_CHECKING

import torch
from whisperx.audio import SAMPLE_RATE
from whisperx.vads.pyannote import Pyannote, load_vad_model

if TYPE_CHECKING:
    import numpy as np


class WhisperXVADModel:
    """Shared VAD model and ``get_vad_segments`` logic.

    Used by :class:`~nemo_curator.adapters.vad.WhisperXVADAdapter` for
    standalone VAD and by
    :class:`~nemo_curator.adapters.diarization.PyAnnoteDiarizationAdapter`
    for sub-segment VAD of long speaker turns.
    """

    def __init__(
        self,
        device: str = "cuda",
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        use_auth_token: str | None = None,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            msg = "CUDA device requested but not available. Set device='cpu' to run without GPU."
            raise RuntimeError(msg)
        self._device = device
        self._vad_onset = vad_onset
        self._vad_offset = vad_offset
        default_vad_options = {
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        }

        prev = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"
        try:
            self._model = load_vad_model(torch.device(device), token=use_auth_token, **default_vad_options)
        finally:
            if prev is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = prev

    def to(self, device: str) -> None:
        """Move the model to the given device."""
        self._model = self._model.to(torch.device(device))

    def get_vad_segments(
        self,
        audio: "np.ndarray",
        merge_max_length: float,
        sample_rate: int = SAMPLE_RATE,
    ) -> list[dict]:
        """Get voice activity detection segments for the given audio.

        Args:
            audio: NumPy array of shape (C, N).
            merge_max_length: Maximum length for merging chunks in seconds.
            sample_rate: Sample rate of the audio.

        Returns:
            List of VAD segment dicts with "start" and "end" keys.
        """
        vad_segments = self._model(
            {
                "waveform": torch.from_numpy(audio),
                "sample_rate": sample_rate,
            }
        )
        return Pyannote.merge_chunks(vad_segments, merge_max_length, onset=self._vad_onset)
