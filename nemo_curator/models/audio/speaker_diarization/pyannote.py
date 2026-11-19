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

"""PyAnnote implementation of the speaker-diarization adapter contract.

Overlapped turns are returned separately. Non-overlapped turns longer than
``max_length`` are cut at WhisperX VAD boundaries into chunks of random length
between ``min_length`` and ``max_length``; turns no longer than ``min_length``
are dropped.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.models.audio.vad.whisperx import load_whisperx_vad_model, whisperx_vad_segments

from .base import DiarizationInputError, DiarizationResult, DiarizationSegment


def has_overlap(turn: Any, overlaps: list) -> bool:  # noqa: ANN401
    """Return whether ``turn`` intersects any segment in ``overlaps`` (sorted by start)."""
    for overlap in overlaps:
        if overlap.start > turn.end:
            break
        if turn.start <= overlap.start < turn.end or turn.start < overlap.end < turn.end:
            return True
        if overlap.start < turn.start and overlap.end > turn.end:
            return True
    return False


@dataclass
class PyAnnoteDiarizationAdapter:
    """Diarize recordings with a pyannote pipeline plus WhisperX VAD splitting."""

    model_id: str = "pyannote/speaker-diarization-3.1"
    sample_rate: int = 16000
    hf_token: str | None = field(default=None, repr=False)
    segmentation_batch_size: int = 128
    embedding_batch_size: int = 128
    min_length: float = 0.5
    max_length: float = 40.0
    vad_onset: float = 0.5
    vad_offset: float = 0.363

    _pipeline: Any = field(default=None, init=False, repr=False)
    _vad_model: Any = field(default=None, init=False, repr=False)
    _rng: random.Random | None = field(default=None, init=False, repr=False)
    _device: str | None = field(default=None, init=False, repr=False)

    def _load_pipeline(self) -> Any:  # noqa: ANN401
        from pyannote.audio import Pipeline

        return Pipeline.from_pretrained(self.model_id, token=self.hf_token)

    def _load_vad(self, device: str) -> Any:  # noqa: ANN401
        return load_whisperx_vad_model(device, vad_onset=self.vad_onset, vad_offset=self.vad_offset)

    def download_weights_on_node(self) -> None:
        """Cache the pyannote pipeline and the WhisperX VAD weights."""
        self._load_pipeline()
        self._load_vad("cpu")

    def load_model(self, *, num_gpus: int) -> None:
        """Load both models on CUDA when ``num_gpus > 0``, else on CPU."""
        if self._pipeline is not None:
            return
        import torch

        self._device = "cuda" if num_gpus > 0 else "cpu"
        pipeline = self._load_pipeline()
        pipeline.segmentation_batch_size = self.segmentation_batch_size
        pipeline.embedding_batch_size = self.embedding_batch_size
        pipeline.to(torch.device(self._device))
        self._vad_model = self._load_vad(self._device)
        self._pipeline = pipeline
        self._rng = random.Random()  # noqa: S311
        logger.info(f"Initialized PyAnnote diarization on {self._device}")

    def unload_model(self) -> None:
        """Release both models and any CUDA cache they occupied."""
        used_cuda = self._device == "cuda"
        self._pipeline = None
        self._vad_model = None
        self._rng = None
        self._device = None
        if used_cuda:
            import torch

            torch.cuda.empty_cache()

    def _vad_split(  # noqa: PLR0913
        self,
        audio: Any,  # noqa: ANN401
        fs: int,
        start: float,
        end: float,
        speaker: str,
        segments: list[DiarizationSegment],
    ) -> None:
        """Append one turn, cut at VAD boundaries into random-length chunks when it is too long."""
        if end - start <= self.max_length:
            segments.append({"speaker": speaker, "start": start, "end": end})
            return
        chunks = whisperx_vad_segments(
            self._vad_model,
            audio[:, int(start * fs) : int(end * fs)].numpy(),
            fs,
            merge_max_length=self.max_length,
            vad_onset=self.vad_onset,
        )
        i, n = 0, len(chunks)
        while i < n:
            target = self._rng.uniform(self.min_length, self.max_length)
            chunk_start, chunk_end = chunks[i]["start"], chunks[i]["end"]
            if chunk_end - chunk_start >= target:
                i += 1
            else:
                while i < n and chunks[i]["end"] - chunk_start < target:
                    chunk_end = chunks[i]["end"]
                    i += 1
            segments.append({"speaker": speaker, "start": start + chunk_start, "end": start + chunk_end})

    def _diarize_one(self, item: dict[str, Any]) -> DiarizationResult:
        import numpy as np
        import soundfile as sf
        import torch
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        from pyannote.core import Segment

        if "waveform" in item:
            audio = torch.from_numpy(np.asarray(item["waveform"], dtype=np.float32)).unsqueeze(0)
            fs = int(item["sample_rate"])
        else:
            try:
                data, fs = sf.read(item["audio_filepath"], dtype="float32")
            except Exception as exc:
                msg = f"Could not decode {item['audio_filepath']!r}: {exc}"
                raise DiarizationInputError(msg) from exc
            audio = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)

        with ProgressHook() as hook:
            result = self._pipeline({"waveform": audio, "sample_rate": fs}, hook=hook)
        # pyannote-audio 4.x returns DiarizeOutput; 3.x returns the Annotation itself.
        diarization = getattr(result, "speaker_diarization", result)
        overlaps = diarization.get_overlap().segments_list_
        # Crop to the audio length (pyannote can emit turns past the end).
        diarization = diarization.crop(Segment(0, audio.shape[-1] / fs))

        segments: list[DiarizationSegment] = []
        overlap_segments: list[DiarizationSegment] = []
        for turn, _track, speaker in diarization.itertracks(yield_label=True):
            if has_overlap(turn, overlaps):
                overlap_segments.append({"speaker": str(speaker), "start": turn.start, "end": turn.end})
            elif turn.end - turn.start > self.min_length:
                self._vad_split(audio, fs, turn.start, turn.end, str(speaker), segments)
        return DiarizationResult(segments=segments, overlap_segments=overlap_segments)

    def diarize_batch(self, items: list[dict[str, Any]]) -> list[DiarizationResult]:
        """Diarize each item in order."""
        if self._pipeline is None:
            msg = "PyAnnoteDiarizationAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)
        return [self._diarize_one(item) for item in items]
