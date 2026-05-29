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

"""PyAnnote diarization adapter.

Implements :class:`~nemo_curator.adapters.diarization.DiarizationAdapter`
on top of PyAnnote 3.x / 4.x's
``pyannote.audio.pipelines.SpeakerDiarization``. All PyAnnote-specific
behaviour (HF auth, in-pipeline batching knobs, overlap detection,
PyAnnote-Segment ``has_overlap`` walk, WhisperX-VAD-driven micro-split
of long turns, RTTM sidecar write) lives here so the generic
:class:`~nemo_curator.stages.audio.inference.speaker_diarization.DiarizationStage`
stays model-agnostic.

Logic moved verbatim from the pre-split
``nemo_curator.stages.audio.inference.speaker_diarization.pyannote.PyAnnoteDiarizationStage``;
numeric output is identical for the same inputs and same random seed.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf
import torch
from fsspec.core import url_to_fs
from loguru import logger

# PyAnnote imports - kept module-level (matches pre-split stage) because
# the adapter is imported only inside DiarizationStage.setup() on a GPU
# worker that already has PyAnnote available.
from pyannote.audio import Pipeline as PyAnnotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment

from nemo_curator.adapters.diarization.base import DiarizationResult, DiarSegment
from nemo_curator.stages.audio.inference.vad.whisperx_vad import WhisperXVADModel

if TYPE_CHECKING:
    from pyannote.core import Annotation


def has_overlap(turn: Segment, overlaps: list) -> bool:
    """Check if a given turn overlaps with any segment in the overlaps list.

    Args:
        turn: A segment representing a speech turn.
        overlaps: List of overlap segments, sorted by start time.

    Returns:
        True if the turn overlaps with any segment, False otherwise.
    """
    turn_overlaps = False
    for overlap in overlaps:
        if overlap.start > turn.end:
            # Overlap happens after turn, no need to keep looping since overlaps is sorted
            break
        if overlap.start >= turn.start and overlap.start < turn.end:
            # Overlap starts during turn
            turn_overlaps = True
            break
        if (overlap.end < turn.end) and (overlap.end > turn.start):
            # Overlap ends during turn
            turn_overlaps = True
            break
        if overlap.start < turn.start and overlap.end > turn.end:
            # Overlap completely contains the turn
            turn_overlaps = True
            break
    return turn_overlaps


@dataclass
class PyAnnoteDiarizationAdapter:
    """PyAnnote-backed implementation of :class:`DiarizationAdapter`.

    Tier-2 knobs (set via ``adapter_kwargs`` in YAML; the stage forwards
    them verbatim):

    Attributes:
        model_id: HuggingFace model id (e.g. ``pyannote/speaker-diarization-3.1``).
            Mirrors the stage's ``model_id`` Tier-1 field; passed through
            by the stage.
        revision: HuggingFace revision pin or ``None``. Currently unused
            by PyAnnote's ``from_pretrained`` but accepted for protocol
            uniformity.
        hf_token: HuggingFace authentication token (required - PyAnnote
            models are gated).
        device: ``"cuda"`` or ``"cpu"``. The stage passes the worker's
            actual device; default ``"cuda"`` matches the pre-split
            behaviour.
        segmentation_batch_size: Forwarded to PyAnnote pipeline.
        embedding_batch_size: Forwarded to PyAnnote pipeline.
        min_length: Minimum speech-turn duration kept by the adapter
            (shorter turns are dropped before being returned to the
            stage).
        max_length: Speech-turn length above which the adapter runs a
            WhisperX-VAD-driven micro-split to break long turns into
            sub-segments bounded between ``min_length`` and
            ``max_length``.
        write_rttm: When True, write an RTTM sidecar next to the input
            audio (same path with ``.rttm`` extension). Pre-split
            behaviour was unconditional write; we keep True as the
            default for compatibility.
        random_seed: Optional integer seed for the WhisperX-VAD-driven
            micro-split's uniform sampling. ``None`` (default) matches
            the pre-split non-deterministic behaviour.
    """

    # ---- Required protocol fields ----
    model_id: str = "pyannote/speaker-diarization-3.1"
    revision: str | None = None

    # ---- PyAnnote-specific knobs ----
    hf_token: str = ""
    device: str = "cuda"
    segmentation_batch_size: int = 128
    embedding_batch_size: int = 128
    min_length: float = 0.5
    max_length: float = 40.0
    write_rttm: bool = True
    random_seed: int | None = None

    # ---- Internal state (not serialised, populated in setup()) ----
    _pipeline: Any = field(default=None, repr=False)
    _vad_model: Any = field(default=None, repr=False)
    _rng: random.Random | None = field(default=None, repr=False)
    last_metrics: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Adapter contract: prefetch + setup + teardown + diarize_batch
    # ------------------------------------------------------------------

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        """Download PyAnnote pipeline weights to local cache.

        PyAnnote's ``Pipeline.from_pretrained`` is the only public entry
        point that triggers the HF download; calling it once with a
        valid token caches the segmentation + embedding sub-models on
        disk. We tolerate the absence of an ``HF_TOKEN`` env var during
        prefetch (the stage's ``prefetch_fail_on_error=False`` path will
        defer the actual setup until the worker is up).
        """
        del revision  # PyAnnote uses ``from_pretrained`` without revision arg.
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            msg = (
                "PyAnnoteDiarizationAdapter.prefetch_weights: HF_TOKEN env var "
                "is not set; weights will be downloaded lazily on the worker."
            )
            raise RuntimeError(msg)
        PyAnnotePipeline.from_pretrained(model_id, token=hf_token)

    def setup(self) -> None:
        if self._pipeline is None:
            self._pipeline = PyAnnotePipeline.from_pretrained(
                self.model_id, token=self.hf_token or None
            )
        self._pipeline.segmentation_batch_size = int(self.segmentation_batch_size)
        self._pipeline.embedding_batch_size = int(self.embedding_batch_size)

        if self._vad_model is None:
            self._vad_model = WhisperXVADModel(
                device=self.device,
                vad_onset=0.5,
                vad_offset=0.363,
            )

        self._pipeline.to(torch.device(self.device))
        self._vad_model.to(self.device)

        self._rng = random.Random(self.random_seed) if self.random_seed is not None else random.Random()  # noqa: S311
        logger.info("PyAnnoteDiarizationAdapter ready on {} (model={})", self.device, self.model_id)

    def teardown(self) -> None:
        self._pipeline = None
        self._vad_model = None
        self._rng = None

    def diarize_batch(self, items: list[dict[str, Any]]) -> list[DiarizationResult]:
        if not items:
            return []
        if self._pipeline is None:
            msg = "PyAnnoteDiarizationAdapter.setup() must be called before diarize_batch()"
            raise RuntimeError(msg)

        results: list[DiarizationResult] = []
        per_item_times: list[float] = []
        per_item_speakers: list[int] = []
        per_item_segments: list[int] = []
        per_item_overlaps: list[int] = []

        for item in items:
            t0 = time.perf_counter()
            audio_filepath = item.get("audio_filepath")
            if not audio_filepath:
                # Empty result; the stage's add_non_speaker_segments still runs.
                results.append(DiarizationResult(diar_segments=[], model_id=self.model_id))
                per_item_times.append(time.perf_counter() - t0)
                per_item_speakers.append(0)
                per_item_segments.append(0)
                per_item_overlaps.append(0)
                continue
            result = self._diarize_one(item)
            results.append(result)
            per_item_times.append(time.perf_counter() - t0)
            per_item_speakers.append(
                len({seg.speaker for seg in result.diar_segments if seg.speaker != "no-speaker"})
            )
            per_item_segments.append(len(result.diar_segments))
            per_item_overlaps.append(len(result.overlap_segments))

        self.last_metrics = {
            "batch_size": float(len(items)),
            "diarize_time_s_total": float(sum(per_item_times)),
            "diarize_time_s_max": float(max(per_item_times)) if per_item_times else 0.0,
            "speakers_detected_max": float(max(per_item_speakers)) if per_item_speakers else 0.0,
            "segments_detected_total": float(sum(per_item_segments)),
            "overlap_segments_detected_total": float(sum(per_item_overlaps)),
        }
        return results

    # ------------------------------------------------------------------
    # Internal helpers (moved verbatim from PyAnnoteDiarizationStage)
    # ------------------------------------------------------------------

    def _diarize_one(self, item: dict[str, Any]) -> DiarizationResult:
        audio_filepath: str = item["audio_filepath"]
        audio_item_id: str | None = item.get("audio_item_id")

        # Pre-split behaviour: read with soundfile (avoids torchcodec/FFmpeg).
        data, fs = sf.read(audio_filepath, dtype="float32")
        s = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
        logger.info("Processing {}", audio_filepath)

        with ProgressHook() as hook:
            result = self._pipeline({"waveform": s, "sample_rate": fs}, hook=hook)

        # pyannote-audio 4.x returns DiarizeOutput; older returns Annotation.
        diarization: Annotation = (
            result.speaker_diarization if hasattr(result, "speaker_diarization") else result
        )

        overlaps = diarization.get_overlap().segments_list_

        # Crop to audio length (fix for the historical PyAnnote bug
        # where annotations could extend a few ms past the waveform).
        diarization = diarization.crop(Segment(0, len(s[0]) / fs))

        if self.write_rttm:
            self._write_rttm(diarization, audio_filepath)

        diar_segments: list[DiarSegment] = []
        overlap_segments: list[DiarSegment] = []

        for speech_turn, _track, speaker in diarization.itertracks(yield_label=True):
            if audio_item_id:
                speaker_id = f"{audio_item_id}_{speaker}"
            elif item.get("speaker_id"):
                speaker_id = f"{item['speaker_id']}_{speaker}"
            else:
                speaker_id = f"{Path(audio_filepath).stem}_{speaker}"

            if has_overlap(speech_turn, overlaps):
                overlap_segments.append(
                    DiarSegment(
                        start=float(speech_turn.start),
                        end=float(speech_turn.end),
                        speaker=speaker_id,
                    )
                )
                continue

            speech_duration = speech_turn.end - speech_turn.start
            if speech_duration > self.min_length:
                self._add_vad_segments(
                    audio=s,
                    fs=fs,
                    start=float(speech_turn.start),
                    end=float(speech_turn.end),
                    segments=diar_segments,
                    speaker_id=speaker_id,
                )

        return DiarizationResult(
            diar_segments=diar_segments,
            overlap_segments=overlap_segments,
            model_id=self.model_id,
        )

    def _write_rttm(self, diarization: Annotation, audio_filepath: str) -> None:
        logger.info("Writing {} turns to RTTM file", len(diarization._tracks))  # noqa: SLF001
        rttm_filepath = os.path.splitext(audio_filepath)[0] + ".rttm"
        rttm_fs, rttm_path = url_to_fs(rttm_filepath)
        with rttm_fs.open(rttm_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)

    def _add_vad_segments(  # noqa: PLR0913
        self,
        audio: torch.Tensor,
        fs: int,
        start: float,
        end: float,
        segments: list[DiarSegment],
        speaker_id: str,
    ) -> None:
        """Sub-split a long speech turn using WhisperX VAD timings.

        For turns longer than ``max_length`` we run WhisperX VAD on the
        slice and pack contiguous VAD segments into sub-turns of
        random length sampled uniformly from
        [``min_length``, ``max_length``]. Mirrors the pre-split stage
        implementation byte-for-byte (same uniform sample, same
        index walk).
        """
        segment_duration = end - start

        if segment_duration > self.max_length:
            audio_seg = audio[:, int(start * fs) : int(end * fs)]
            vad_segments = self._vad_model.get_vad_segments(
                audio_seg.numpy(), self.max_length, sample_rate=fs
            )
            i = 0
            n = len(vad_segments)

            while i < n:
                random_duration = self._rng.uniform(self.min_length, self.max_length)
                start_seg = vad_segments[i]["start"]
                end_seg = vad_segments[i]["end"]

                if end_seg - start_seg >= random_duration:
                    segments.append(
                        DiarSegment(
                            start=start + start_seg,
                            end=start + end_seg,
                            speaker=speaker_id,
                        )
                    )
                    i += 1
                    continue

                while i < n and (vad_segments[i]["end"] - start_seg) < random_duration:
                    end_seg = vad_segments[i]["end"]
                    i += 1

                segments.append(
                    DiarSegment(
                        start=start + start_seg,
                        end=start + end_seg,
                        speaker=speaker_id,
                    )
                )
        else:
            segments.append(DiarSegment(start=start, end=end, speaker=speaker_id))
