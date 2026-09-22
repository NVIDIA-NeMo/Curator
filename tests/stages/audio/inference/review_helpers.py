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

"""Shared model-free helpers for audio inference tests."""

# ruff: noqa: F401, TC002

from __future__ import annotations

import inspect
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio._agent._residency import resolve_audio_path
from nemo_curator.stages.audio.inference import base as inference_base
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.audio.inference.base import _channel_first_waveform
from nemo_curator.stages.audio.inference.speaker_diarization import pyannote as pyannote_module
from nemo_curator.stages.audio.inference.speaker_diarization.pyannote import PyAnnoteDiarizationStage
from nemo_curator.stages.audio.inference.speaker_diarization.sortformer import InferenceSortformerStage
from nemo_curator.stages.audio.inference.vad.whisperx_vad import WhisperXVADStage
from nemo_curator.stages.audio.tagging.utils import add_non_speaker_segments
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_ASR_TARGET = "nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter"
_SAMPLE_RATE = 10
_SEGMENTS = [
    {"start": 0.2, "end": 0.6, "speaker": "speaker_0"},
    {"start": 0.7, "end": 1.0, "speaker": "speaker_0"},
]


class _ProgressHook(AbstractContextManager):
    def __exit__(self, *_args: object) -> None:
        return None


class _FakeAnnotation:
    def __init__(
        self,
        segments: list[dict[str, Any]],
        *,
        overlap_segments: list[object] | None = None,
        fail_after_rttm: bool = False,
    ) -> None:
        self._segments = segments
        self._overlap_segments = overlap_segments or []
        self._tracks = list(segments)
        self._fail_after_rttm = fail_after_rttm

    def get_overlap(self) -> SimpleNamespace:
        return SimpleNamespace(segments_list_=self._overlap_segments)

    def crop(self, _segment: object) -> _FakeAnnotation:
        return self

    def write_rttm(self, stream: Any) -> None:  # noqa: ANN401
        stream.write("SPEAKER fake 1 0.000 0.100 <NA> <NA> speaker_0 <NA> <NA>\n")

    def itertracks(self, *, yield_label: bool) -> Iterator[tuple[SimpleNamespace, None, str]]:
        assert yield_label
        if self._fail_after_rttm:
            msg = "failure after RTTM write"
            raise RuntimeError(msg)
        for segment in self._segments:
            yield SimpleNamespace(start=segment["start"], end=segment["end"]), None, segment["speaker"]


def _make_stage(  # noqa: PLR0913
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_residency: str = "file",
    fanout: bool = False,
    write_rttm: bool = False,
    rttm_out_dir: str | None = None,
    fail_after_rttm: bool = False,
) -> tuple[Any, list[int]]:
    seen_lengths: list[int] = []
    if kind == "pyannote":
        annotation = _FakeAnnotation(_SEGMENTS, fail_after_rttm=fail_after_rttm)

        def infer(payload: dict[str, Any], *, hook: object) -> _FakeAnnotation:
            assert hook is not None
            seen_lengths.append(int(payload["waveform"].shape[-1]))
            return annotation

        monkeypatch.setattr(pyannote_module, "ProgressHook", _ProgressHook)
        monkeypatch.setattr(pyannote_module, "add_non_speaker_segments", lambda *_args: None)
        stage = PyAnnoteDiarizationStage(
            audio_filepath_key="audio_filepath",
            input_residency=input_residency,
            fanout=fanout,
            write_rttm=write_rttm,
            min_length=0.0,
            resources=Resources(gpus=0),
            num_speakers_key="num_speakers",
        )
        stage._pipeline = infer
        return stage, seen_lengths

    if kind == "whisperx":
        model = MagicMock()

        def vad(audio: np.ndarray, _max_length: float, *, sample_rate: int) -> list[dict[str, Any]]:
            assert sample_rate == _SAMPLE_RATE
            seen_lengths.append(int(audio.shape[-1]))
            return [dict(segment) for segment in _SEGMENTS]

        model.get_vad_segments.side_effect = vad
        stage = WhisperXVADStage(
            audio_filepath_key="audio_filepath",
            input_residency=input_residency,
            fanout=fanout,
            min_length=0.0,
            resources=Resources(gpus=0),
        )
        stage._vad_model = model
        return stage, seen_lengths

    if kind == "sortformer":
        model = MagicMock()

        def diarize(*, audio: list[str], batch_size: int) -> list[list[str]]:
            assert batch_size == 1
            seen_lengths.append(sf.info(audio[0]).frames)
            return [[f"{segment['start']} {segment['end']} {segment['speaker']}" for segment in _SEGMENTS]]

        model.diarize.side_effect = diarize
        stage = InferenceSortformerStage(
            diar_model=model,
            input_residency=input_residency,
            fanout=fanout,
            rttm_out_dir=rttm_out_dir,
            resources=Resources(gpus=0),
        )
        return stage, seen_lengths

    raise AssertionError(kind)


def _write_audio(path: Path, waveform: np.ndarray) -> None:
    sf.write(path, waveform.T, _SAMPLE_RATE, subtype="FLOAT")
