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

"""Batched recurrent Silero VAD inference through TensorRT."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .base import VADResult
from .silero import (
    SILERO_TARGET_SAMPLE_RATE,
    _prepare_waveform,
    _timestamps_to_result,
    _validate_detection_options,
    _validate_num_gpus,
)

if TYPE_CHECKING:
    import torch


_WINDOW_SIZE = 512
_CONTEXT_SIZE = 64
_STATE_SIZE = 128
_INPUT_NAME = "input"
_STATE_INPUT_NAME = "state"
_OUTPUT_NAME = "output"
_STATE_OUTPUT_NAME = "stateN"
_REQUIRED_INPUTS = {_INPUT_NAME, _STATE_INPUT_NAME}
_REQUIRED_OUTPUTS = {_OUTPUT_NAME, _STATE_OUTPUT_NAME}
_MODEL_INPUT_DIMENSIONS = 2
_STATE_INPUT_DIMENSIONS = 3


class _TensorRTSession(Protocol):
    device: torch.device
    input_names: list[str]
    output_names: list[str]

    def input_shape_range(self, name: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    def close(self) -> None: ...


class _PrecomputedProbabilityModel:
    """Present precomputed recurrent probabilities to Silero postprocessing."""

    def __init__(self, probabilities: torch.Tensor) -> None:
        self.probabilities = probabilities
        self.index = 0

    def reset_states(self) -> None:
        self.index = 0

    def __call__(self, _audio: torch.Tensor, _sampling_rate: int) -> torch.Tensor:
        if self.index >= len(self.probabilities):
            msg = "Silero timestamp postprocessing requested more frames than TensorRT produced"
            raise RuntimeError(msg)
        probability = self.probabilities[self.index]
        self.index += 1
        return probability.reshape(1, 1)


def _prepare_16khz_waveform(item: dict[str, Any]) -> torch.Tensor:
    """Reuse Silero input validation, then enforce the TensorRT engine rate."""
    waveform, sample_rate = _prepare_waveform(item, device="cpu")
    if sample_rate != SILERO_TARGET_SAMPLE_RATE:
        import torchaudio

        waveform = (
            torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=SILERO_TARGET_SAMPLE_RATE,
            )(waveform.unsqueeze(0))
            .squeeze(0)
            .contiguous()
        )
    return waveform


@dataclass
class TensorRTSileroVADAdapter:
    """Persistent 16 kHz Silero TensorRT adapter with ragged batching.

    ``session`` is an injection seam for unit tests.  Production callers leave
    it unset so ``load_model`` constructs the shared
    :class:`TensorRTEncoderSession` from ``engine_path``.
    """

    engine_path: str
    threshold: float = 0.5
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0
    min_interval_ms: int = 500
    speech_pad_ms: int = 300
    session: InitVar[_TensorRTSession | None] = None

    _session: _TensorRTSession | None = field(default=None, init=False, repr=False)
    _max_batch_size: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self, session: _TensorRTSession | None) -> None:
        if not str(self.engine_path).strip():
            msg = "TensorRTSileroVADAdapter.engine_path must be non-empty"
            raise ValueError(msg)
        _validate_detection_options(
            threshold=self.threshold,
            min_duration_sec=self.min_duration_sec,
            max_duration_sec=self.max_duration_sec,
            min_interval_ms=self.min_interval_ms,
            speech_pad_ms=self.speech_pad_ms,
            owner=type(self).__name__,
        )
        self._session = session

    def download_weights_on_node(self) -> None:
        """Validate the caller-provided target-specific engine path."""
        path = Path(self.engine_path)
        if not path.is_file():
            msg = f"Silero TensorRT engine not found: {path}"
            raise FileNotFoundError(msg)

    @staticmethod
    def _validate_session(session: _TensorRTSession) -> int:
        inputs = set(getattr(session, "input_names", ()))
        outputs = set(getattr(session, "output_names", ()))
        if inputs != _REQUIRED_INPUTS:
            msg = f"Silero TensorRT engine inputs must be {sorted(_REQUIRED_INPUTS)}, got {sorted(inputs)}"
            raise ValueError(msg)
        if outputs != _REQUIRED_OUTPUTS:
            msg = f"Silero TensorRT engine outputs must be {sorted(_REQUIRED_OUTPUTS)}, got {sorted(outputs)}"
            raise ValueError(msg)
        if getattr(session, "device", None) is None:
            msg = "Silero TensorRT session must expose its torch device"
            raise ValueError(msg)
        input_minimum, input_optimum, input_maximum = session.input_shape_range(_INPUT_NAME)
        state_minimum, state_optimum, state_maximum = session.input_shape_range(_STATE_INPUT_NAME)
        if any(len(shape) != _MODEL_INPUT_DIMENSIONS for shape in (input_minimum, input_optimum, input_maximum)):
            msg = "Silero TensorRT input profile must have rank 2"
            raise ValueError(msg)
        if any(len(shape) != _STATE_INPUT_DIMENSIONS for shape in (state_minimum, state_optimum, state_maximum)):
            msg = "Silero TensorRT state profile must have rank 3"
            raise ValueError(msg)
        if any(
            shape[1:] != (_CONTEXT_SIZE + _WINDOW_SIZE,) for shape in (input_minimum, input_optimum, input_maximum)
        ):
            msg = f"Silero TensorRT input profile has incompatible shapes {input_minimum}..{input_maximum}"
            raise ValueError(msg)
        if any(shape[1:] != (2, _STATE_SIZE) for shape in (state_minimum, state_optimum, state_maximum)):
            msg = f"Silero TensorRT state profile has incompatible shapes {state_minimum}..{state_maximum}"
            raise ValueError(msg)
        input_batches = (input_minimum[0], input_optimum[0], input_maximum[0])
        state_batches = (state_minimum[0], state_optimum[0], state_maximum[0])
        if input_batches != state_batches or not input_batches[0] <= input_batches[1] <= input_batches[2]:
            msg = (
                "Silero TensorRT input and state profiles must expose identical ordered batch ranges, "
                f"got {input_batches} and {state_batches}"
            )
            raise ValueError(msg)
        if input_batches[0] != 1:
            msg = "Silero TensorRT profiles must support batch size 1 for ragged tail compaction"
            raise ValueError(msg)
        return int(input_maximum[0])

    def load_model(self, *, num_gpus: int) -> None:
        """Load one persistent TensorRT session on the assigned GPU."""
        gpu_count = _validate_num_gpus(num_gpus, owner=type(self).__name__, allow_gpu=True)
        if gpu_count != 1:
            msg = f"{type(self).__name__}.load_model requires exactly one GPU, got {gpu_count}"
            raise ValueError(msg)
        if self._session is not None:
            self._max_batch_size = self._validate_session(self._session)
            return

        from nemo_curator.stages.audio.inference.tensorrt_encoder import TensorRTEncoderSession

        session = TensorRTEncoderSession(self.engine_path)
        try:
            max_batch_size = self._validate_session(session)
        except Exception:
            session.close()
            raise
        self._session = session
        self._max_batch_size = max_batch_size

    def unload_model(self) -> None:
        """Close the persistent execution context and release its buffers."""
        if self._session is not None:
            self._session.close()
            self._session = None
        self._max_batch_size = None

    def infer_step(
        self,
        model_input: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one 576-sample recurrent step for an active row batch."""
        if self._session is None or self._max_batch_size is None:
            msg = "Silero TensorRT session is not loaded"
            raise RuntimeError(msg)
        expected_input = _CONTEXT_SIZE + _WINDOW_SIZE
        if model_input.ndim != _MODEL_INPUT_DIMENSIONS or model_input.shape[1] != expected_input:
            msg = f"Silero TensorRT input must have shape [batch, {expected_input}], got {tuple(model_input.shape)}"
            raise ValueError(msg)
        expected_state = (model_input.shape[0], 2, _STATE_SIZE)
        if tuple(state.shape) != expected_state:
            msg = f"Silero TensorRT state must have shape {expected_state}, got {tuple(state.shape)}"
            raise ValueError(msg)
        if model_input.shape[0] > self._max_batch_size:
            msg = (
                "Silero TensorRT batch exceeds the loaded engine profile: "
                f"{model_input.shape[0]} > {self._max_batch_size}"
            )
            raise ValueError(msg)

        outputs = self._session.infer(
            {
                _INPUT_NAME: model_input.contiguous(),
                _STATE_INPUT_NAME: state.contiguous(),
            }
        )
        probability = outputs[_OUTPUT_NAME]
        next_state = outputs[_STATE_OUTPUT_NAME]
        if probability.numel() != model_input.shape[0]:
            msg = f"Silero TensorRT output must contain one probability per row, got {tuple(probability.shape)}"
            raise RuntimeError(msg)
        if tuple(next_state.shape) != expected_state:
            msg = f"Silero TensorRT next state must have shape {expected_state}, got {tuple(next_state.shape)}"
            raise RuntimeError(msg)
        return probability.reshape(model_input.shape[0], 1), next_state

    def _infer_probabilities(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """Advance independent recordings in lockstep and compact completed rows."""
        if not waveforms:
            return []
        if self._session is None:
            msg = "Silero TensorRT session is not loaded; call load_model() before detect_batch()"
            raise RuntimeError(msg)

        import torch

        device = self._session.device
        prepared = [waveform.reshape(-1).to(device="cpu", dtype=torch.float32) for waveform in waveforms]
        steps = [(int(waveform.numel()) + _WINDOW_SIZE - 1) // _WINDOW_SIZE for waveform in prepared]
        max_steps = max(steps, default=0)
        if max_steps == 0:
            return [torch.empty(0, dtype=torch.float32) for _ in prepared]

        row_count = len(prepared)
        state = torch.zeros((row_count, 2, _STATE_SIZE), dtype=torch.float32, device=device)
        context = torch.zeros((row_count, _CONTEXT_SIZE), dtype=torch.float32, device=device)
        probabilities = torch.zeros((row_count, max_steps), dtype=torch.float32, device=device)

        for step in range(max_steps):
            active = [index for index, count in enumerate(steps) if step < count]
            start = step * _WINDOW_SIZE
            for batch_start in range(0, len(active), self._max_batch_size):
                batch_active = active[batch_start : batch_start + self._max_batch_size]
                active_index = torch.tensor(batch_active, dtype=torch.int64, device=device)
                chunks = []
                for index in batch_active:
                    chunk = prepared[index][start : start + _WINDOW_SIZE]
                    if chunk.numel() < _WINDOW_SIZE:
                        chunk = torch.nn.functional.pad(chunk, (0, _WINDOW_SIZE - chunk.numel()))
                    chunks.append(chunk)
                audio = torch.stack(chunks, dim=0).to(device=device, non_blocking=True)
                active_context = context.index_select(0, active_index)
                active_state = state.index_select(0, active_index)
                output, next_state = self.infer_step(
                    torch.cat((active_context, audio), dim=1),
                    active_state,
                )
                probabilities[active_index, step] = output[:, 0]
                state.index_copy_(0, active_index, next_state)
                context.index_copy_(0, active_index, audio[:, -_CONTEXT_SIZE:])

        probabilities_cpu = probabilities.cpu()
        return [probabilities_cpu[index, :count].clone() for index, count in enumerate(steps)]

    def detect_batch(self, items: list[dict[str, Any]]) -> list[VADResult]:
        """Detect speech for a ragged source-rate batch in input order."""
        if not items:
            return []
        if self._session is None:
            msg = "Silero TensorRT session is not loaded; call load_model() before detect_batch()"
            raise RuntimeError(msg)

        import torch
        from silero_vad import get_speech_timestamps

        waveforms = [_prepare_16khz_waveform(item) for item in items]
        probabilities = self._infer_probabilities(waveforms)

        results = []
        for waveform, recording_probabilities in zip(waveforms, probabilities, strict=True):
            try:
                timestamps = get_speech_timestamps(
                    waveform,
                    _PrecomputedProbabilityModel(recording_probabilities),
                    sampling_rate=SILERO_TARGET_SAMPLE_RATE,
                    threshold=self.threshold,
                    min_speech_duration_ms=self.min_duration_sec * 1000,
                    max_speech_duration_s=self.max_duration_sec,
                    min_silence_duration_ms=self.min_interval_ms,
                    speech_pad_ms=self.speech_pad_ms,
                )
                results.append(_timestamps_to_result(timestamps, SILERO_TARGET_SAMPLE_RATE))
            except (MemoryError, torch.cuda.OutOfMemoryError):
                raise
            except Exception as exc:  # noqa: BLE001
                results.append(VADResult(segments=[], error=f"{type(exc).__name__}: {exc}"))
        return results
