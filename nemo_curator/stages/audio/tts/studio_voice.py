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

"""Optional NVIDIA Maxine StudioVoice speech enhancement (TTS Granary ``EnhancedAudioPipeline``).

The stage is a Triton gRPC client; the StudioVoice server runs outside Curator. The TTS Granary reference starts
one server per GPU rank inside the Maxine StudioVoice NIM container (``nvcr.io/nim/nvidia/maxine-studio-voice``)::

    export STREAMING=true MODEL_TYPE=48k-ll
    # link the cached TensorRT packages into the model repository, one per GPU architecture:
    #   <nim_cache>/ngc/hub/models--nim--nvidia--maxine-studio-voice/snapshots/*/sm_<arch>/*_1.trtpkg
    #     -> /config/models/maxine-studio-voice/*_sm<arch>.trtpkg
    tritonserver --model-repository=/opt/maxine/models --model-control-mode=explicit \\
        --load-model=studio_voice_low_latency-48k \\
        --http-address=127.0.0.1 --http-port=20000 \\
        --grpc-address=127.0.0.1 --grpc-port=20001 \\
        --metrics-address=127.0.0.1 --metrics-port=20002
    # ready when http://127.0.0.1:20000/v2/health/ready returns 200

and points this stage at ``server_url="127.0.0.1:20001"``. The low-latency 48 kHz model is streamed in 10 ms
chunks (480 samples) with one Triton sequence per segment.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from nemo_curator.stages.audio.tts.audio_io import (
    TAR_MEMBER_FIELD,
    SegmentAudioError,
    TarAudioConfig,
    iter_segments,
    load_segment_audio,
    sample_id,
)
from nemo_curator.stages.audio.tts.fields import get_dotted, set_dotted
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

STUDIO_VOICE_MODEL = "studio_voice_low_latency-48k"
STUDIO_VOICE_SAMPLE_RATE = 48000
STUDIO_VOICE_FAILED = "studio_voice_failed"
EMPTY_AUDIO = "empty_audio"

_FATAL_FRAGMENTS = (
    "REQUEST FOR UNKNOWN MODEL",
    "HAS NO AVAILABLE VERSIONS",
    "SERVER UNAVAILABLE",
    "DEADLINE EXCEEDED",
    "STATUSCODE.UNAVAILABLE",
    "STATUSCODE.DEADLINE_EXCEEDED",
    "STATUSCODE.INTERNAL",
    "STATUSCODE.NOT_FOUND",
    "STATUSCODE.RESOURCE_EXHAUSTED",
)


def is_fatal_runtime_error(exc: BaseException) -> bool:
    """Server-side failures that must abort the worker instead of being recorded per segment."""
    message = str(exc).upper()
    return any(fragment in message for fragment in _FATAL_FRAGMENTS)


def empty_annotation(error: str, detail: str | None = None) -> dict[str, Any]:
    annotation: dict[str, Any] = {
        "audio_filepath": None,
        "duration": None,
        "sampling_rate": None,
        "method": "studio_voice",
        "error": error,
    }
    if detail is not None:
        annotation["error_detail"] = detail
    return annotation


class StudioVoiceTritonClient:
    """gRPC client for the StudioVoice Triton model (port of the TTS Granary ``StudioVoiceTritonClient``)."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        target: str,
        model_name: str = STUDIO_VOICE_MODEL,
        sample_rate: int = STUDIO_VOICE_SAMPLE_RATE,
        timeout_sec: float = 120.0,
        chunk_samples: int | None = None,
        wait_ready_sec: float = 120.0,
        trim_to_input: bool = True,
        grpcclient: Any = None,  # noqa: ANN401
    ) -> None:
        if grpcclient is None:
            try:
                import tritonclient.grpc as grpcclient
            except ImportError as exc:
                msg = "StudioVoiceEnhancementStage requires 'tritonclient[grpc]' (pip install 'tritonclient[grpc]')"
                raise ImportError(msg) from exc
        if sample_rate != STUDIO_VOICE_SAMPLE_RATE:
            msg = f"Expected sample_rate={STUDIO_VOICE_SAMPLE_RATE} for 48k-LL, got {sample_rate}"
            raise ValueError(msg)
        self.grpcclient = grpcclient
        self.target = target
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.timeout_sec = timeout_sec
        self.chunk_samples = chunk_samples or sample_rate // 100
        self.trim_to_input = trim_to_input
        self.client = grpcclient.InferenceServerClient(url=target, verbose=False)
        self._wait_until_ready(wait_ready_sec)

    def _wait_until_ready(self, wait_ready_sec: float) -> None:
        deadline = time.monotonic() + wait_ready_sec
        last_error: BaseException | None = None
        while True:
            try:
                if self.client.is_server_ready() and self.client.is_model_ready(model_name=self.model_name):
                    logger.info("StudioVoice Triton model ready: {} at {}", self.model_name, self.target)
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(1.0)
        msg = f"Triton model did not become ready: {self.model_name} at {self.target}"
        if last_error is not None:
            raise RuntimeError(msg) from last_error
        raise RuntimeError(msg)

    def _infer_chunk(self, chunk: np.ndarray, sequence_id: int, sequence_start: bool) -> np.ndarray:
        input_data = np.asarray(chunk, dtype=np.float32).reshape(1, -1)
        infer_input = self.grpcclient.InferInput("input", input_data.shape, "FP32")
        infer_input.set_data_from_numpy(input_data)
        infer_output = self.grpcclient.InferRequestedOutput("output")
        result = self.client.infer(
            model_name=self.model_name,
            inputs=[infer_input],
            outputs=[infer_output],
            sequence_id=sequence_id,
            sequence_start=sequence_start,
            sequence_end=False,
            client_timeout=self.timeout_sec,
        )
        output = result.as_numpy("output")
        if output is None:
            msg = "StudioVoice Triton returned no output tensor"
            raise RuntimeError(msg)
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def enhance(self, samples: np.ndarray) -> np.ndarray:
        """Enhance a 48 kHz mono signal: zero-pad to whole 10 ms chunks, stream them, trim to input length."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        input_len = len(audio)
        if input_len == 0:
            msg = "Input audio is empty"
            raise ValueError(msg)
        pad = (self.chunk_samples - (len(audio) % self.chunk_samples)) % self.chunk_samples
        if pad:
            audio = np.pad(audio, (0, pad), mode="constant")
        sequence_id = random.randint(1_000_000, 999_999_999)  # noqa: S311
        chunks = [
            self._infer_chunk(audio[start : start + self.chunk_samples], sequence_id, chunk_idx == 0)
            for chunk_idx, start in enumerate(range(0, len(audio), self.chunk_samples))
        ]
        enhanced = np.concatenate(chunks).astype(np.float32, copy=False)
        return enhanced[:input_len] if self.trim_to_input else enhanced


def _safe_name(value: str) -> str:
    return value.replace(os.sep, "_").replace("/", "_").strip() or "unknown"


@dataclass
class StudioVoiceEnhancementStage(ProcessingStage[AudioTask, AudioTask]):
    """Enhance every segment with Maxine StudioVoice and record the enhanced audio location.

    Port of TTS Granary ``pipeline.stages.speech_enhancement.studio_voice``. Each segment is decoded, averaged to
    mono, resampled to 48 kHz, streamed through the Triton model and written as a 16-bit PCM WAV to
    ``output_dir/<sample_id>/<tar member stem>.wav``. Writes
    ``{"audio_filepath", "duration", "sampling_rate": 48000, "error"}`` at ``output_key``.

    Rows and segments are never dropped. Fatal server errors (unavailable, deadline exceeded, unknown model, ...)
    are re-raised so the worker fails instead of silently annotating everything as failed; other failures are
    recorded as ``studio_voice_failed`` with ``error_detail``. With ``skip_existing`` a segment whose output WAV
    already exists is not re-enhanced (resume after preemption).

    Unlike the reference (which packs the enhanced audio into a tar and stores byte offsets), enhanced audio is
    written as individual files; enhancement is optional and nothing downstream reads it by offset.
    """

    name: str = "StudioVoiceEnhancement"
    server_url: str = "127.0.0.1:20001"
    output_dir: str = "studio_voice"
    output_key: str = "enhanced_audio"
    model_name: str = STUDIO_VOICE_MODEL
    timeout_sec: float = 120.0
    wait_ready_sec: float = 180.0
    chunk_samples: int | None = None
    trim_to_input: bool = True
    skip_existing: bool = True
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    client: Any | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1

    def __post_init__(self) -> None:
        super().__init__()
        self._audio_config = TarAudioConfig(self.tar_root, tuple(self.strip_tar_key_prefixes))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key.split(".")[0]]

    def setup(self, _worker_metadata: Any = None) -> None:  # noqa: ANN401
        if self.client is not None:
            return
        self.client = StudioVoiceTritonClient(
            target=self.server_url,
            model_name=self.model_name,
            timeout_sec=self.timeout_sec,
            chunk_samples=self.chunk_samples,
            wait_ready_sec=self.wait_ready_sec,
            trim_to_input=self.trim_to_input,
        )

    def output_path(self, row: dict[str, Any], segment: dict[str, Any], index: int) -> Path:
        member = get_dotted(segment, TAR_MEMBER_FIELD) or segment.get("audio_filepath") or f"segment_{index:06d}"
        stem = Path(str(member)).stem or f"segment_{index:06d}"
        return Path(self.output_dir) / _safe_name(sample_id(row)) / f"{_safe_name(stem)}.wav"

    def enhance_segment(self, row: dict[str, Any], segment: dict[str, Any], index: int) -> dict[str, Any]:
        import soundfile as sf

        out_path = self.output_path(row, segment, index)
        if self.skip_existing and out_path.is_file():
            try:
                info = sf.info(str(out_path))
                return self._success(out_path, info.frames)
            except Exception as exc:  # noqa: BLE001  (e.g. truncated by preemption) -> re-enhance
                logger.warning("Re-enhancing unreadable existing output {}: {}", out_path, exc)
        try:
            audio, _ = load_segment_audio(segment, STUDIO_VOICE_SAMPLE_RATE, self._audio_config)
        except SegmentAudioError as exc:
            return empty_annotation(exc.code)
        if audio.size == 0:
            return empty_annotation(EMPTY_AUDIO)
        try:
            enhanced = self.client.enhance(audio)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_name(f".{out_path.name}.tmp")
            sf.write(str(tmp_path), np.asarray(enhanced, dtype=np.float32), STUDIO_VOICE_SAMPLE_RATE, format="WAV")
            tmp_path.replace(out_path)
        except Exception as exc:
            if is_fatal_runtime_error(exc):
                raise
            logger.warning("StudioVoice enhancement failed for {}: {}", out_path.name, exc)
            return empty_annotation(STUDIO_VOICE_FAILED, repr(exc))
        return self._success(out_path, len(enhanced))

    @staticmethod
    def _success(path: Path, num_samples: int) -> dict[str, Any]:
        return {
            "audio_filepath": str(path),
            "duration": float(num_samples) / STUDIO_VOICE_SAMPLE_RATE,
            "sampling_rate": STUDIO_VOICE_SAMPLE_RATE,
            "method": "studio_voice",
            "error": None,
        }

    def process(self, task: AudioTask) -> AudioTask:
        for index, segment in iter_segments(task.data):
            set_dotted(segment, self.output_key, self.enhance_segment(task.data, segment, index))
        return task
