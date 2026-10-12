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

"""Vocalization stage: synthesize every spoken reading with a pluggable TTS backend.

A backend is anything with ``synthesize(text, out_path, **kw) -> Path``. Select one with a config
dict ``{"backend": <name>, ...params}``:

- ``nvidia_tts``: an NVIDIA Riva TTS NIM (``POST {base_url}/v1/audio/synthesize``) with a named
  preset voice, for example Magpie TTS Multilingual.
- ``nvidia_tts_zeroshot``: the same NIM API with a zero-shot voice that clones the speaker of a
  3-10 s ``reference_audio`` clip (multipart field ``audio_prompt``).
  Both accept a ``route`` template with a ``{model}`` placeholder for NIMs served through a
  gateway under per-model paths, and send ``api_key_env`` as a bearer token when it is set.
- ``openai_speech``: any OpenAI-compatible ``/audio/speech`` endpoint.
- ``elevenlabs``: ElevenLabs text-to-speech (raw PCM wrapped into WAV locally).
- ``custom``: your own ``"module:function"`` with the signature above.

The stage resamples every clip to 16 kHz mono PCM16, the format ASR models expect.
"""

import importlib
import math
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.audio.numpipe.http_utils import (
    api_key,
    auth_header,
    build_url,
    post_with_retry,
    require_wav,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

TARGET_SAMPLE_RATE = 16000
ZERO_SHOT_MIN_S, ZERO_SHOT_MAX_S = 3.0, 10.0
NIM_SYNTHESIZE_ROUTE = "/v1/audio/synthesize"


class TtsBackend(Protocol):
    def synthesize(self, text: str, out_path: str | Path, **kw: object) -> Path: ...


def _write_wav(out_path: str | Path, content: bytes) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(content)
    return out


@dataclass
class NvidiaTts:
    """NVIDIA Riva TTS NIM, ``POST {base_url}/v1/audio/synthesize``, with a named preset voice.

    ``base_url`` is the NIM's HTTP address, for example ``http://localhost:9000``. A self-hosted NIM
    needs no key; when the ``api_key_env`` variable is set it is sent as a bearer token. Behind a
    gateway that exposes NIMs under per-model paths, set ``route`` to that path template (with
    ``{model}`` as the placeholder) and ``model`` to the gateway's name for the TTS model.
    """

    base_url: str
    voice: str = "Magpie-Multilingual.EN-US.Aria"
    language: str = "en-US"
    sample_rate_hz: int | None = None
    route: str = NIM_SYNTHESIZE_ROUTE
    model: str | None = None
    api_key_env: str = "NVIDIA_API_KEY"
    timeout: float = 90.0

    def synthesize(self, text: str, out_path: str | Path, *, voice: str | None = None, **_: object) -> Path:
        files: dict[str, tuple] = {
            "text": (None, text),
            "language": (None, self.language),
            "voice": (None, voice or self.voice),
            "encoding": (None, "LINEAR_PCM"),
        }
        if self.sample_rate_hz:
            files["sample_rate_hz"] = (None, str(self.sample_rate_hz))
        response = post_with_retry(
            build_url(self.base_url, self.route, self.model),
            headers={**auth_header(self.api_key_env), "Accept": "audio/wav"},
            timeout=self.timeout,
            files=files,
        )
        require_wav(response.content, "nvidia_tts")
        return _write_wav(out_path, response.content)


@dataclass
class NvidiaTtsZeroShot:
    """NVIDIA Riva TTS NIM with a zero-shot voice: clones the speaker of ``reference_audio``.

    The reference is a 16-bit mono WAV of 3-10 s, sent as the ``audio_prompt`` field. A per-row
    ``reference_audio`` (and optional ``reference_text``, its transcript) overrides the defaults.
    ``route`` and ``model`` work as in :class:`NvidiaTts`.
    """

    base_url: str
    voice: str = "Magpie-ZeroShot-Multilingual"
    language: str = "en-US"
    reference_audio: str | None = None
    reference_text: str | None = None
    route: str = NIM_SYNTHESIZE_ROUTE
    model: str | None = None
    api_key_env: str = "NVIDIA_API_KEY"
    timeout: float = 90.0

    def _reference(self, reference_audio: str | Path | None) -> tuple[str, bytes]:
        ref = reference_audio or self.reference_audio
        if not ref:
            msg = "nvidia_tts_zeroshot needs a reference_audio (3-10 s WAV), per row or in the backend config"
            raise ValueError(msg)
        ref = Path(ref)
        import soundfile

        duration = float(soundfile.info(str(ref)).duration)
        if not ZERO_SHOT_MIN_S <= duration <= ZERO_SHOT_MAX_S:
            msg = f"reference {ref.name} is {duration:.2f}s; zero-shot cloning needs {ZERO_SHOT_MIN_S}-{ZERO_SHOT_MAX_S}s"
            raise ValueError(msg)
        return ref.name, ref.read_bytes()

    def synthesize(
        self,
        text: str,
        out_path: str | Path,
        *,
        reference_audio: str | Path | None = None,
        reference_text: str | None = None,
        **_: object,
    ) -> Path:
        name, raw = self._reference(reference_audio)
        files: dict[str, tuple] = {
            "text": (None, text),
            "language": (None, self.language),
            "voice": (None, self.voice),
            "encoding": (None, "LINEAR_PCM"),
            "audio_prompt": (name, raw, "audio/wav"),
        }
        transcript = reference_text or self.reference_text
        if transcript:
            files["audio_prompt_transcript"] = (None, transcript)
        response = post_with_retry(
            build_url(self.base_url, self.route, self.model),
            headers={**auth_header(self.api_key_env), "Accept": "audio/wav"},
            timeout=self.timeout,
            files=files,
        )
        require_wav(response.content, "nvidia_tts_zeroshot")
        return _write_wav(out_path, response.content)


@dataclass
class OpenAiSpeech:
    """Any OpenAI-compatible ``/audio/speech`` endpoint."""

    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 90.0

    def synthesize(self, text: str, out_path: str | Path, *, voice: str | None = None, **_: object) -> Path:
        response = post_with_retry(
            f"{self.base_url.rstrip('/')}/audio/speech",
            headers={"Authorization": f"Bearer {api_key(self.api_key_env)}"},
            timeout=self.timeout,
            json={"model": self.model, "input": text, "voice": voice or self.voice, "response_format": "wav"},
        )
        require_wav(response.content, "openai_speech")
        return _write_wav(out_path, response.content)


@dataclass
class ElevenLabsTts:
    """ElevenLabs text-to-speech. Requests raw PCM and wraps it into a WAV container locally."""

    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    model: str = "eleven_multilingual_v2"
    base_url: str = "https://api.elevenlabs.io/v1"
    api_key_env: str = "ELEVENLABS_API_KEY"
    sample_rate: int = 24000
    timeout: float = 90.0

    def synthesize(self, text: str, out_path: str | Path, *, voice: str | None = None, **_: object) -> Path:
        key = api_key(self.api_key_env)
        response = post_with_retry(
            f"{self.base_url.rstrip('/')}/text-to-speech/{voice or self.voice_id}",
            headers={"xi-api-key": key},
            timeout=self.timeout,
            params={"output_format": f"pcm_{self.sample_rate}"},
            json={"text": text, "model_id": self.model},
        )
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(response.content)
        return out


@dataclass
class CustomTts:
    """User-supplied TTS: ``entrypoint="my_module:synthesize"`` with ``fn(text, out_path, **kw) -> path``."""

    entrypoint: str

    def __post_init__(self) -> None:
        module_name, _, fn_name = self.entrypoint.partition(":")
        if not fn_name:
            msg = f"custom TTS entrypoint must be 'module:function', got {self.entrypoint!r}"
            raise ValueError(msg)
        self._fn = getattr(importlib.import_module(module_name), fn_name)

    def synthesize(self, text: str, out_path: str | Path, **kw: object) -> Path:
        return Path(self._fn(text, str(out_path), **kw))


TTS_BACKENDS: dict[str, type] = {
    "nvidia_tts": NvidiaTts,
    "nvidia_tts_zeroshot": NvidiaTtsZeroShot,
    "openai_speech": OpenAiSpeech,
    "elevenlabs": ElevenLabsTts,
    "custom": CustomTts,
}


def make_tts_backend(config: dict[str, Any]) -> TtsBackend:
    """Build a TTS backend from ``{"backend": <name>, ...params}``."""
    cfg = dict(config)
    name = cfg.pop("backend", None)
    if name not in TTS_BACKENDS:
        msg = f"unknown TTS backend {name!r}; choices: {sorted(TTS_BACKENDS)}"
        raise ValueError(msg)
    return TTS_BACKENDS[name](**cfg)


def resample_to_16k_mono(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> float:
    """Rewrite ``path`` as ``target_sr`` mono PCM16 in place and return the new duration in seconds."""
    import numpy as np
    import soundfile
    from scipy.signal import resample_poly

    data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        g = math.gcd(target_sr, sr)
        mono = resample_poly(mono, target_sr // g, sr // g).astype("float32")
    mono = np.clip(mono, -1.0, 1.0)
    soundfile.write(str(path), mono, target_sr, subtype="PCM_16")
    return len(mono) / float(target_sr)


@dataclass
class TtsSynthesisStage(ProcessingStage[AudioTask, AudioTask]):
    """Synthesize ``text_key`` of every entry into ``wav_dir/<id>.wav`` and record the audio metadata.

    Writes ``audio_filepath``, ``duration``, ``sample_rate`` and ``tts_backend``. A row may carry
    its own ``voice`` (preset backends) or ``reference_audio`` and ``reference_text`` (zero-shot
    backends), which override the backend defaults. Existing clips are reused unless ``overwrite``
    is set, so an interrupted run can be resumed.

    Args:
        wav_dir: Output directory for the clips.
        backend: TTS backend config, see :func:`make_tts_backend`.
        text_key: Key holding the text to synthesize.
        filepath_key: Key that will hold the clip path.
        resample_16k: Convert clips to 16 kHz mono PCM16 (recommended for ASR).
        overwrite: Re-synthesize clips that already exist.
    """

    wav_dir: str
    backend: dict[str, Any] = field(default_factory=dict)
    text_key: str = "spoken_ref"
    filepath_key: str = "audio_filepath"
    resample_16k: bool = True
    overwrite: bool = False
    name: str = "tts_synthesis"

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._backend = make_tts_backend(self.backend)
        Path(self.wav_dir).mkdir(parents=True, exist_ok=True)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["id", self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key, "duration", "sample_rate", "tts_backend"]

    def process(self, task: AudioTask) -> AudioTask:
        import soundfile

        t0 = time.perf_counter()
        wav = Path(self.wav_dir) / f"{task.data['id']}.wav"
        if self.overwrite or not wav.exists():
            kw: dict[str, Any] = {}
            for key in ("voice", "reference_audio", "reference_text"):
                if task.data.get(key):
                    kw[key] = task.data[key]
            self._backend.synthesize(task.data[self.text_key], wav, **kw)
            if self.resample_16k:
                resample_to_16k_mono(wav)
        info = soundfile.info(str(wav))
        task.data[self.filepath_key] = str(wav.resolve())
        task.data["duration"] = float(info.duration)
        task.data["sample_rate"] = int(info.samplerate)
        task.data["tts_backend"] = self.backend.get("backend", "")
        self._log_metrics({"process_time": time.perf_counter() - t0, "audio_seconds": float(info.duration)})
        return task
