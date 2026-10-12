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

"""ASR stage: transcribe every clip with a pluggable backend (the model under test, or a judge voter).

A backend is anything with ``transcribe(wav_path) -> str``. Select one with a config dict
``{"backend": <name>, ...params}``:

- ``nvidia_asr``: an NVIDIA Riva ASR NIM (``POST {base_url}/v1/audio/transcriptions``), for
  example Parakeet; multipart ``file`` in, ``{"text": ...}`` out. Accepts a ``route`` template
  with a ``{model}`` placeholder for NIMs served through a gateway under per-model paths.
- ``openai_transcriptions``: any OpenAI-compatible ``/audio/transcriptions`` endpoint.
- ``llm``: any OpenAI-compatible chat model that accepts ``input_audio`` (audio in, transcript out).
- ``nemo``: any NeMo ASR model, a pretrained name or a local ``.nemo`` file (``nemo_toolkit[asr]``).
- ``custom``: your own ``"module:function"`` taking a WAV path and returning text.

For batched GPU inference with NeMo models, Curator's ASR inference stages can be used in place of
this stage; only ``pred_text`` has to be present for the scoring stage.
"""

import base64
import importlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.audio.numpipe.http_utils import api_key, auth_header, build_url, post_with_retry
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

NIM_TRANSCRIPTIONS_ROUTE = "/v1/audio/transcriptions"


class AsrBackend(Protocol):
    def transcribe(self, wav_path: str | Path) -> str: ...


def _wav_bytes(wav_path: str | Path) -> bytes:
    return Path(wav_path).read_bytes()


@dataclass
class NvidiaAsr:
    """NVIDIA Riva ASR NIM, ``POST {base_url}/v1/audio/transcriptions`` (for example Parakeet).

    ``base_url`` is the NIM's HTTP address, for example ``http://localhost:9000``. A self-hosted NIM
    needs no key; when the ``api_key_env`` variable is set it is sent as a bearer token.

    ``language`` is sent as a form field. ``model`` names the model when the NIM wants one: it goes
    into the path when ``route`` has a ``{model}`` placeholder (a gateway with per-model paths), or
    into the ``model`` form field otherwise.
    """

    base_url: str
    language: str = "en-US"
    route: str = NIM_TRANSCRIPTIONS_ROUTE
    model: str | None = None
    api_key_env: str = "NVIDIA_API_KEY"
    timeout: float = 90.0

    def transcribe(self, wav_path: str | Path) -> str:
        files: dict[str, tuple] = {
            "file": (Path(wav_path).name, _wav_bytes(wav_path), "audio/wav"),
            "language": (None, self.language),
        }
        if self.model and "{model}" not in self.route:
            files["model"] = (None, self.model)
        response = post_with_retry(
            build_url(self.base_url, self.route, self.model),
            headers=auth_header(self.api_key_env),
            timeout=self.timeout,
            files=files,
        )
        return (response.json().get("text") or "").strip()


@dataclass
class OpenAiTranscriptions:
    """Any OpenAI-compatible ``/audio/transcriptions`` endpoint (Whisper, a self-hosted server, ...)."""

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    language: str | None = None
    timeout: float = 120.0

    def transcribe(self, wav_path: str | Path) -> str:
        fields: dict[str, Any] = {
            "file": (Path(wav_path).name, _wav_bytes(wav_path), "audio/wav"),
            "model": (None, self.model),
        }
        if self.language:
            fields["language"] = (None, self.language)
        response = post_with_retry(
            f"{self.base_url.rstrip('/')}/audio/transcriptions",
            headers=auth_header(self.api_key_env),
            timeout=self.timeout,
            files=fields,
        )
        return (response.json().get("text") or "").strip()


@dataclass
class LlmAsr:
    """An audio-capable chat model used as a transcriber (OpenAI-compatible ``/chat/completions``).

    ``base_url`` is the chat endpoint (ending in ``/v1``) and ``model`` a model there that accepts
    ``input_audio`` content. The prompt asks for a verbatim transcript only; the model is never told
    what number to expect.
    """

    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    prompt: str = (
        "Transcribe the number spoken in this audio exactly as words, in order "
        "(digit-by-digit if that is how it is read). Output ONLY the transcript, nothing else."
    )
    max_tokens: int = 256
    timeout: float = 120.0

    def transcribe(self, wav_path: str | Path) -> str:
        audio_b64 = base64.b64encode(_wav_bytes(wav_path)).decode()
        body = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                    ],
                }
            ],
        }
        response = post_with_retry(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key(self.api_key_env)}"},
            timeout=self.timeout,
            json=body,
        )
        return (response.json()["choices"][0]["message"]["content"] or "").strip()


@dataclass
class CustomAsr:
    """User-supplied ASR: ``entrypoint="my_module:transcribe"`` with ``fn(wav_path) -> str``."""

    entrypoint: str

    def __post_init__(self) -> None:
        module_name, _, fn_name = self.entrypoint.partition(":")
        if not fn_name:
            msg = f"custom ASR entrypoint must be 'module:function', got {self.entrypoint!r}"
            raise ValueError(msg)
        self._fn = getattr(importlib.import_module(module_name), fn_name)

    def transcribe(self, wav_path: str | Path) -> str:
        return str(self._fn(str(wav_path))).strip()


@dataclass
class NemoAsr:
    """Any NeMo ASR model: a pretrained name (``from_pretrained``) or a local ``.nemo`` file.

    Requires ``nemo_toolkit[asr]``. The model is loaded once per worker, on the first call.
    """

    model: str
    device: str = "cuda"
    batch_size: int = 8
    _asr: Any = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._asr is not None:
            return
        import nemo.collections.asr as nemo_asr

        if self.model.endswith(".nemo") or os.path.exists(self.model):
            self._asr = nemo_asr.models.ASRModel.restore_from(self.model, map_location=self.device)
        else:
            self._asr = nemo_asr.models.ASRModel.from_pretrained(self.model, map_location=self.device)

    def transcribe(self, wav_path: str | Path) -> str:
        self._ensure_loaded()
        out = self._asr.transcribe([str(wav_path)], batch_size=self.batch_size)
        if isinstance(out, tuple):  # (hypotheses, all_hypotheses) in some NeMo versions
            out = out[0]
        item = out[0]
        if hasattr(item, "text"):  # Hypothesis object
            return item.text
        if isinstance(item, list):  # n-best list
            return item[0].text if hasattr(item[0], "text") else str(item[0])
        return str(item)


ASR_BACKENDS: dict[str, type] = {
    "nvidia_asr": NvidiaAsr,
    "openai_transcriptions": OpenAiTranscriptions,
    "llm": LlmAsr,
    "nemo": NemoAsr,
    "custom": CustomAsr,
}


def make_asr_backend(config: dict[str, Any]) -> AsrBackend:
    """Build an ASR backend from ``{"backend": <name>, ...params}``."""
    cfg = dict(config)
    name = cfg.pop("backend", None)
    if name not in ASR_BACKENDS:
        msg = f"unknown ASR backend {name!r}; choices: {sorted(ASR_BACKENDS)}"
        raise ValueError(msg)
    return ASR_BACKENDS[name](**cfg)


@dataclass
class AsrStage(ProcessingStage[AudioTask, AudioTask]):
    """Transcribe ``filepath_key`` with the configured backend and store the text in ``pred_text_key``.

    Give the stage GPU resources (``resources: {gpus: 1}`` in YAML) when the backend is a local
    NeMo model; the hosted backends are CPU stages that Curator fans out across workers.
    """

    backend: dict[str, Any] = field(default_factory=dict)
    filepath_key: str = "audio_filepath"
    pred_text_key: str = "pred_text"
    name: str = "asr"

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._backend = make_asr_backend(self.backend)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key]

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        task.data[self.pred_text_key] = self._backend.transcribe(task.data[self.filepath_key])
        self._log_metrics({"process_time": time.perf_counter() - t0})
        return task
