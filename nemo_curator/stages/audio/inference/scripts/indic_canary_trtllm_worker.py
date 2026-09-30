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

"""Standalone process host for the isolated Indic Canary TRT-LLM runtime.

This script deliberately imports no :mod:`nemo_curator` module. It is executed
by the separately locked Python 3.12 runtime and loads the vendored model code
from the absolute path supplied by its trusted parent process.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.util
import json
import os
import signal
import sys
import traceback
from multiprocessing.connection import Connection, Listener
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

_PROTOCOL_VERSION = 1
_AUTHKEY_BYTES = 32
_MAX_JSON_FRAME_BYTES = 1 << 20
_MAX_JSON_MESSAGE_BYTES = 64 << 20
_CHUNKED_JSON_OPERATION = "chunked_json"
_MAX_BATCH_ITEMS = 4096
_TARGET_SAMPLE_RATE = 16_000
_MIN_DURATION_SAMPLES = 400
_MAX_DURATION_SEC = 40.0
_PR_SET_PDEATHSIG = 1


class _CanaryModel(Protocol):
    tokenizer: object
    max_batch_size: object

    def process_batch(
        self,
        audio: list[object],
        audio_input_lengths: list[int],
        prompts_cfg: list[dict[str, object]],
        *,
        num_beams: int,
        max_new_tokens: int,
    ) -> list[str]: ...


class _Runtime(Protocol):
    def pad_or_trim(self, waveform: object, length: int) -> object: ...


def _read_exact(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(fd, remaining)
        if not chunk:
            msg = f"secret pipe closed after {size - remaining}/{size} bytes"
            raise RuntimeError(msg)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_status(fd: int, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    os.write(fd, encoded)


def _arm_parent_death_signal(expected_parent_pid: int) -> None:
    """Ask Linux to terminate this worker if its Ray actor parent disappears."""
    if sys.platform != "linux":
        msg = "Indic Canary TRT-LLM worker is supported only on Linux"
        raise RuntimeError(msg)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    if os.getppid() != expected_parent_pid:
        msg = "Indic Canary parent exited before the worker initialized"
        raise RuntimeError(msg)


def _load_runtime(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_nemo_curator_indic_canary_runtime", path)
    if spec is None or spec.loader is None:
        msg = f"cannot load Indic Canary runtime from {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _receive_json(connection: Connection) -> dict[str, Any]:
    raw = connection.recv_bytes(_MAX_JSON_FRAME_BYTES)
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        msg = "protocol frame must decode to a JSON object"
        raise TypeError(msg)
    return decoded


def _send_json(connection: Connection, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) <= _MAX_JSON_FRAME_BYTES:
        connection.send_bytes(encoded)
        return
    if len(encoded) > _MAX_JSON_MESSAGE_BYTES:
        msg = f"JSON response exceeds the {_MAX_JSON_MESSAGE_BYTES}-byte protocol limit"
        raise ValueError(msg)

    chunk_count = (len(encoded) + _MAX_JSON_FRAME_BYTES - 1) // _MAX_JSON_FRAME_BYTES
    connection.send_bytes(
        json.dumps(
            {
                "protocol": _PROTOCOL_VERSION,
                "op": _CHUNKED_JSON_OPERATION,
                "byte_count": len(encoded),
                "chunk_count": chunk_count,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for start in range(0, len(encoded), _MAX_JSON_FRAME_BYTES):
        connection.send_bytes(encoded[start : start + _MAX_JSON_FRAME_BYTES])


def _normalize_language(tokenizer: object, language: str) -> str | None:
    candidates = [language]
    if "-" in language:
        candidates.append(language.split("-", maxsplit=1)[0])
    supported_languages = set(getattr(tokenizer, "langs", ()))
    supports_prompt_language = getattr(tokenizer, "supports_prompt_language", None)
    for candidate in candidates:
        if (
            callable(supports_prompt_language) and supports_prompt_language(candidate)
        ) or candidate in supported_languages:
            return candidate
    return None


def _prompt_config(language: str, *, pnc: bool) -> dict[str, object]:
    return {
        "task": "transcribe",
        "pnc": pnc,
        "source_language": language,
        "target_language": language,
        "itn": False,
        "romanized": False,
        "timestamp": False,
        "diarize": False,
    }


def _normalize_languages(model: _CanaryModel, request: dict[str, Any]) -> list[str | None]:
    languages = request.get("languages")
    if not isinstance(languages, list) or len(languages) > _MAX_BATCH_ITEMS:
        msg = f"languages must be a list with at most {_MAX_BATCH_ITEMS} rows"
        raise TypeError(msg)
    if not all(isinstance(language, str) for language in languages):
        msg = "every language must be a string"
        raise TypeError(msg)
    return [_normalize_language(model.tokenizer, language) for language in languages]


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return int(value)


def _positive_duration(value: object, *, name: str) -> float:
    duration = float(value)  # type: ignore[arg-type]
    if not 0 < duration <= _MAX_DURATION_SEC:
        msg = f"{name} must be in (0, {_MAX_DURATION_SEC}]"
        raise ValueError(msg)
    return duration


def _infer(  # noqa: C901, PLR0912, PLR0915
    model: _CanaryModel,
    runtime: _Runtime,
    request: dict[str, Any],
    connection: Connection,
) -> list[dict[str, Any]]:
    """Receive one complete batch, run it, and preserve input row order."""
    import numpy as np
    import torch

    items = request.get("items")
    if not isinstance(items, list) or len(items) > _MAX_BATCH_ITEMS:
        msg = f"infer.items must be a list with at most {_MAX_BATCH_ITEMS} rows"
        raise TypeError(msg)
    max_duration_sec = _positive_duration(request.get("max_duration_sec"), name="max_duration_sec")
    min_duration_sec = _positive_duration(request.get("min_duration_sec"), name="min_duration_sec")
    min_duration_sec = min(min_duration_sec, max_duration_sec)
    max_samples = int(max_duration_sec * _TARGET_SAMPLE_RATE)
    min_samples = int(min_duration_sec * _TARGET_SAMPLE_RATE)
    num_beams = _positive_int(request.get("num_beams"), name="num_beams")
    max_new_tokens = _positive_int(request.get("max_new_tokens"), name="max_new_tokens")
    pnc = request.get("pnc")
    if not isinstance(pnc, bool):
        msg = "pnc must be a boolean"
        raise TypeError(msg)

    # Consume and validate every frame before model work. This keeps request
    # framing deterministic; any later failure is still terminal by design.
    received: list[tuple[dict[str, Any], bytes]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            msg = f"item {index} must be a JSON object"
            raise TypeError(msg)
        nbytes = int(item.get("nbytes", -1))
        sample_count = int(item.get("sample_count", -1))
        original_sample_count = int(item.get("original_sample_count", -1))
        if (
            sample_count < 0
            or sample_count > max_samples
            or nbytes != sample_count * np.dtype("<f4").itemsize
            or original_sample_count < sample_count
        ):
            msg = f"item {index} has invalid float32 payload dimensions"
            raise ValueError(msg)
        payload = connection.recv_bytes(maxlength=max_samples * np.dtype("<f4").itemsize)
        if len(payload) != nbytes:
            msg = f"item {index} declared {nbytes} waveform bytes but sent {len(payload)}"
            raise ValueError(msg)
        if int(item.get("sample_rate") or 0) != _TARGET_SAMPLE_RATE:
            msg = f"item {index} must be {_TARGET_SAMPLE_RATE} Hz"
            raise ValueError(msg)
        received.append((item, payload))

    results: list[dict[str, Any]] = [{"text": ""} for _ in items]
    prepared: list[Any] = []
    durations: list[int] = []
    normalized_languages: list[str] = []
    valid_indices: list[int] = []
    truncated: list[bool] = []
    audio_durations: list[float] = []

    for index, (item, payload) in enumerate(received):
        language = str(item.get("language_code") or "").strip().lower()
        normalized_language = _normalize_language(model.tokenizer, language)
        if normalized_language is None:
            results[index] = {
                "text": "",
                "skipped": True,
                "skip_reason": "language_not_supported",
                "unsupported_language": language or None,
                "extras": {"language_code": language, "language_unsupported": True},
            }
            continue

        waveform = np.frombuffer(payload, dtype="<f4").copy()
        prepared.append(torch.from_numpy(waveform))
        sample_count = int(item["sample_count"])
        original_sample_count = int(item["original_sample_count"])
        durations.append(min(max(sample_count, _MIN_DURATION_SAMPLES), max_samples))
        normalized_languages.append(normalized_language)
        valid_indices.append(index)
        truncated.append(original_sample_count > max_samples)
        audio_durations.append(float(original_sample_count) / _TARGET_SAMPLE_RATE)

    if not prepared:
        return results

    pad_length = max([min_samples, *[int(waveform.shape[0]) for waveform in prepared]])
    padded = [runtime.pad_or_trim(waveform, pad_length) for waveform in prepared]
    bounded_durations = [min(duration, pad_length) for duration in durations]
    prompts = [_prompt_config(language, pnc=pnc) for language in normalized_languages]
    runtime_batch_size = getattr(model, "max_batch_size", len(padded))
    if isinstance(runtime_batch_size, bool) or not isinstance(runtime_batch_size, Integral) or runtime_batch_size < 1:
        msg = f"Indic Canary runtime reported an invalid max_batch_size: {runtime_batch_size!r}"
        raise RuntimeError(msg)

    predictions: list[str] = []
    for start in range(0, len(padded), int(runtime_batch_size)):
        stop = start + int(runtime_batch_size)
        batch_predictions = model.process_batch(
            padded[start:stop],
            bounded_durations[start:stop],
            prompts[start:stop],
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        expected_count = len(padded[start:stop])
        if len(batch_predictions) != expected_count:
            msg = f"Indic Canary returned {len(batch_predictions)} transcriptions for {expected_count} inputs"
            raise RuntimeError(msg)
        predictions.extend(batch_predictions)

    for valid_position, prediction in enumerate(predictions):
        item_index = valid_indices[valid_position]
        results[item_index] = {
            "text": str(prediction),
            "skipped": False,
            "skip_reason": None,
            "unsupported_language": None,
            "extras": {
                "language_code": normalized_languages[valid_position],
                "truncated": truncated[valid_position],
                "audio_duration_sec": audio_durations[valid_position],
            },
        }
    return results


def _request_operation(request: dict[str, Any]) -> str:
    if request.get("protocol") != _PROTOCOL_VERSION:
        msg = "protocol version mismatch"
        raise ValueError(msg)
    operation = request.get("op")
    if operation not in {"infer", "normalize_languages", "shutdown"}:
        msg = f"unknown operation {operation!r}"
        raise ValueError(msg)
    return str(operation)


def _serve(args: argparse.Namespace) -> int:  # noqa: PLR0915
    listener: Listener | None = None
    connection: Connection | None = None
    model: Any = None
    try:
        _arm_parent_death_signal(args.parent_pid)
        authkey = _read_exact(args.auth_fd, _AUTHKEY_BYTES)
        os.close(args.auth_fd)
        listener = Listener(args.socket, family="AF_UNIX", authkey=authkey)
        os.chmod(args.socket, 0o600)
        runtime = _load_runtime(Path(args.runtime))
        model = runtime.CanaryTRTLLM(
            args.engine_dir,
            device="cuda:0",
            kv_cache_free_gpu_memory_fraction=args.kv_cache_fraction,
            cross_kv_cache_fraction=args.cross_kv_fraction,
        )
    except BaseException as exc:  # noqa: BLE001 - status must report any startup failure
        try:
            _write_status(
                args.status_fd,
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
            )
        finally:
            os.close(args.status_fd)
            if listener is not None:
                listener.close()
        traceback.print_exc()
        return 1

    _write_status(args.status_fd, {"ok": True, "pid": os.getpid(), "python": sys.version.split()[0]})
    os.close(args.status_fd)
    try:
        connection = listener.accept()
        _send_json(
            connection,
            {
                "protocol": _PROTOCOL_VERSION,
                "ok": True,
                "op": "ready",
                "pid": os.getpid(),
                "python": sys.version.split()[0],
                "max_batch_size": int(model.max_batch_size),
            },
        )
        while True:
            try:
                request = _receive_json(connection)
            except EOFError:
                break
            request_id = str(request.get("request_id") or "")
            try:
                operation = _request_operation(request)
                if operation == "shutdown":
                    _send_json(
                        connection,
                        {"protocol": _PROTOCOL_VERSION, "ok": True, "request_id": request_id},
                    )
                    break
                if operation == "normalize_languages":
                    _send_json(
                        connection,
                        {
                            "protocol": _PROTOCOL_VERSION,
                            "ok": True,
                            "request_id": request_id,
                            "normalized_languages": _normalize_languages(model, request),
                        },
                    )
                    continue
                results = _infer(model, runtime, request, connection)
                _send_json(
                    connection,
                    {
                        "protocol": _PROTOCOL_VERSION,
                        "ok": True,
                        "request_id": request_id,
                        "results": results,
                    },
                )
            except BaseException as exc:  # noqa: BLE001 - request failures poison the worker
                _send_json(
                    connection,
                    {
                        "protocol": _PROTOCOL_VERSION,
                        "ok": False,
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                traceback.print_exc()
                break
    finally:
        if connection is not None:
            connection.close()
        listener.close()
        del model
        gc.collect()
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--auth-fd", required=True, type=int)
    parser.add_argument("--status-fd", required=True, type=int)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--kv-cache-fraction", type=float, default=0.2)
    parser.add_argument("--cross-kv-fraction", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return _serve(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
