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

"""Adapter for an Indic Canary model exported as a TensorRT-LLM engine."""

from __future__ import annotations

import json
import math
import os
import queue
import secrets
import select
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from multiprocessing.connection import Client, Connection
from numbers import Integral
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from loguru import logger

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio.inference.indic_canary_runtime_env import (
    ensure_runtime_python,
    resolve_runtime_python,
    runtime_subprocess_environment,
)

_TARGET_SAMPLE_RATE = 16_000
_DEFAULT_MAX_DURATION_SEC = 40.0
_DEFAULT_MIN_DURATION_SEC = 0.5
_DEFAULT_RUNTIME_STARTUP_TIMEOUT_SEC = 600.0
_PROTOCOL_VERSION = 1
_AUTHKEY_BYTES = 32
_MAX_JSON_FRAME_BYTES = 1 << 20
_MAX_JSON_MESSAGE_BYTES = 64 << 20
_CHUNKED_JSON_OPERATION = "chunked_json"
_PROCESS_SHUTDOWN_TIMEOUT_SEC = 10.0
_CONTROL_MESSAGE_TIMEOUT_SEC = 10.0
_REQUIRED_ENGINE_FILES = (
    "encoder/encoder.plan",
    "encoder/config.json",
    "decoder/config.json",
    "decoder/rank0.engine",
    "decoder/vocab.json",
    "preprocessor/config.json",
    "preprocessor/mel_basis.pt",
)
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_WORKER_SCRIPT = _PACKAGE_ROOT / "stages/audio/inference/scripts/indic_canary_trtllm_worker.py"
_RUNTIME_MODULE = _PACKAGE_ROOT / "stages/audio/inference/indic_canary_trtllm_runtime.py"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    """Encode one deterministic, pickle-free protocol frame."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _decode_json_object(payload: bytes) -> dict[str, Any]:
    """Decode one protocol frame and require an object at its root."""
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        msg = "Indic Canary worker protocol frame must be a JSON object"
        raise TypeError(msg)
    return decoded


class _CanaryWorkerClient:
    """Own one authenticated, persistent TRT-LLM subprocess."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        runtime_python: Path,
        environment: dict[str, str],
        engine_dir: Path,
        kv_cache_free_gpu_memory_fraction: float,
        cross_kv_cache_fraction: float,
        startup_timeout_sec: float,
    ) -> None:
        self._connection: Connection | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._terminal_error: str | None = None
        self._lock = threading.Lock()
        # AF_UNIX paths are limited to roughly 108 bytes on Linux; anchor the
        # private 0700 directory under /tmp instead of an arbitrarily long TMPDIR.
        self._tempdir = tempfile.TemporaryDirectory(prefix="nc-canary-", dir="/tmp")
        self._socket_path = Path(self._tempdir.name) / "worker.sock"
        self._log_path = Path(self._tempdir.name) / "worker.log"
        self._start(
            runtime_python=runtime_python,
            environment=environment,
            engine_dir=engine_dir,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
            cross_kv_cache_fraction=cross_kv_cache_fraction,
            startup_timeout_sec=startup_timeout_sec,
        )

    def _start(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        *,
        runtime_python: Path,
        environment: dict[str, str],
        engine_dir: Path,
        kv_cache_free_gpu_memory_fraction: float,
        cross_kv_cache_fraction: float,
        startup_timeout_sec: float,
    ) -> None:
        if not _WORKER_SCRIPT.is_file() or not _RUNTIME_MODULE.is_file():
            msg = f"Indic Canary package is missing its worker or runtime module: {_WORKER_SCRIPT}, {_RUNTIME_MODULE}"
            self._tempdir.cleanup()
            raise FileNotFoundError(msg)

        auth_read, auth_write = os.pipe()
        status_read, status_write = os.pipe()
        authkey = secrets.token_bytes(_AUTHKEY_BYTES)
        argv = [
            str(runtime_python),
            "-I",
            "-u",
            str(_WORKER_SCRIPT),
            "--socket",
            str(self._socket_path),
            "--runtime",
            str(_RUNTIME_MODULE),
            "--engine-dir",
            str(engine_dir),
            "--auth-fd",
            str(auth_read),
            "--status-fd",
            str(status_write),
            "--parent-pid",
            str(os.getpid()),
            "--kv-cache-fraction",
            str(kv_cache_free_gpu_memory_fraction),
            "--cross-kv-fraction",
            str(cross_kv_cache_fraction),
        ]
        try:
            with self._log_path.open("ab", buffering=0) as log_file:
                self._process = subprocess.Popen(  # noqa: S603 - fixed interpreter and packaged worker paths
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=log_file,
                    close_fds=True,
                    pass_fds=(auth_read, status_write),
                    start_new_session=True,
                    env=environment,
                )
        except Exception:
            for fd in (auth_read, auth_write, status_read, status_write):
                os.close(fd)
            self._tempdir.cleanup()
            raise

        os.close(auth_read)
        os.close(status_write)
        parent_fds = {auth_write, status_read}

        def close_parent_fd(fd: int) -> None:
            if fd not in parent_fds:
                return
            parent_fds.remove(fd)
            with suppress(OSError):
                os.close(fd)

        try:
            try:
                os.write(auth_write, authkey)
            finally:
                # Closing the writer is part of the auth protocol: the child
                # must see EOF if the complete key could not be delivered.
                close_parent_fd(auth_write)

            deadline = time.monotonic() + startup_timeout_sec
            try:
                status = self._read_startup_status(status_read, deadline)
            finally:
                close_parent_fd(status_read)
            if not status.get("ok"):
                error_type = str(status.get("error_type") or "RuntimeError")
                error = str(status.get("error") or "unknown startup error")
                self._raise_startup_error(f"{error_type}: {error}")

            remaining = max(0.0, deadline - time.monotonic())
            if remaining == 0:
                self._raise_startup_error("startup timed out before worker connection")
            self._connection = self._connect_with_timeout(authkey, remaining)
            remaining = max(0.0, deadline - time.monotonic())
            if remaining == 0:
                self._raise_startup_error("startup timed out before the ready frame")
            self._wait_until_readable(remaining)
            ready = self._receive_json()
            if ready.get("protocol") != _PROTOCOL_VERSION or ready.get("ok") is not True or ready.get("op") != "ready":
                self._raise_startup_error(f"invalid ready frame: {ready!r}")
        except BaseException as exc:
            for fd in tuple(parent_fds):
                close_parent_fd(fd)
            if isinstance(exc, RuntimeError) and "Indic Canary worker failed to start" in str(exc):
                raise
            if not isinstance(exc, Exception):
                self._cleanup_after_start_failure()
                raise
            self._raise_startup_error(str(exc))

    def _connect_with_timeout(self, authkey: bytes, timeout_sec: float) -> Connection:  # noqa: C901
        """Bound the AF_UNIX connect and authentication handshake."""
        outcome: queue.Queue[tuple[Connection | None, Exception | None]] = queue.Queue(maxsize=1)
        cancelled = threading.Event()
        ownership_lock = threading.Lock()

        def connect() -> None:
            try:
                connection = Client(str(self._socket_path), family="AF_UNIX", authkey=authkey)
            except Exception as exc:  # noqa: BLE001 - transfer child-thread failure to the owner
                with ownership_lock:
                    if not cancelled.is_set():
                        outcome.put((None, exc))
            else:
                with ownership_lock:
                    close_connection = cancelled.is_set()
                    if not close_connection:
                        outcome.put((connection, None))
                if close_connection:
                    connection.close()

        thread = threading.Thread(target=connect, name="indic-canary-worker-connect", daemon=True)
        thread.start()
        try:
            connection, error = outcome.get(timeout=timeout_sec)
        except queue.Empty:
            # Killing the peer also unblocks a Client stuck in its auth
            # handshake. The daemon flag is a last-resort process-exit guard.
            with ownership_lock:
                cancelled.set()
            self._terminate_process()
            thread.join(timeout=1.0)
            try:
                late_connection, _ = outcome.get_nowait()
            except queue.Empty:
                pass
            else:
                if late_connection is not None:
                    late_connection.close()
            self._raise_startup_error("startup timed out connecting to the authenticated worker socket")
        if error is not None:
            raise error
        if connection is None:
            msg = "worker connection returned neither a connection nor an error"
            raise RuntimeError(msg)
        return connection

    def _read_startup_status(self, fd: int, deadline: float) -> dict[str, Any]:
        chunks: list[bytes] = []
        while time.monotonic() < deadline:
            process = self._process
            if process is None:
                break
            timeout = min(0.1, max(0.0, deadline - time.monotonic()))
            readable, _, _ = select.select([fd], [], [], timeout)
            if not readable:
                if process.poll() is not None:
                    break
                continue
            chunk = os.read(fd, _MAX_JSON_FRAME_BYTES + 1)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(map(len, chunks)) > _MAX_JSON_FRAME_BYTES:
                self._raise_startup_error("startup status exceeded the protocol limit")
            if b"\n" in chunk:
                return _decode_json_object(b"".join(chunks).split(b"\n", maxsplit=1)[0])

        process = self._process
        if process is not None and process.poll() is not None:
            self._raise_startup_error(f"worker exited with status {process.returncode} before readiness")
        return self._raise_startup_error("startup timed out before readiness")

    def _wait_until_readable(self, timeout_sec: float) -> None:
        if self._connection is None:
            self._raise_startup_error("worker connection was not created")
        if not self._connection_is_readable(timeout_sec):
            self._raise_startup_error("startup timed out waiting for the ready frame")

    def _connection_is_readable(self, timeout_sec: float) -> bool:
        connection = self._connection
        if connection is None:
            return False
        readable, _, _ = select.select([connection.fileno()], [], [], timeout_sec)
        return bool(readable)

    def _log_tail(self) -> str:
        try:
            payload = self._log_path.read_bytes()
        except OSError:
            return ""
        return payload[-8192:].decode("utf-8", errors="replace").strip()

    def _raise_startup_error(self, detail: str) -> NoReturn:
        log_tail = self._log_tail()
        self._cleanup_after_start_failure()
        suffix = f"\nWorker log:\n{log_tail}" if log_tail else ""
        msg = f"Indic Canary worker failed to start: {detail}{suffix}"
        raise RuntimeError(msg)

    def _cleanup_after_start_failure(self) -> None:
        """Best-effort cleanup without letting one resource mask the rest."""
        for cleanup in (self._close_connection, self._terminate_process, self._tempdir.cleanup):
            try:
                cleanup()
            except Exception as exc:  # noqa: BLE001 - preserve the original startup failure
                logger.warning("Indic Canary startup cleanup failed: {}", exc)

    def _send_json(self, payload: dict[str, Any]) -> None:
        connection = self._require_connection()
        connection.send_bytes(_json_bytes(payload))

    def _receive_json(self) -> dict[str, Any]:
        connection = self._require_connection()
        first_frame = _decode_json_object(connection.recv_bytes(_MAX_JSON_FRAME_BYTES))
        if first_frame.get("op") != _CHUNKED_JSON_OPERATION:
            return first_frame

        byte_count = first_frame.get("byte_count")
        chunk_count = first_frame.get("chunk_count")
        if (
            first_frame.get("protocol") != _PROTOCOL_VERSION
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, Integral)
            or not _MAX_JSON_FRAME_BYTES < byte_count <= _MAX_JSON_MESSAGE_BYTES
            or isinstance(chunk_count, bool)
            or not isinstance(chunk_count, Integral)
            or chunk_count != (byte_count + _MAX_JSON_FRAME_BYTES - 1) // _MAX_JSON_FRAME_BYTES
        ):
            msg = f"invalid chunked JSON envelope: {first_frame!r}"
            raise TypeError(msg)

        payload = bytearray()
        for _ in range(chunk_count):
            chunk = connection.recv_bytes(_MAX_JSON_FRAME_BYTES)
            expected = min(_MAX_JSON_FRAME_BYTES, byte_count - len(payload))
            if len(chunk) != expected:
                msg = f"invalid chunked JSON payload size: expected {expected}, received {len(chunk)}"
                raise ValueError(msg)
            payload.extend(chunk)
        return _decode_json_object(bytes(payload))

    def _require_connection(self) -> Connection:
        connection = self._connection
        if connection is None:
            msg = "Indic Canary worker connection is closed"
            raise RuntimeError(msg)
        return connection

    def _poison(self, detail: str) -> RuntimeError:
        self._terminal_error = detail
        self._close_connection()
        self._terminate_process()
        log_tail = self._log_tail()
        suffix = f"\nWorker log:\n{log_tail}" if log_tail else ""
        return RuntimeError(f"Indic Canary worker failed terminally: {detail}{suffix}")

    def _exchange(self, request: dict[str, Any], payloads: list[bytes] | None = None) -> dict[str, Any]:
        if self._terminal_error is not None:
            msg = f"Indic Canary worker is unavailable after a terminal failure: {self._terminal_error}"
            raise RuntimeError(msg)
        request_id = uuid.uuid4().hex
        request = {"protocol": _PROTOCOL_VERSION, "request_id": request_id, **request}
        with self._lock:
            try:
                self._send_json(request)
                connection = self._require_connection()
                for payload in payloads or ():
                    connection.send_bytes(payload)
                response = self._receive_json()
            except (EOFError, OSError, RuntimeError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
                raise self._poison(str(exc)) from exc

            if response.get("protocol") != _PROTOCOL_VERSION:
                detail = "response protocol version mismatch"
                raise self._poison(detail)
            if response.get("request_id") != request_id:
                detail = "response request ID mismatch"
                raise self._poison(detail)
            if response.get("ok") is not True:
                error_type = str(response.get("error_type") or "RuntimeError")
                error = str(response.get("error") or "unknown worker error")
                detail = f"{error_type}: {error}"
                raise self._poison(detail)
            return response

    def normalize_languages(self, languages: list[str]) -> list[str | None]:
        """Resolve languages inside the child before the parent touches audio."""
        response = self._exchange({"op": "normalize_languages", "languages": languages})
        normalized = response.get("normalized_languages")
        if (
            not isinstance(normalized, list)
            or len(normalized) != len(languages)
            or not all(language is None or isinstance(language, str) for language in normalized)
        ):
            detail = "worker returned an invalid normalized-language response"
            raise self._poison(detail)
        return normalized

    def infer(  # noqa: PLR0913
        self,
        *,
        item_headers: list[dict[str, Any]],
        waveform_payloads: list[bytes],
        num_beams: int,
        max_new_tokens: int,
        pnc: bool,
        max_duration_sec: float,
        min_duration_sec: float,
    ) -> list[dict[str, Any]]:
        """Send one batch and return manifest-serializable result dictionaries."""
        if len(item_headers) != len(waveform_payloads):
            msg = "Indic Canary worker request metadata and waveform counts differ"
            raise ValueError(msg)

        response = self._exchange(
            {
                "op": "infer",
                "items": item_headers,
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
                "pnc": pnc,
                "max_duration_sec": max_duration_sec,
                "min_duration_sec": min_duration_sec,
            },
            waveform_payloads,
        )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(item_headers):
            detail = (
                f"worker returned {len(results) if isinstance(results, list) else 'invalid'} "
                f"results for {len(item_headers)} inputs"
            )
            raise self._poison(detail)
        if not all(isinstance(result, dict) for result in results):
            detail = "worker returned a non-object result"
            raise self._poison(detail)
        return results

    def _close_connection(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            with suppress(OSError):
                connection.close()

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=_PROCESS_SHUTDOWN_TIMEOUT_SEC)
            except (OSError, subprocess.TimeoutExpired):
                with suppress(OSError):
                    os.killpg(process.pid, signal.SIGKILL)
                try:
                    process.wait(timeout=_PROCESS_SHUTDOWN_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    logger.error("Indic Canary worker process group {} could not be reaped", process.pid)

    def close(self) -> None:
        """Request orderly shutdown, then reap the complete worker process group."""
        acknowledged = False
        lock_acquired = self._lock.acquire(timeout=_CONTROL_MESSAGE_TIMEOUT_SEC)
        if not lock_acquired:
            logger.warning("Indic Canary worker teardown timed out waiting for an in-flight request")
            self._close_connection()
            self._terminate_process()
            self._tempdir.cleanup()
            return
        try:
            process = self._process
            if (
                self._terminal_error is None
                and self._connection is not None
                and process is not None
                and process.poll() is None
            ):
                request_id = uuid.uuid4().hex
                try:
                    self._send_json(
                        {
                            "protocol": _PROTOCOL_VERSION,
                            "request_id": request_id,
                            "op": "shutdown",
                        }
                    )
                    if not self._connection_is_readable(_CONTROL_MESSAGE_TIMEOUT_SEC):
                        logger.warning("Indic Canary worker timed out before its shutdown acknowledgement")
                    else:
                        response = self._receive_json()
                        acknowledged = response.get("request_id") == request_id and response.get("ok") is True
                    if not acknowledged:
                        logger.warning("Indic Canary worker returned an invalid shutdown acknowledgement")
                except (EOFError, OSError, RuntimeError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
                    logger.warning("Indic Canary worker shutdown failed: {}", exc)
            self._close_connection()
        finally:
            self._lock.release()

        process = self._process
        if process is not None and process.poll() is None:
            if acknowledged:
                try:
                    process.wait(timeout=_PROCESS_SHUTDOWN_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    self._terminate_process()
            else:
                self._terminate_process()
        self._tempdir.cleanup()


class IndicCanaryTRTLLMASR:
    """Run static-batch Indic Canary inference in an isolated runtime process."""

    def __init__(  # noqa: PLR0913
        self,
        engine_dir: str,
        *,
        num_beams: int = 4,
        max_new_tokens: int = 374,
        pnc: bool = False,
        max_duration_sec: float = _DEFAULT_MAX_DURATION_SEC,
        min_duration_sec: float = _DEFAULT_MIN_DURATION_SEC,
        kv_cache_free_gpu_memory_fraction: float = 0.2,
        cross_kv_cache_fraction: float = 0.2,
        runtime_python: str | None = None,
        runtime_startup_timeout_sec: float = _DEFAULT_RUNTIME_STARTUP_TIMEOUT_SEC,
    ) -> None:
        if not engine_dir:
            msg = "IndicCanaryTRTLLMASR.engine_dir must point at a prebuilt engine directory"
            raise ValueError(msg)
        if num_beams < 1 or max_new_tokens < 1:
            msg = "num_beams and max_new_tokens must both be at least 1"
            raise ValueError(msg)
        try:
            max_duration_sec = float(max_duration_sec)
            min_duration_sec = float(min_duration_sec)
            runtime_startup_timeout_sec = float(runtime_startup_timeout_sec)
        except (TypeError, ValueError) as exc:
            msg = "duration limits and runtime_startup_timeout_sec must be finite positive numbers"
            raise ValueError(msg) from exc
        if (
            not math.isfinite(max_duration_sec)
            or not math.isfinite(min_duration_sec)
            or max_duration_sec <= 0
            or min_duration_sec <= 0
        ):
            msg = "max_duration_sec and min_duration_sec must both be finite and positive"
            raise ValueError(msg)
        if not math.isfinite(runtime_startup_timeout_sec) or runtime_startup_timeout_sec <= 0:
            msg = "runtime_startup_timeout_sec must be finite and positive"
            raise ValueError(msg)
        if max_duration_sec > _DEFAULT_MAX_DURATION_SEC:
            msg = f"max_duration_sec cannot exceed the {_DEFAULT_MAX_DURATION_SEC:.0f}-second TensorRT encoder window"
            raise ValueError(msg)
        if not 0 < kv_cache_free_gpu_memory_fraction < 1:
            msg = "kv_cache_free_gpu_memory_fraction must be between 0 and 1"
            raise ValueError(msg)
        if not 0 < cross_kv_cache_fraction < 1:
            msg = "cross_kv_cache_fraction must be between 0 and 1"
            raise ValueError(msg)

        self.model_id = engine_dir
        self.engine_dir = engine_dir
        self.num_beams = int(num_beams)
        self.max_new_tokens = int(max_new_tokens)
        self.pnc = bool(pnc)
        self.max_duration_sec = max_duration_sec
        self.min_duration_sec = min(min_duration_sec, self.max_duration_sec)
        self.max_samples = int(self.max_duration_sec * _TARGET_SAMPLE_RATE)
        self.kv_cache_free_gpu_memory_fraction = float(kv_cache_free_gpu_memory_fraction)
        self.cross_kv_cache_fraction = float(cross_kv_cache_fraction)
        self.runtime_python = runtime_python
        self.runtime_startup_timeout_sec = runtime_startup_timeout_sec
        self._model: _CanaryWorkerClient | None = None

    def download_weights_on_node(self) -> None:
        """Validate engine artifacts and the isolated runtime without allocating GPU state."""
        root = Path(self.engine_dir)
        missing = [
            str(root / relative_path)
            for relative_path in _REQUIRED_ENGINE_FILES
            if not (root / relative_path).is_file()
        ]
        if missing:
            msg = f"engine_dir {self.engine_dir!r} is missing required file(s): {missing}"
            raise FileNotFoundError(msg)
        ensure_runtime_python(self.runtime_python)

    def load_model(self, *, num_gpus: int) -> None:
        """Start the isolated TensorRT-LLM worker on the one assigned GPU."""
        if self._model is not None:
            return
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, Integral) or num_gpus != 1:
            msg = f"IndicCanaryTRTLLMASR requires exactly one GPU, got {num_gpus!r}"
            raise ValueError(msg)
        self.download_weights_on_node()
        runtime_python = resolve_runtime_python(self.runtime_python)
        environment = runtime_subprocess_environment(runtime_python)
        logger.info(
            "Loading Indic Canary TensorRT-LLM engine from {} in isolated runtime {}",
            self.engine_dir,
            runtime_python,
        )
        self._model = _CanaryWorkerClient(
            runtime_python=runtime_python,
            environment=environment,
            engine_dir=Path(self.engine_dir).expanduser().resolve(),
            kv_cache_free_gpu_memory_fraction=self.kv_cache_free_gpu_memory_fraction,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
            startup_timeout_sec=self.runtime_startup_timeout_sec,
        )

    def unload_model(self) -> None:
        """Stop the isolated runtime and release all of its CUDA allocations."""
        if self._model is not None:
            self._model.close()
            self._model = None

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe one batch through the isolated runtime process."""
        if not items:
            return []
        if self._model is None:
            msg = "IndicCanaryTRTLLMASR is not initialized; call load_model() first"
            raise RuntimeError(msg)

        languages = [str(item.get("language_code") or "").strip().lower() for item in items]
        normalized_languages = self._model.normalize_languages(languages)
        results = [ASRResult(text="") for _ in items]
        supported_indices: list[int] = []
        item_headers: list[dict[str, Any]] = []
        waveform_payloads: list[bytes] = []
        for index, (item, language, normalized_language) in enumerate(
            zip(items, languages, normalized_languages, strict=True)
        ):
            if normalized_language is None:
                results[index] = ASRResult(
                    text="",
                    skipped=True,
                    skip_reason="language_not_supported",
                    unsupported_language=language or None,
                    extras={"language_code": language, "language_unsupported": True},
                )
                continue
            waveform = np.asarray(item.get("waveform"), dtype=np.float32)
            if waveform.ndim != 1:
                msg = f"ASRStage must provide a mono 1-D waveform, got shape {waveform.shape}"
                raise ValueError(msg)
            sample_rate = int(item.get("sample_rate") or 0)
            if sample_rate != _TARGET_SAMPLE_RATE:
                msg = f"ASRStage must provide {_TARGET_SAMPLE_RATE} Hz audio; received {sample_rate} Hz"
                raise ValueError(msg)
            original_sample_count = int(waveform.size)
            clipped = np.ascontiguousarray(waveform[: self.max_samples], dtype="<f4")
            payload = clipped.tobytes(order="C")
            item_headers.append(
                {
                    "nbytes": len(payload),
                    "sample_count": int(clipped.size),
                    "original_sample_count": original_sample_count,
                    "sample_rate": sample_rate,
                    "language_code": normalized_language,
                }
            )
            waveform_payloads.append(payload)
            supported_indices.append(index)

        wire_results = (
            self._model.infer(
                item_headers=item_headers,
                waveform_payloads=waveform_payloads,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                pnc=self.pnc,
                max_duration_sec=self.max_duration_sec,
                min_duration_sec=self.min_duration_sec,
            )
            if item_headers
            else []
        )
        if len(wire_results) != len(supported_indices):
            msg = f"Indic Canary worker returned {len(wire_results)} results for {len(supported_indices)} inputs"
            raise RuntimeError(msg)
        for index, result in zip(supported_indices, wire_results, strict=True):
            results[index] = ASRResult(
                text=str(result.get("text") or ""),
                skipped=bool(result.get("skipped", False)),
                skip_reason=str(result["skip_reason"]) if result.get("skip_reason") is not None else None,
                unsupported_language=(
                    str(result["unsupported_language"]) if result.get("unsupported_language") is not None else None
                ),
                extras=dict(result.get("extras") or {}),
            )
        return results


__all__ = ["IndicCanaryTRTLLMASR"]
