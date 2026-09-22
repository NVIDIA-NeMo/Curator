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

"""CPU-only lifecycle and proxy-contract tests for Indic Canary."""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_curator.models.asr import indic_canary
from nemo_curator.models.asr.base import ASRAdapter
from nemo_curator.models.asr.indic_canary import IndicCanaryTRTLLMASR

_SAMPLE_RATE = 16_000
_REQUIRED_ENGINE_FILES = (
    "encoder/encoder.plan",
    "encoder/config.json",
    "decoder/config.json",
    "decoder/rank0.engine",
    "decoder/vocab.json",
    "preprocessor/config.json",
    "preprocessor/mel_basis.pt",
)


def _write_engine_files(engine_dir: Path) -> None:
    for relative_path in _REQUIRED_ENGINE_FILES:
        path = engine_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")


def _item(
    language: str,
    *,
    samples: int = _SAMPLE_RATE,
    sample_rate: int = _SAMPLE_RATE,
) -> dict[str, object]:
    return {
        "waveform": np.arange(samples, dtype=np.float32),
        "sample_rate": sample_rate,
        "language_code": language,
    }


def test_adapter_conforms_to_asr_protocol() -> None:
    assert isinstance(IndicCanaryTRTLLMASR("/models/indic-canary"), ASRAdapter)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"engine_dir": ""}, "engine_dir must point"),
        ({"engine_dir": "/engine", "num_beams": 0}, "at least 1"),
        ({"engine_dir": "/engine", "max_new_tokens": 0}, "at least 1"),
        ({"engine_dir": "/engine", "max_duration_sec": 0}, "must both be finite and positive"),
        ({"engine_dir": "/engine", "max_duration_sec": float("nan")}, "must both be finite and positive"),
        ({"engine_dir": "/engine", "max_duration_sec": float("inf")}, "must both be finite and positive"),
        ({"engine_dir": "/engine", "max_duration_sec": 40.01}, "cannot exceed the 40-second"),
        ({"engine_dir": "/engine", "min_duration_sec": 0}, "must both be finite and positive"),
        ({"engine_dir": "/engine", "min_duration_sec": float("nan")}, "must both be finite and positive"),
        ({"engine_dir": "/engine", "kv_cache_free_gpu_memory_fraction": 1.0}, "between 0 and 1"),
        ({"engine_dir": "/engine", "cross_kv_cache_fraction": 0.0}, "between 0 and 1"),
        ({"engine_dir": "/engine", "runtime_startup_timeout_sec": 0}, "must be finite and positive"),
        ({"engine_dir": "/engine", "runtime_startup_timeout_sec": float("nan")}, "must be finite and positive"),
    ],
)
def test_adapter_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IndicCanaryTRTLLMASR(**kwargs)  # type: ignore[arg-type]


def test_download_weights_validates_engine_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_runtime = MagicMock(return_value=Path("/runtime/bin/python"))
    monkeypatch.setattr(indic_canary, "ensure_runtime_python", ensure_runtime)
    adapter = IndicCanaryTRTLLMASR(str(tmp_path), runtime_python="/configured/python")

    with pytest.raises(FileNotFoundError, match="missing required file"):
        adapter.download_weights_on_node()
    ensure_runtime.assert_not_called()

    _write_engine_files(tmp_path)
    adapter.download_weights_on_node()
    ensure_runtime.assert_called_once_with("/configured/python")


def test_load_model_starts_one_isolated_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_engine_files(tmp_path)
    runtime_python = tmp_path / "runtime" / "bin" / "python"
    environment = {"CUDA_VISIBLE_DEVICES": "3", "PYTHONNOUSERSITE": "1"}
    ensure_runtime = MagicMock(return_value=runtime_python)
    resolve_runtime = MagicMock(return_value=runtime_python)
    runtime_environment = MagicMock(return_value=environment)
    worker = MagicMock()
    worker_cls = MagicMock(return_value=worker)
    monkeypatch.setattr(indic_canary, "ensure_runtime_python", ensure_runtime)
    monkeypatch.setattr(indic_canary, "resolve_runtime_python", resolve_runtime)
    monkeypatch.setattr(indic_canary, "runtime_subprocess_environment", runtime_environment)
    monkeypatch.setattr(indic_canary, "_CanaryWorkerClient", worker_cls)
    adapter = IndicCanaryTRTLLMASR(
        str(tmp_path),
        kv_cache_free_gpu_memory_fraction=0.1,
        cross_kv_cache_fraction=0.3,
        runtime_python="/configured/python",
        runtime_startup_timeout_sec=12.5,
    )

    adapter.load_model(num_gpus=1)
    adapter.load_model(num_gpus=1)

    assert adapter._model is worker
    ensure_runtime.assert_called_once_with("/configured/python")
    resolve_runtime.assert_called_once_with("/configured/python")
    runtime_environment.assert_called_once_with(runtime_python)
    worker_cls.assert_called_once_with(
        runtime_python=runtime_python,
        environment=environment,
        engine_dir=tmp_path.resolve(),
        kv_cache_free_gpu_memory_fraction=0.1,
        cross_kv_cache_fraction=0.3,
        startup_timeout_sec=12.5,
    )


@pytest.mark.parametrize("num_gpus", [0, 2, -1, 0.5, True])
def test_load_model_requires_exactly_one_gpu(num_gpus: object) -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")

    with pytest.raises(ValueError, match="requires exactly one GPU"):
        adapter.load_model(num_gpus=num_gpus)  # type: ignore[arg-type]


def test_unload_model_closes_worker() -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")
    worker = MagicMock()
    adapter._model = worker

    adapter.unload_model()

    worker.close.assert_called_once_with()
    assert adapter._model is None


def test_transcribe_batch_preserves_results_and_sends_bounded_float32_frames() -> None:
    worker = MagicMock()
    worker.normalize_languages.return_value = ["hi", None, "ta"]
    worker.infer.return_value = [
        {
            "text": "नमस्ते",
            "skipped": False,
            "skip_reason": None,
            "unsupported_language": None,
            "extras": {"language_code": "hi", "truncated": False, "audio_duration_sec": 1.0},
        },
        {
            "text": "வணக்கம்",
            "skipped": False,
            "skip_reason": None,
            "unsupported_language": None,
            "extras": {
                "language_code": "ta",
                "truncated": True,
                "audio_duration_sec": 40.0 + 1 / _SAMPLE_RATE,
            },
        },
    ]
    adapter = IndicCanaryTRTLLMASR(
        "/models/indic-canary",
        num_beams=3,
        max_new_tokens=77,
        pnc=True,
    )
    adapter._model = worker

    results = adapter.transcribe_batch(
        [
            _item("HI", samples=_SAMPLE_RATE),
            _item("zz", samples=8),
            _item("ta", samples=40 * _SAMPLE_RATE + 1),
        ]
    )

    assert [result.text for result in results] == ["नमस्ते", "", "வணக்கம்"]
    assert [result.skipped for result in results] == [False, True, False]
    assert results[1].skip_reason == "language_not_supported"
    assert results[1].unsupported_language == "zz"
    assert results[1].extras == {"language_code": "zz", "language_unsupported": True}
    assert results[2].extras["truncated"] is True
    worker.normalize_languages.assert_called_once_with(["hi", "zz", "ta"])

    call = worker.infer.call_args.kwargs
    assert call["num_beams"] == 3
    assert call["max_new_tokens"] == 77
    assert call["pnc"] is True
    assert call["max_duration_sec"] == 40.0
    assert call["min_duration_sec"] == 0.5
    assert [header["language_code"] for header in call["item_headers"]] == ["hi", "ta"]
    assert call["item_headers"][1] == {
        "nbytes": 40 * _SAMPLE_RATE * 4,
        "sample_count": 40 * _SAMPLE_RATE,
        "original_sample_count": 40 * _SAMPLE_RATE + 1,
        "sample_rate": _SAMPLE_RATE,
        "language_code": "ta",
    }
    payload = call["waveform_payloads"][0]
    assert isinstance(payload, bytes)
    assert np.frombuffer(payload, dtype="<f4").tolist() == np.arange(_SAMPLE_RATE, dtype=np.float32).tolist()


def test_transcribe_batch_requires_loaded_worker() -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")

    with pytest.raises(RuntimeError, match="not initialized"):
        adapter.transcribe_batch([_item("hi")])

    assert adapter.transcribe_batch([]) == []


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (
            {"waveform": np.zeros((1, 8), dtype=np.float32), "sample_rate": _SAMPLE_RATE, "language_code": "hi"},
            "mono 1-D",
        ),
        ({"waveform": np.zeros(8, dtype=np.float32), "sample_rate": 8_000, "language_code": "hi"}, "16000 Hz"),
    ],
)
def test_transcribe_batch_rejects_invalid_audio_without_poisoning_worker(
    item: dict[str, object],
    message: str,
) -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")
    worker = MagicMock()
    worker.normalize_languages.return_value = ["hi"]
    adapter._model = worker

    with pytest.raises(ValueError, match=message):
        adapter.transcribe_batch([item])

    worker.infer.assert_not_called()


def test_transcribe_batch_skips_unsupported_language_before_touching_audio() -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")
    worker = MagicMock()
    worker.normalize_languages.return_value = [None]
    adapter._model = worker

    results = adapter.transcribe_batch(
        [
            {
                "waveform": np.zeros((2, 8), dtype=np.float32),
                "sample_rate": 8_000,
                "language_code": "zz",
            }
        ]
    )

    assert results[0].skipped is True
    assert results[0].unsupported_language == "zz"
    assert results[0].extras == {"language_code": "zz", "language_unsupported": True}
    worker.infer.assert_not_called()


def test_transcribe_batch_rejects_worker_cardinality_mismatch() -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")
    worker = MagicMock()
    worker.normalize_languages.return_value = ["hi"]
    worker.infer.return_value = []
    adapter._model = worker

    with pytest.raises(RuntimeError, match="returned 0 results for 1 inputs"):
        adapter.transcribe_batch([_item("hi")])


def test_worker_client_failure_is_terminal_and_never_replayed(tmp_path: Path) -> None:
    class _ErrorConnection:
        def __init__(self) -> None:
            self.sent: list[bytes] = []
            self.closed = False

        def send_bytes(self, payload: bytes) -> None:
            self.sent.append(payload)

        def recv_bytes(self, _maxlength: int) -> bytes:
            request_id = json.loads(self.sent[0])["request_id"]
            return json.dumps(
                {
                    "protocol": 1,
                    "ok": False,
                    "request_id": request_id,
                    "error_type": "RuntimeError",
                    "error": "engine failed",
                }
            ).encode()

        def close(self) -> None:
            self.closed = True

    connection = _ErrorConnection()
    client = object.__new__(indic_canary._CanaryWorkerClient)
    client._connection = connection
    client._process = None
    client._terminal_error = None
    client._lock = threading.Lock()
    client._log_path = tmp_path / "missing.log"
    payload = np.zeros(8, dtype="<f4").tobytes()
    kwargs = {
        "item_headers": [
            {
                "nbytes": len(payload),
                "sample_count": 8,
                "original_sample_count": 8,
                "sample_rate": _SAMPLE_RATE,
                "language_code": "hi",
            }
        ],
        "waveform_payloads": [payload],
        "num_beams": 4,
        "max_new_tokens": 374,
        "pnc": False,
        "max_duration_sec": 40.0,
        "min_duration_sec": 0.5,
    }

    with pytest.raises(RuntimeError, match="failed terminally: RuntimeError: engine failed"):
        client.infer(**kwargs)
    sent_count = len(connection.sent)
    assert connection.closed is True

    with pytest.raises(RuntimeError, match="unavailable after a terminal failure"):
        client.infer(**kwargs)
    assert len(connection.sent) == sent_count


def test_worker_client_reassembles_bounded_chunked_json_response() -> None:
    expected = {
        "protocol": 1,
        "ok": True,
        "request_id": "large-response",
        "results": [{"text": "x" * indic_canary._MAX_JSON_FRAME_BYTES}],
    }
    encoded = indic_canary._json_bytes(expected)
    chunks = [
        encoded[start : start + indic_canary._MAX_JSON_FRAME_BYTES]
        for start in range(0, len(encoded), indic_canary._MAX_JSON_FRAME_BYTES)
    ]
    envelope = indic_canary._json_bytes(
        {
            "protocol": 1,
            "op": indic_canary._CHUNKED_JSON_OPERATION,
            "byte_count": len(encoded),
            "chunk_count": len(chunks),
        }
    )
    connection = MagicMock()
    connection.recv_bytes.side_effect = [envelope, *chunks]
    client = object.__new__(indic_canary._CanaryWorkerClient)
    client._connection = connection

    assert client._receive_json() == expected
    assert connection.recv_bytes.call_count == len(chunks) + 1


def test_worker_client_close_bounds_shutdown_ack_wait() -> None:
    class _NoAckConnection:
        def __init__(self) -> None:
            self.sent: list[bytes] = []
            self.closed = False

        def send_bytes(self, payload: bytes) -> None:
            self.sent.append(payload)

        def recv_bytes(self, _maxlength: int) -> bytes:
            msg = "close must not block in recv_bytes without readability"
            raise AssertionError(msg)

        def close(self) -> None:
            self.closed = True

    connection = _NoAckConnection()
    client = object.__new__(indic_canary._CanaryWorkerClient)
    client._connection = connection
    client._process = MagicMock()
    client._process.poll.return_value = None
    client._terminal_error = None
    client._lock = threading.Lock()
    client._connection_is_readable = MagicMock(return_value=False)
    client._terminate_process = MagicMock()
    client._tempdir = MagicMock()

    client.close()

    assert len(connection.sent) == 1
    assert json.loads(connection.sent[0])["op"] == "shutdown"
    assert connection.closed is True
    client._connection_is_readable.assert_called_once_with(indic_canary._CONTROL_MESSAGE_TIMEOUT_SEC)
    client._terminate_process.assert_called_once_with()
    client._tempdir.cleanup.assert_called_once_with()


def test_worker_client_bounds_socket_authentication_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    started = threading.Event()
    late_connection_closed = threading.Event()
    late_connection = MagicMock()
    late_connection.close.side_effect = late_connection_closed.set

    def blocking_client(*_args: object, **_kwargs: object) -> MagicMock:
        started.set()
        release.wait()
        return late_connection

    monkeypatch.setattr(indic_canary, "Client", blocking_client)
    client = object.__new__(indic_canary._CanaryWorkerClient)
    client._socket_path = tmp_path / "worker.sock"
    client._terminate_process = MagicMock()
    client._raise_startup_error = MagicMock(side_effect=RuntimeError("startup timed out"))

    before = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="startup timed out"):
            client._connect_with_timeout(b"x" * 32, 0.01)
    finally:
        release.set()

    assert started.is_set()
    assert time.monotonic() - before < 1.5
    client._terminate_process.assert_called_once_with()
    assert late_connection_closed.wait(timeout=1.0)


@pytest.mark.parametrize("failure", ["auth_write", "malformed_status", "control_flow"])
def test_worker_client_cleans_up_every_post_spawn_startup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    worker = tmp_path / "startup_worker.py"
    worker.write_text(
        """
import os
import sys
import time

auth_fd = int(sys.argv[sys.argv.index("--auth-fd") + 1])
status_fd = int(sys.argv[sys.argv.index("--status-fd") + 1])
os.read(auth_fd, 32)
os.close(auth_fd)
os.write(status_fd, b"not-json\\n")
os.close(status_fd)
time.sleep(60)
""",
        encoding="utf-8",
    )
    runtime_module = tmp_path / "runtime.py"
    runtime_module.write_text("", encoding="utf-8")
    monkeypatch.setattr(indic_canary, "_WORKER_SCRIPT", worker)
    monkeypatch.setattr(indic_canary, "_RUNTIME_MODULE", runtime_module)

    client = object.__new__(indic_canary._CanaryWorkerClient)
    client._connection = None
    client._process = None
    client._terminal_error = None
    client._lock = threading.Lock()
    client._tempdir = tempfile.TemporaryDirectory(prefix="nc-canary-test-", dir=tmp_path)
    tempdir_path = Path(client._tempdir.name)
    client._socket_path = tempdir_path / "worker.sock"
    client._log_path = tempdir_path / "worker.log"

    if failure == "auth_write":
        monkeypatch.setattr(indic_canary.os, "write", MagicMock(side_effect=BrokenPipeError("closed auth pipe")))
    elif failure == "control_flow":
        client._read_startup_status = MagicMock(side_effect=KeyboardInterrupt)

    expected_exception = KeyboardInterrupt if failure == "control_flow" else RuntimeError
    with pytest.raises(
        expected_exception, match=None if failure == "control_flow" else "Indic Canary worker failed to start"
    ):
        client._start(
            runtime_python=Path(sys.executable),
            environment=dict(os.environ),
            engine_dir=tmp_path,
            kv_cache_free_gpu_memory_fraction=0.2,
            cross_kv_cache_fraction=0.2,
            startup_timeout_sec=5.0,
        )

    assert client._process is not None
    assert client._process.poll() is not None
    assert not tempdir_path.exists()


def test_startup_cleanup_does_not_let_connection_close_mask_process_reaping() -> None:
    client = object.__new__(indic_canary._CanaryWorkerClient)
    client._close_connection = MagicMock(side_effect=OSError("socket close failed"))
    client._terminate_process = MagicMock()
    client._tempdir = MagicMock()
    client._log_path = Path("/missing/worker.log")

    with pytest.raises(RuntimeError, match="original startup failure"):
        client._raise_startup_error("original startup failure")

    client._terminate_process.assert_called_once_with()
    client._tempdir.cleanup.assert_called_once_with()


def test_close_connection_releases_ownership_when_socket_close_fails() -> None:
    connection = MagicMock()
    connection.close.side_effect = OSError("socket close failed")
    client = object.__new__(indic_canary._CanaryWorkerClient)
    client._connection = connection

    client._close_connection()

    assert client._connection is None
    connection.close.assert_called_once_with()


def test_worker_client_runs_persistent_cross_process_byte_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runtime = tmp_path / "fake_runtime.py"
    fake_runtime.write_text(
        """
import torch

class _Tokenizer:
    langs = ["hi"]
    @staticmethod
    def supports_prompt_language(language):
        return language == "hi"

class CanaryTRTLLM:
    def __init__(self, *args, **kwargs):
        self.tokenizer = _Tokenizer()
        self.max_batch_size = 2

    def process_batch(self, audio, durations, prompts, *, num_beams, max_new_tokens):
        return [f"{prompt['source_language']}:{duration}:{float(row[:duration].sum()):.1f}" for row, duration, prompt in zip(audio, durations, prompts, strict=True)]

def pad_or_trim(waveform, length):
    if waveform.shape[0] >= length:
        return waveform[:length]
    return torch.nn.functional.pad(waveform, (0, length - waveform.shape[0]))
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(indic_canary, "_RUNTIME_MODULE", fake_runtime)
    client = indic_canary._CanaryWorkerClient(
        runtime_python=Path(sys.executable),
        environment=dict(os.environ),
        engine_dir=tmp_path,
        kv_cache_free_gpu_memory_fraction=0.2,
        cross_kv_cache_fraction=0.2,
        startup_timeout_sec=30.0,
    )
    process = client._process
    payload = np.asarray([0.25, -0.25, 0.5], dtype="<f4").tobytes()
    kwargs = {
        "item_headers": [
            {
                "nbytes": len(payload),
                "sample_count": 3,
                "original_sample_count": 3,
                "sample_rate": _SAMPLE_RATE,
                "language_code": "hi-IN",
            }
        ],
        "waveform_payloads": [payload],
        "num_beams": 4,
        "max_new_tokens": 374,
        "pnc": False,
        "max_duration_sec": 40.0,
        "min_duration_sec": 0.5,
    }

    try:
        assert client.normalize_languages(["hi-IN", "zz"]) == ["hi", None]
        first = client.infer(**kwargs)
        second = client.infer(**kwargs)
    finally:
        client.close()

    assert first == second
    assert first[0]["text"] == "hi:400:0.5"
    assert process is not None
    assert process.returncode == 0
