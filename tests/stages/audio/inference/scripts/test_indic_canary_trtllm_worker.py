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

"""Protocol and inference tests for the standalone Indic Canary worker."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.inference.scripts import indic_canary_trtllm_worker as worker

_SAMPLE_RATE = 16_000


class _FrameConnection:
    def __init__(self, incoming: list[bytes]) -> None:
        self.incoming = list(incoming)
        self.sent: list[bytes] = []
        self.closed = False

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        payload = self.incoming.pop(0)
        if maxlength is not None and len(payload) > maxlength:
            msg = "frame exceeds maxlength"
            raise OSError(msg)
        return payload

    def send_bytes(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self) -> object:
        msg = "pickle-bearing recv() must never be used"
        raise AssertionError(msg)

    def send(self, _payload: object) -> None:
        msg = "pickle-bearing send() must never be used"
        raise AssertionError(msg)

    def close(self) -> None:
        self.closed = True


class _RecordingModel:
    def __init__(self, *, max_batch_size: int = 2) -> None:
        self.tokenizer = SimpleNamespace(
            langs=["hi", "ta"],
            supports_prompt_language=lambda language: language in {"hi", "ta"},
        )
        self.max_batch_size = max_batch_size
        self.calls: list[dict[str, Any]] = []

    def process_batch(
        self,
        padded: list[torch.Tensor],
        durations: list[int],
        prompts: list[dict[str, object]],
        *,
        num_beams: int,
        max_new_tokens: int,
    ) -> list[str]:
        self.calls.append(
            {
                "padded": padded,
                "durations": durations,
                "prompts": prompts,
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
            }
        )
        return [f"text-{prompt['source_language']}" for prompt in prompts]


def _runtime() -> SimpleNamespace:
    def pad_or_trim(waveform: torch.Tensor, length: int) -> torch.Tensor:
        if waveform.shape[0] >= length:
            return waveform[:length]
        return torch.nn.functional.pad(waveform, (0, length - waveform.shape[0]))

    return SimpleNamespace(pad_or_trim=pad_or_trim)


def _waveform(samples: int) -> bytes:
    return np.arange(samples, dtype="<f4").tobytes()


def _header(language: str, samples: int, *, original_samples: int | None = None) -> dict[str, object]:
    return {
        "nbytes": samples * 4,
        "sample_count": samples,
        "original_sample_count": samples if original_samples is None else original_samples,
        "sample_rate": _SAMPLE_RATE,
        "language_code": language,
    }


def _request(items: list[dict[str, object]]) -> dict[str, Any]:
    return {
        "protocol": 1,
        "request_id": "request-1",
        "op": "infer",
        "items": items,
        "num_beams": 3,
        "max_new_tokens": 77,
        "pnc": True,
        "max_duration_sec": 1.0,
        "min_duration_sec": 0.5,
    }


def test_worker_has_no_nemo_curator_imports() -> None:
    source_path = Path(worker.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    )

    assert not any(module == "nemo_curator" or module.startswith("nemo_curator.") for module in imported_modules)


def test_send_json_chunks_large_results_into_bounded_byte_frames() -> None:
    connection = _FrameConnection([])
    payload = {"protocol": 1, "ok": True, "results": [{"text": "x" * worker._MAX_JSON_FRAME_BYTES}]}

    worker._send_json(connection, payload)  # type: ignore[arg-type]

    envelope = json.loads(connection.sent[0])
    assert envelope == {
        "protocol": 1,
        "op": worker._CHUNKED_JSON_OPERATION,
        "byte_count": sum(len(chunk) for chunk in connection.sent[1:]),
        "chunk_count": len(connection.sent) - 1,
    }
    assert all(len(chunk) <= worker._MAX_JSON_FRAME_BYTES for chunk in connection.sent[1:])
    assert json.loads(b"".join(connection.sent[1:])) == payload


def test_infer_uses_byte_frames_and_preserves_reference_semantics() -> None:
    model = _RecordingModel(max_batch_size=2)
    items = [
        _header("hi-IN", 100),
        _header("zz", 8),
        _header("ta", 800),
        _header("hi", _SAMPLE_RATE, original_samples=_SAMPLE_RATE + 1),
    ]
    connection = _FrameConnection([_waveform(100), _waveform(8), _waveform(800), _waveform(_SAMPLE_RATE)])

    results = worker._infer(model, _runtime(), _request(items), connection)  # type: ignore[arg-type]

    assert [result["text"] for result in results] == ["text-hi", "", "text-ta", "text-hi"]
    assert results[1] == {
        "text": "",
        "skipped": True,
        "skip_reason": "language_not_supported",
        "unsupported_language": "zz",
        "extras": {"language_code": "zz", "language_unsupported": True},
    }
    assert results[3]["extras"] == {
        "language_code": "hi",
        "truncated": True,
        "audio_duration_sec": (_SAMPLE_RATE + 1) / _SAMPLE_RATE,
    }
    assert connection.incoming == []
    assert [len(call["padded"]) for call in model.calls] == [2, 1]
    assert [prompt["source_language"] for call in model.calls for prompt in call["prompts"]] == [
        "hi",
        "ta",
        "hi",
    ]
    assert [duration for call in model.calls for duration in call["durations"]] == [400, 800, _SAMPLE_RATE]
    assert all(call["num_beams"] == 3 and call["max_new_tokens"] == 77 for call in model.calls)


def test_infer_rejects_runtime_cardinality_mismatch() -> None:
    model = _RecordingModel()
    model.process_batch = lambda *_args, **_kwargs: []  # type: ignore[method-assign]
    connection = _FrameConnection([_waveform(8)])

    with pytest.raises(RuntimeError, match="returned 0 transcriptions for 1 inputs"):
        worker._infer(model, _runtime(), _request([_header("hi", 8)]), connection)  # type: ignore[arg-type]


def test_infer_rejects_declared_payload_mismatch() -> None:
    connection = _FrameConnection([_waveform(7)])

    with pytest.raises(ValueError, match="declared 32 waveform bytes but sent 28"):
        worker._infer(  # type: ignore[arg-type]
            _RecordingModel(),
            _runtime(),
            _request([_header("hi", 8)]),
            connection,
        )


def test_request_failure_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_request = json.dumps(
        {"protocol": 1, "request_id": "bad", "op": "not-an-operation"},
        separators=(",", ":"),
    ).encode()
    unconsumed_request = json.dumps(
        {"protocol": 1, "request_id": "later", "op": "shutdown"},
        separators=(",", ":"),
    ).encode()
    connection = _FrameConnection([bad_request, unconsumed_request])

    class _Listener:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.closed = False

        def accept(self) -> _FrameConnection:
            return connection

        def close(self) -> None:
            self.closed = True

    listener = _Listener()
    monkeypatch.setattr(worker, "Listener", lambda *_args, **_kwargs: listener)
    monkeypatch.setattr(worker, "_arm_parent_death_signal", lambda _pid: None)
    monkeypatch.setattr(
        worker,
        "_load_runtime",
        lambda _path: SimpleNamespace(CanaryTRTLLM=lambda *_args, **_kwargs: _RecordingModel()),
    )
    monkeypatch.setattr(worker.os, "chmod", lambda *_args: None)

    auth_read, auth_write = os.pipe()
    status_read, status_write = os.pipe()
    os.write(auth_write, b"x" * 32)
    os.close(auth_write)
    args = SimpleNamespace(
        parent_pid=os.getpid(),
        auth_fd=auth_read,
        status_fd=status_write,
        socket=str(tmp_path / "worker.sock"),
        runtime="/runtime.py",
        engine_dir="/engine",
        kv_cache_fraction=0.2,
        cross_kv_fraction=0.2,
    )

    assert worker._serve(args) == 0
    status = os.read(status_read, 4096)
    os.close(status_read)

    assert json.loads(status)["ok"] is True
    assert len(connection.incoming) == 1
    assert json.loads(connection.sent[0])["op"] == "ready"
    assert json.loads(connection.sent[1]) == {
        "protocol": 1,
        "ok": False,
        "request_id": "bad",
        "error_type": "ValueError",
        "error": "unknown operation 'not-an-operation'",
    }
    assert connection.closed is True
    assert listener.closed is True
