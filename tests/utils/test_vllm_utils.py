# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for shared vLLM utility helpers (CPU-only, no GPU required)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

import nemo_curator.utils.vllm_utils as _vllm_utils
from nemo_curator.utils.vllm_utils import pick_free_port, resolve_local_model_path


class TestPickFreePort:
    def test_returns_valid_port(self):
        port = pick_free_port()
        assert isinstance(port, int)
        assert 1 <= port <= 65535

    def test_two_calls_return_ports(self):
        # Both calls should succeed (ports may differ)
        p1 = pick_free_port()
        p2 = pick_free_port()
        assert isinstance(p1, int)
        assert isinstance(p2, int)


class TestResolveLocalModelPath:
    def test_existing_dir_returned_unchanged(self, tmp_path: Path):
        result = resolve_local_model_path(str(tmp_path))
        assert result == str(tmp_path)

    def test_not_cached_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch):
        from huggingface_hub.errors import LocalEntryNotFoundError

        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            lambda *_a, **_kw: (_ for _ in ()).throw(LocalEntryNotFoundError("not found")),
        )
        with pytest.raises(RuntimeError, match="not cached locally"):
            resolve_local_model_path("some-org/some-model")

    def test_error_message_contains_download_hint(self, monkeypatch: pytest.MonkeyPatch):
        from huggingface_hub.errors import LocalEntryNotFoundError

        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            lambda *_a, **_kw: (_ for _ in ()).throw(LocalEntryNotFoundError("not found")),
        )
        with pytest.raises(RuntimeError, match="huggingface-cli download"):
            resolve_local_model_path("org/model")


class TestCreateVllmLlm:
    """Tests for create_vllm_llm: port-collision retry and error propagation."""

    def _inject_fake_vllm(self, monkeypatch: pytest.MonkeyPatch, llm_class: type) -> None:
        """Insert a fake vllm module so the local `from vllm import LLM` import succeeds."""
        import sys
        import types

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = llm_class
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    def test_eaddrinuse_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch):
        """On EADDRINUSE the helper should retry and return the LLM on success."""
        call_count = 0
        err_msg = "EADDRINUSE: port already in use"

        class FakeLLM:
            def __init__(self, **_kw):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise RuntimeError(err_msg)

        self._inject_fake_vllm(monkeypatch, FakeLLM)
        monkeypatch.setattr(_vllm_utils, "pick_free_port", lambda: 12345)
        monkeypatch.setattr("time.sleep", lambda _: None)

        result = _vllm_utils.create_vllm_llm("fake/model", max_port_retries=3)
        assert isinstance(result, FakeLLM)
        assert call_count == 2

    def test_eaddrinuse_exhausted_reraises(self, monkeypatch: pytest.MonkeyPatch):
        """After max_port_retries all fail with EADDRINUSE, the error is re-raised."""
        err_msg = "address already in use"

        class FakeLLM:
            def __init__(self, **_kw):
                raise RuntimeError(err_msg)

        self._inject_fake_vllm(monkeypatch, FakeLLM)
        monkeypatch.setattr(_vllm_utils, "pick_free_port", lambda: 12345)
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(RuntimeError, match="address already in use"):
            _vllm_utils.create_vllm_llm("fake/model", max_port_retries=2)

    def test_wrapped_engine_startup_failure_retries(self, monkeypatch: pytest.MonkeyPatch):
        """In vLLM v1 the bind error is wrapped; the generic startup message must still retry."""
        call_count = 0
        # This is what the parent process actually sees in v1 (EADDRINUSE is buried
        # in the EngineCore subprocess and never reaches here).
        err_msg = "Engine core initialization failed. See root cause above. Failed core proc(s): {}"

        class FakeLLM:
            def __init__(self, **_kw):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError(err_msg)

        self._inject_fake_vllm(monkeypatch, FakeLLM)
        monkeypatch.setattr(_vllm_utils, "pick_free_port", lambda: 12345)
        monkeypatch.setattr("time.sleep", lambda _: None)

        result = _vllm_utils.create_vllm_llm("fake/model", max_port_retries=5)
        assert isinstance(result, FakeLLM)
        assert call_count == 3

    def test_wrapped_non_retryable_startup_failure_raises_without_retry(self, monkeypatch: pytest.MonkeyPatch):
        """Fatal EngineCore failures should fail fast when the exception chain exposes the cause."""
        call_count = 0

        class FakeLLM:
            def __init__(self, **_kw):
                nonlocal call_count
                call_count += 1
                outer_msg = "Engine core initialization failed"
                cause_msg = "CUDA out of memory"
                raise RuntimeError(outer_msg) from RuntimeError(cause_msg)

        self._inject_fake_vllm(monkeypatch, FakeLLM)
        monkeypatch.setattr(_vllm_utils, "pick_free_port", lambda: 12345)
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(RuntimeError, match="Engine core initialization failed"):
            _vllm_utils.create_vllm_llm("fake/model", max_port_retries=3)

        assert call_count == 1

    def test_extra_engine_kwargs_forwarded(self, monkeypatch: pytest.MonkeyPatch):
        """Extra engine kwargs (e.g. gpu_memory_utilization) reach vllm.LLM."""
        captured_kwargs: dict = {}

        class FakeLLM:
            def __init__(self, **kw):
                captured_kwargs.update(kw)

        self._inject_fake_vllm(monkeypatch, FakeLLM)
        monkeypatch.setattr(_vllm_utils, "pick_free_port", lambda: 12345)

        _vllm_utils.create_vllm_llm(
            "fake/model",
            max_num_seqs=128,
            gpu_memory_utilization=0.95,
            max_num_batched_tokens=16384,
        )
        assert captured_kwargs.get("max_num_seqs") == 128
        assert captured_kwargs.get("gpu_memory_utilization") == 0.95
        assert captured_kwargs.get("max_num_batched_tokens") == 16384

    def test_non_eaddrinuse_raises_immediately(self, monkeypatch: pytest.MonkeyPatch):
        """A non-port-collision RuntimeError should propagate without retry."""
        call_count = 0
        err_msg = "CUDA out of memory"

        class FakeLLM:
            def __init__(self, **_kw):
                nonlocal call_count
                call_count += 1
                raise RuntimeError(err_msg)

        self._inject_fake_vllm(monkeypatch, FakeLLM)
        monkeypatch.setattr(_vllm_utils, "pick_free_port", lambda: 12345)

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            _vllm_utils.create_vllm_llm("fake/model", max_port_retries=3)

        assert call_count == 1  # no retry

    def test_default_limit_mm_per_prompt(self, monkeypatch: pytest.MonkeyPatch):
        """When limit_mm_per_prompt is None, it defaults to {'image': 1}."""
        captured_kwargs: dict = {}

        class FakeLLM:
            def __init__(self, **kw):
                captured_kwargs.update(kw)

        self._inject_fake_vllm(monkeypatch, FakeLLM)
        monkeypatch.setattr(_vllm_utils, "pick_free_port", lambda: 12345)

        _vllm_utils.create_vllm_llm("fake/model", limit_mm_per_prompt=None)
        assert captured_kwargs.get("limit_mm_per_prompt") == {"image": 1}
