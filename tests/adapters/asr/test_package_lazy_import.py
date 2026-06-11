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

"""Tests that ``nemo_curator.adapters.asr`` does not eagerly import GPU adapters."""

from __future__ import annotations

import builtins
import sys

import pytest


def test_importing_asr_subpackage_does_not_load_qwen_omni(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import nemo_curator.adapters.asr`` must not pull in ``qwen_omni`` at init time."""
    original_import = builtins.__import__
    blocked: list[str] = []

    def tracking_import(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.endswith("nemo_curator.adapters.asr.qwen_omni") or name == "nemo_curator.adapters.asr.qwen_omni":
            blocked.append(name)
            msg = f"blocked eager import of {name}"
            raise ImportError(msg)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    for mod_name in list(sys.modules):
        if mod_name in {"nemo_curator.adapters.asr", "nemo_curator.adapters.asr.base"}:
            del sys.modules[mod_name]

    import nemo_curator.adapters.asr as asr_pkg

    assert blocked == []
    assert asr_pkg.ASRAdapter is not None
    assert asr_pkg.ASRResult is not None
    assert "QwenOmniASRAdapter" in asr_pkg._LAZY


def test_asr_subpackage_lazy_getattr_resolves_qwen_adapter() -> None:
    from nemo_curator.adapters.asr import QwenOmniASRAdapter

    assert QwenOmniASRAdapter.__name__ == "QwenOmniASRAdapter"
