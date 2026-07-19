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

"""Tests that the audio ASR adapter package does not eagerly import GPU adapters."""

from __future__ import annotations

import builtins
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def test_importing_asr_subpackage_does_not_load_concrete_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """The package init must not pull in the concrete ASR implementation."""
    original_import = builtins.__import__
    blocked: list[str] = []

    def tracking_import(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "nemo_curator.stages.audio.inference.asr.adapters.qwen_omni":
            blocked.append(name)
            msg = f"blocked eager import of {name}"
            raise ImportError(msg)
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    module_names = {
        "nemo_curator.stages.audio.inference.asr.adapters",
        "nemo_curator.stages.audio.inference.asr.adapters.base",
        "nemo_curator.stages.audio.inference.asr.adapters.qwen_omni",
    }
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        for mod_name in module_names:
            sys.modules.pop(mod_name, None)

        import nemo_curator.stages.audio.inference.asr.adapters as asr_pkg

        assert blocked == []
        assert asr_pkg.ASRAdapter is not None
        assert asr_pkg.ASRResult is not None
        assert "QwenOmniASRAdapter" in asr_pkg._LAZY
    finally:
        for mod_name in module_names:
            sys.modules.pop(mod_name, None)
        for mod_name, module in saved_modules.items():
            if module is not None:
                sys.modules[mod_name] = module


def test_asr_subpackage_lazy_getattr_resolves_qwen_adapter() -> None:
    from nemo_curator.stages.audio.inference.asr.adapters import QwenOmniASRAdapter

    assert QwenOmniASRAdapter.__name__ == "QwenOmniASRAdapter"
