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

"""Tests that the ASR model package does not eagerly import GPU adapters."""

from __future__ import annotations

import builtins
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def test_importing_asr_subpackage_does_not_load_concrete_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """The package init must not pull in concrete ASR adapters."""
    original_import = builtins.__import__
    blocked: list[str] = []
    concrete_modules = {
        "nemo_curator.models.asr.faster_whisper",
        "nemo_curator.models.asr.nemo_asr",
        "nemo_curator.models.asr.qwen_asr",
        "nemo_curator.models.asr.qwen_omni",
    }

    def tracking_import(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name in concrete_modules:
            blocked.append(name)
            msg = f"blocked eager import of {name}"
            raise ImportError(msg)
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    module_names = {
        "nemo_curator.models.asr",
        "nemo_curator.models.asr.base",
        "nemo_curator.models.asr.faster_whisper",
        "nemo_curator.models.asr.nemo_asr",
        "nemo_curator.models.asr.qwen_asr",
        "nemo_curator.models.asr.qwen_omni",
    }
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        for mod_name in module_names:
            sys.modules.pop(mod_name, None)

        import nemo_curator.models.asr as asr_pkg

        assert blocked == []
        assert "FasterWhisperASR" not in vars(asr_pkg)
        assert "NeMoASRAdapter" not in vars(asr_pkg)
        assert "QwenASRAdapter" not in vars(asr_pkg)
        assert "QwenOmniASRAdapter" not in vars(asr_pkg)
    finally:
        for mod_name in module_names:
            sys.modules.pop(mod_name, None)
        for mod_name, module in saved_modules.items():
            if module is not None:
                sys.modules[mod_name] = module


def test_hydra_resolves_adapters_from_module_paths() -> None:
    import hydra.utils

    targets = {
        "nemo_curator.models.asr.faster_whisper.FasterWhisperASR": "FasterWhisperASR",
        "nemo_curator.models.asr.qwen_asr.QwenASRAdapter": "QwenASRAdapter",
        "nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter": "QwenOmniASRAdapter",
    }
    for target, expected_name in targets.items():
        assert hydra.utils.get_class(target).__name__ == expected_name


def test_hydra_resolves_nemo_adapter_from_module_path() -> None:
    import hydra.utils

    adapter_cls = hydra.utils.get_class("nemo_curator.models.asr.nemo_asr.NeMoASRAdapter")

    assert adapter_cls.__name__ == "NeMoASRAdapter"
