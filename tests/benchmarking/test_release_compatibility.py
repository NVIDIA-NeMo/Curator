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

"""Check explicit selection and benchmark-only 26.07 argument adaptation."""

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def load_module(name: str):
    path = Path(__file__).resolve().parents[2] / "benchmarking/scripts/release_compatibility" / name
    spec = importlib.util.spec_from_file_location("compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_selection(monkeypatch: "MonkeyPatch"):
    compat = load_module("__init__.py")
    monkeypatch.delenv("CURATOR_BENCHMARK_COMPAT_PROFILE", raising=False)
    assert compat.selected_profile() is None
    monkeypatch.setenv("CURATOR_BENCHMARK_COMPAT_PROFILE", "26.07")
    assert compat.selected_profile() == "26.07"
    monkeypatch.setenv("CURATOR_BENCHMARK_COMPAT_PROFILE", "unknown")
    with pytest.raises(ValueError, match="Unknown"):
        compat.selected_profile()


def test_minhash_preserves_arguments_and_rejects_normalization():
    compat = load_module("curator_26_07.py")
    kwargs = {"output_path": "output", "seed": 42, "num_hashes": 260, "pool": True}
    assert compat.create_minhash_stage(dict, normalize_text=False, **kwargs) == kwargs
    with pytest.raises(ValueError, match="normalize_text=False"):
        compat.create_minhash_stage(dict, normalize_text=True, **kwargs)
