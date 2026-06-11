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

"""ASR adapter family for the stage-adapter pattern.

Exposes :class:`ASRAdapter` / :class:`ASRResult` (always importable) from
``base``. Concrete adapters (e.g. :class:`QwenOmniASRAdapter`) live in
submodules, resolved via YAML ``adapter_target`` or lazy attribute access
(PEP 562) to avoid importing heavy GPU deps eagerly.
"""

from nemo_curator.stages.audio.inference.asr.adapters.base import ASRAdapter, ASRResult

_LAZY: dict[str, str] = {
    "QwenOmniASRAdapter": ".qwen_omni",
}

__all__ = ["ASRAdapter", "ASRResult", *list(_LAZY)]


def __getattr__(name: str) -> object:
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(_LAZY[name], package=__name__)
        return getattr(mod, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
