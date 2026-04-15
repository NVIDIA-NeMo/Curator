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

import importlib
import logging

_logger = logging.getLogger(__name__)

__all__: list[str] = []


def _try_import(module_path: str, names: list[str]) -> None:
    try:
        mod = importlib.import_module(module_path)
        for name in names:
            globals()[name] = getattr(mod, name)
            __all__.append(name)
    except Exception as exc:  # noqa: BLE001
        _logger.debug("Skipping %s: %s", module_path, exc)


_try_import("nemo_curator.stages.audio.request.omni_llm_request", ["OmniLLMRequestStage"])
_try_import("nemo_curator.stages.audio.request.prepare_omni_lhotse", ["PrepareOmniLhotseStage"])
_try_import("nemo_curator.stages.audio.request.prepare_omni_request", ["PrepareOmniRequestStage"])
_try_import("nemo_curator.stages.audio.request.text_only_llm_request", ["TextOnlyLLMRequestStage"])
_try_import("nemo_curator.stages.audio.request.prompt_template", ["build_prompt_conversation", "load_prompt_config"])
_try_import("nemo_curator.stages.audio.request.transcription_cascade", [
    "TranscriptionCascadeConfig", "run_cascade_on_row", "get_prompt_path", "create_pnc_stage",
])
