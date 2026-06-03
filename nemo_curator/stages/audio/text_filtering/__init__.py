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

"""Text filtering stages for ASR postprocessing."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AbbreviationConcatStage": "nemo_curator.stages.audio.text_filtering.abbreviation_concat",
    "DisfluencyWerGuardStage": "nemo_curator.stages.audio.text_filtering.disfluency_wer_guard",
    "FastTextLIDStage": "nemo_curator.stages.audio.text_filtering.fasttext_lid",
    "FinalizeFieldsStage": "nemo_curator.stages.audio.text_filtering.finalize_fields",
    "InitializeFieldsStage": "nemo_curator.stages.audio.text_filtering.initialize_fields",
    "ITNRestorationStage": "nemo_curator.stages.audio.text_filtering.itn_restoration",
    "PnCContentGuardStage": "nemo_curator.stages.audio.text_filtering.pnc_content_guard",
    "PnCRestorationStage": "nemo_curator.stages.audio.text_filtering.pnc_restoration",
    "RegexSubstitutionStage": "nemo_curator.stages.audio.text_filtering.regex_substitution",
    "WhisperHallucinationStage": "nemo_curator.stages.audio.text_filtering.whisper_hallucination",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
