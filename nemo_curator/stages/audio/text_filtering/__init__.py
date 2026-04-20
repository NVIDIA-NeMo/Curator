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

from nemo_curator.stages.audio.text_filtering.fasttext_lid import FastTextLIDStage
from nemo_curator.stages.audio.text_filtering.finalize_fields import FinalizeFieldsStage
from nemo_curator.stages.audio.text_filtering.initialize_fields import InitializeFieldsStage
from nemo_curator.stages.audio.text_filtering.regex_substitution import RegexSubstitutionStage
from nemo_curator.stages.audio.text_filtering.whisper_hallucination import WhisperHallucinationStage

__all__ = [
    "FastTextLIDStage",
    "FinalizeFieldsStage",
    "InitializeFieldsStage",
    "RegexSubstitutionStage",
    "WhisperHallucinationStage",
]
