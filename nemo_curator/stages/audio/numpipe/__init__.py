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

"""NumPipe: automatic generation, vocalization, judging and evaluation of spoken numbers.

The pipeline measures how accurately an ASR model transcribes numbers, on the *value* that was
spoken rather than on the written form, and can produce verified training data with the same
stages:

1. :class:`NumericEntityGeneratorStage` — numbers with a known written value and a spoken reading.
2. :class:`TtsSynthesisStage` — speech for every reading (pluggable TTS backend).
3. :class:`JudgeStage` + :class:`FilterFaithfulStage` — drop clips the synthesizer misread, using
   independent ASR/LLM transcribers that are never told the target.
4. :class:`AsrStage` + :class:`NumericEqualityStage` — transcribe with the model under test and
   compare the recognized value with the intended one.

:mod:`summary` turns the scored manifest into accuracy tables with confidence intervals.
"""

from nemo_curator.stages.audio.numpipe.asr import AsrStage
from nemo_curator.stages.audio.numpipe.generate import GenerationConfig, NumericEntityGeneratorStage
from nemo_curator.stages.audio.numpipe.judge import FilterFaithfulStage, JudgeStage
from nemo_curator.stages.audio.numpipe.score import NumericEqualityStage
from nemo_curator.stages.audio.numpipe.tts import TtsSynthesisStage

__all__ = [
    "AsrStage",
    "FilterFaithfulStage",
    "GenerationConfig",
    "JudgeStage",
    "NumericEntityGeneratorStage",
    "NumericEqualityStage",
    "TtsSynthesisStage",
]
