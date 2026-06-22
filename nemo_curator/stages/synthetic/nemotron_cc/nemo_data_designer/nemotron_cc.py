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

"""
NDD-backed NemotronCC synthetic data generation stages.

Drop-in replacements for the stages in
``nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc`` that use
NeMo Data Designer instead of LLMClient/AsyncLLMClient.
"""

from dataclasses import dataclass

from nemo_curator.stages.synthetic.nemotron_cc.nemo_data_designer.base import NDDBaseSyntheticStage
from nemo_curator.stages.synthetic.nemotron_cc.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
)


@dataclass
class WikipediaParaphrasingStage(NDDBaseSyntheticStage):
    system_prompt: str = NEMOTRON_CC_SYSTEM_PROMPT
    prompt: str = WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE
    input_field: str = "text"
    output_field: str = "rephrased"


@dataclass
class DiverseQAStage(NDDBaseSyntheticStage):
    system_prompt: str = NEMOTRON_CC_SYSTEM_PROMPT
    prompt: str = DIVERSE_QA_PROMPT_TEMPLATE
    input_field: str = "text"
    output_field: str = "diverse_qa"


@dataclass
class DistillStage(NDDBaseSyntheticStage):
    system_prompt: str = NEMOTRON_CC_DISTILL_SYSTEM_PROMPT
    prompt: str = DISTILL_PROMPT_TEMPLATE
    input_field: str = "text"
    output_field: str = "distill"


@dataclass
class ExtractKnowledgeStage(NDDBaseSyntheticStage):
    system_prompt: str = NEMOTRON_CC_SYSTEM_PROMPT
    prompt: str = EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE
    input_field: str = "text"
    output_field: str = "extract_knowledge"


@dataclass
class KnowledgeListStage(NDDBaseSyntheticStage):
    system_prompt: str = NEMOTRON_CC_SYSTEM_PROMPT
    prompt: str = KNOWLEDGE_LIST_PROMPT_TEMPLATE
    input_field: str = "text"
    output_field: str = "knowledge_list"
