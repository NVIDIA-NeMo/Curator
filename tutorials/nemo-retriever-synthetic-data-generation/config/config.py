# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from nemo_curator.modules.config import BaseConfig


@dataclass
class RetrieverEvalSDGConfig(BaseConfig):
    """
    Configuration for SDG pipeline for Retriever Evals

    Attributes:

    """

    base_url: str
    api_key: str
    generator_model: str = "mistralai/mixtral-8x22b-instruct-v0.1"
    generator_url: Optional[str] = None
    generator_api_key: Optional[str] = None
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0
    num_questions: Optional[int] = 1
    max_tokens: Optional[int] = 2048
    squad_format: Optional[bool] = False

    generator_system_prompt: Optional[
        str
    ] = """You are data annotator, your task is
    to generate a question for the given document. Also generate answer to the generated
    question."""

    generator_user_prompt_template: Optional[
        str
    ] = """Generate {n_openlines} questions and corresponding answers based on Input Document.
    Input Document:
    {document}
    """

    # easiness filter parameters
    easiness_filter: str = None
    easiness_url: Optional[str] = None
    easiness_api_key: Optional[str] = None
    truncate: str = "END"
    percentile: float = 70
    batch_size: Optional[int] = 1

    # answerability filter parameters
    answerability_filter: str = None
    answerability_url: Optional[str] = None
    answerability_api_key: Optional[str] = None
    num_criteria: int = 4
    answerability_system_prompt: str = """You are an evaluator who is rating questions to given context passages based on the given criteria. Assess the given question for clarity and answerability given enough domain knowledge, consider the following evaluation criterion:
      Criterion 1 - Can the question be understood and answered without needing additional context or access to external references not provided within the question itself? Questions should be self-contained, meaning they do not rely on specific documents, tables, or prior knowledge not shared within the question.
      Criterion 2 - Is it clear what type of answer or information the question seeks? The question should convey its purpose without ambiguity, allowing for a direct and relevant response.
      Criterion 3 - Does the content in the context contain information that can answer the question or part of the question?
      Criterion 4 - Does the content in the context completely answer the question?

      Provide your response in a mandatory dictionary format, and a short explanation of the rating like
      {
      \"criterion_1_explanation\": "<Brief explanation of why criterion_1 was satisfied or not satisfied>",
      \"criterion_1\": "<Y/N>",
      \"criterion_2_explanation\":  "<State the purpose of the question and justify why it was satisfied or not satisfied>",
      \"criterion_2\": "<Y/N>",
      \"criterion_3_explanation\": "<Show what parts of the content contain relevant information to the question if this criterion is satisfied, state why the information is irrelevant if unsatisfied>",
      \"criterion_3\": "<Y/N>",
      \"criterion_4_explanation\": "<Extract spans from the content that help completely answer the question if criterion is satisfied, state what parts are missing if not satisfied>",
      \"criterion_4\": "<Y/N>"
      }
      Provide only the dictionary response and nothing else.
    """
    answerability_user_prompt_template: str = """Context Passage:
    {context}
    Question:
    {question}
    """
