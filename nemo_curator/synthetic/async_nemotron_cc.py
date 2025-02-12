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

from typing import List

from nemo_curator.services import AsyncLLMClient
from nemo_curator.synthetic.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
)


class AsyncNemotronCCGenerator:
    """
    Provides a collection of methods for generating synthetic data
    described in the Nemotron-CC paper (https://arxiv.org/abs/2412.02595).
    """

    def __init__(self, llm_client: AsyncLLMClient) -> None:
        """
        Initialize the AsyncNemotronCCGenerator instance.

        Args:
            llm_client (LLMClient): The language model client used for querying the model.
        """
        self.client = llm_client

    async def _prompt(
        self,
        model: str,
        document: str,
        prompt_template: str,
        system_prompt: str,
        prompt_kwargs: dict,
        model_kwargs: dict,
    ) -> List[str]:
        prompt = prompt_template.format(document=document, **prompt_kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        return await self.client.query_model(
            messages=messages, model=model, **model_kwargs
        )

    async def rewrite_to_wikipedia_style(
        self,
        document: str,
        model: str,
        prompt_template: str = WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
        system_prompt: str = NEMOTRON_CC_SYSTEM_PROMPT,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Rewrites a document into a Wikipedia-style narrative.

        Args:
            document (str): The input document text to rewrite.
            model (str): The model identifier to use.
            prompt_template (str, optional): The prompt template for rewriting. Defaults to WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE.
            system_prompt (str, optional): The system prompt to use. Defaults to NEMOTRON_CC_SYSTEM_PROMPT.
            prompt_kwargs (dict, optional): Additional keyword arguments for the prompt. Defaults to {}.
            model_kwargs (dict, optional): Additional keyword arguments for the model invocation. Defaults to {}.

        Returns:
            List[str]: A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        return await self._prompt(
            model, document, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )

    async def generate_diverse_qa(
        self,
        document: str,
        model: str,
        prompt_template: str = DIVERSE_QA_PROMPT_TEMPLATE,
        system_prompt: str = NEMOTRON_CC_SYSTEM_PROMPT,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Generates diverse QA pairs from the provided document.

        Args:
            document (str): The input document text used to generate QA pairs.
            model (str): The model identifier to use.
            prompt_template (str, optional): The prompt template for generating QA pairs. Defaults to DIVERSE_QA_PROMPT_TEMPLATE.
            system_prompt (str, optional): The system prompt to use. Defaults to NEMOTRON_CC_SYSTEM_PROMPT.
            prompt_kwargs (dict, optional): Additional keyword arguments for the prompt. Defaults to {}.
            model_kwargs (dict, optional): Additional keyword arguments for the model invocation. Defaults to {}.

        Returns:
            List[str]: A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        return await self._prompt(
            model, document, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )

    async def distill(
        self,
        document: str,
        model: str,
        prompt_template: str = DISTILL_PROMPT_TEMPLATE,
        system_prompt: str = NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Distills the essential content from a document.

        Args:
            document (str): The input document text to distill.
            model (str): The model identifier to use.
            prompt_template (str, optional): The prompt template for distillation. Defaults to DISTILL_PROMPT_TEMPLATE.
            system_prompt (str, optional): The system prompt to use. Defaults to NEMOTRON_CC_DISTILL_SYSTEM_PROMPT.
            prompt_kwargs (dict, optional): Additional keyword arguments for the prompt. Defaults to {}.
            model_kwargs (dict, optional): Additional keyword arguments for the model invocation. Defaults to {}.

        Returns:
            List[str]: A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        return await self._prompt(
            model, document, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )

    async def extract_knowledge(
        self,
        document: str,
        model: str,
        prompt_template: str = EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
        system_prompt: str = NEMOTRON_CC_SYSTEM_PROMPT,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Extracts knowledge from the provided document.

        Args:
            document (str): The input document text from which to extract knowledge.
            model (str): The model identifier to use.
            prompt_template (str, optional): The prompt template for knowledge extraction. Defaults to EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE.
            system_prompt (str, optional): The system prompt to use. Defaults to NEMOTRON_CC_SYSTEM_PROMPT.
            prompt_kwargs (dict, optional): Additional keyword arguments for the prompt. Defaults to {}.
            model_kwargs (dict, optional): Additional keyword arguments for the model invocation. Defaults to {}.

        Returns:
            List[str]: A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        return await self._prompt(
            model, document, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )

    async def generate_knowledge_list(
        self,
        document: str,
        model: str,
        prompt_template: str = KNOWLEDGE_LIST_PROMPT_TEMPLATE,
        system_prompt: str = NEMOTRON_CC_SYSTEM_PROMPT,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Generates a list of knowledge items from the provided document.

        Args:
            document (str): The input document text to process.
            model (str): The model identifier to use.
            prompt_template (str, optional): The prompt template for generating a knowledge list. Defaults to KNOWLEDGE_LIST_PROMPT_TEMPLATE.
            system_prompt (str, optional): The system prompt to use. Defaults to NEMOTRON_CC_SYSTEM_PROMPT.
            prompt_kwargs (dict, optional): Additional keyword arguments for the prompt. Defaults to {}.
            model_kwargs (dict, optional): Additional keyword arguments for the model invocation. Defaults to {}.

        Returns:
            List[str]: A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        return await self._prompt(
            model, document, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )
