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

import random
from typing import Any, List, Optional

from transformers import AutoTokenizer

from nemo_curator import BaseModule
from nemo_curator.datasets import DocumentDataset
from nemo_curator.services import LLMClient
from nemo_curator.synthetic.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
)


class NemotronCCGenerator:
    """
    Provides a collection of methods for generating synthetic data
    described in the Nemotron-CC paper (https://arxiv.org/abs/2412.02595).
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """
        Initialize the NemotronCCGenerator instance.

        Args:
            llm_client (LLMClient): The language model client used for querying the model.
        """
        self.client = llm_client

    def _prompt(
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

        return self.client.query_model(messages=messages, model=model, **model_kwargs)

    def rewrite_to_wikipedia_style(
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
        return self._prompt(
            model, document, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )

    def generate_diverse_qa(
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
        return self._prompt(
            model, document, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )

    def distill(
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
        return self._prompt(
            model, document, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )

    def extract_knowledge(
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
        return self._prompt(
            model, document, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )

    def generate_knowledge_list(
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
        return self._prompt(
            model, document, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )


class NemotronCCDiverseQAPostprocessor(BaseModule):
    """
    Postprocesses the output of the Nemotron-CC Diverse QA generation pipeline.
    This postprocessor will sample a random number of QA pairs up to max_num_pairs.
    If a tokenizer is provided, the number of QA pairs will be sampled from at least 1 and at most floor(max_num_pairs * num_tokens / 150).
    Otherwise, the number of QA pairs will be sampled randomly strictly up to max_num_pairs.

    The generated QA pairs are shuffled and then appended to the original text.
    """

    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        text_field: str = "text",
        response_field: str = "response",
        max_num_pairs: int = 1,
        prefix: str = "Here are the questions and answers based on the provided text:",
    ) -> None:
        """
        Args:
            tokenizer (Optional[AutoTokenizer]): The tokenizer to use for tokenization.
                If specified, the number of QA pairs will be sampled based on the token count of the text.
                If not specified, the number of QA pairs will be sampled randomly up to max_num_pairs.
            text_field (str): The field in the dataset that contains the text used to generate QA pairs.
            response_field (str): The field in the dataset that contains the response from the LLM.
            max_num_pairs (int): The maximum number of QA pairs to sample.
            prefix (str): The prefix of the response from the LLM.
        """
        super().__init__(input_backend="pandas")
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.response_field = response_field
        self.max_num_pairs = max_num_pairs
        self.prefix = prefix

    def _postprocess_llm_response(self, text: str, llm_response: str) -> str:
        lines = [line.strip() for line in llm_response.split("\n") if line.strip()]
        if not lines:
            return ""

        # Remove the "- " prefix
        lines = [line[2:].strip() if line.startswith("- ") else line for line in lines]

        if lines[0] == self.prefix:
            lines = lines[1:]

        # Merge question and answer lines
        qa_pairs = []
        for line in lines:
            if line.startswith("Question:"):
                qa_pairs.append(line)
            else:
                if qa_pairs:
                    qa_pairs[-1] += "\n" + line
                else:
                    return ""

        if len(qa_pairs) == 0:
            return ""

        # Shuffle the QA pairs and sample up to max_num_pairs
        random.shuffle(qa_pairs)
        if self.tokenizer is not None:
            num_tokens = len(self.tokenizer.tokenize(text))
            qa_pairs = qa_pairs[
                : random.randint(1, max(1, int(self.max_num_pairs * num_tokens / 150)))
            ]
        else:
            qa_pairs = qa_pairs[: random.randint(1, self.max_num_pairs)]
        qa_pairs_str = "\n\n".join(qa_pairs)

        # Concatenate the document and the QA pairs
        return f"{text}\n\n{qa_pairs_str}"

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        df = dataset.df
        df[self.response_field] = df.apply(
            lambda row: self._postprocess_llm_response(
                row[self.text_field], row[self.response_field]
            ),
            axis=1,
            meta=(None, "object"),
        )
        df = df[df[self.response_field] != ""]

        return DocumentDataset(df)


# Although this could be implemented as a DocumentModifier,
# I have kept it separate to match the other postprocessors.
class NemotronCCKnowledgeListPostprocessor(BaseModule):
    """
    Processes and cleans the output generated by the Nemotron-CC Knowledge List pipeline.

    This class is responsible for postprocessing raw text responses produced by the
    Nemotron-CC Knowledge List generation pipeline. It removes formatting artifacts
    such as bullet point prefixes ("- ") and extra indentation from each line, ensuring
    that the final output is a clean, uniformly formatted list of knowledge items.
    The processing includes skipping any initial non-bullet line and merging related lines
    to reconstruct multi-line questions or answers.
    """

    def __init__(self, text_field: str = "text") -> None:
        super().__init__(input_backend="pandas")
        self.text_field = text_field

    def _postprocess_llm_response(self, text: str) -> str:
        lines = []
        for idx, line in enumerate(text.split("\n")):
            if idx == 0 and not line.startswith("-"):
                continue

            if line.startswith("  ") or line.startswith("- "):
                lines.append(line[2:].strip())
            else:
                lines.append(line)
        return "\n".join(lines)

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        df = dataset.df
        df[self.text_field] = df[self.text_field].apply(
            self._postprocess_llm_response, meta=(self.text_field, "object")
        )
        return DocumentDataset(df)
