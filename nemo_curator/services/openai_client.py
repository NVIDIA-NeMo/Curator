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
import warnings
from typing import Iterable, List, Optional, Union

from openai import AsyncOpenAI, OpenAI
from openai._types import NOT_GIVEN, NotGiven

from nemo_curator.services.conversation_formatter import ConversationFormatter

from .model_client import AsyncLLMClient, LLMClient


class OpenAIClient(LLMClient):
    """
    A wrapper around OpenAI's Python client for querying models
    """

    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client

    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: Optional[ConversationFormatter] = None,
        max_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        n: Union[Optional[int], NotGiven] = NOT_GIVEN,
        seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
        stop: Union[Optional[str], List[str], NotGiven] = NOT_GIVEN,
        stream: Union[Optional[bool], NotGiven] = False,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        top_k: Optional[int] = None,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
    ) -> List[str]:

        if conversation_formatter is not None:
            warnings.warn("conversation_formatter is not used in an OpenAIClient")
        if top_k is not None:
            warnings.warn("top_k is not used in an OpenAIClient")

        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
        )

        return [choice.message.content for choice in response.choices]

    def query_reward_model(self, *, messages: Iterable, model: str) -> dict:
        """
        Prompts an LLM Reward model to score a conversation between a user and assistant
        Args:
            messages: The conversation to calculate a score for.
                Should be formatted like:
                    [{"role": "user", "content": "Write a sentence"}, {"role": "assistant", "content": "This is a sentence"}, ...]
            model: The name of the model that should be used to calculate the reward.
                Must be a reward model, cannot be a regular LLM.
        Returns:
            A mapping of score_name -> score
        """
        response = self.client.chat.completions.create(messages=messages, model=model)

        if response.choices[0].logprobs is None:
            raise ValueError(
                f"Logprobs not found. {model} is likely not a reward model."
            )

        scores = {
            score.token: score.logprob for score in response.choices[0].logprobs.content
        }

        return scores


class AsyncOpenAIClient(AsyncLLMClient):
    """
    A wrapper around OpenAI's Python async client for querying models
    """

    def __init__(self, async_openai_client: AsyncOpenAI) -> None:
        self.client = async_openai_client

    async def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: Optional[ConversationFormatter] = None,
        max_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        n: Union[Optional[int], NotGiven] = NOT_GIVEN,
        seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
        stop: Union[Optional[str], List[str], NotGiven] = NOT_GIVEN,
        stream: Union[Optional[bool], NotGiven] = False,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        top_k: Optional[int] = None,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
    ) -> List[str]:

        if conversation_formatter is not None:
            warnings.warn("conversation_formatter is not used in an AsyncOpenAIClient")
        if top_k is not None:
            warnings.warn("top_k is not used in an AsyncOpenAIClient")

        response = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
        )

        return [choice.message.content for choice in response.choices]

    async def query_reward_model(self, *, messages: Iterable, model: str) -> dict:
        """
        Prompts an LLM Reward model to score a conversation between a user and assistant
        Args:
            messages: The conversation to calculate a score for.
                Should be formatted like:
                    [{"role": "user", "content": "Write a sentence"}, {"role": "assistant", "content": "This is a sentence"}, ...]
            model: The name of the model that should be used to calculate the reward.
                Must be a reward model, cannot be a regular LLM.
        Returns:
            A mapping of score_name -> score
        """
        response = await self.client.chat.completions.create(
            messages=messages, model=model
        )

        if response.choices[0].logprobs is None:
            raise ValueError(
                f"Logprobs not found. {model} is likely not a reward model."
            )

        scores = {
            score.token: score.logprob for score in response.choices[0].logprobs.content
        }

        return scores
