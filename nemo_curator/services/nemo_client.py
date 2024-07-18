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
from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Union

from nemo_curator.services.conversation_formatter import ConversationFormatter

from .model_client import AsyncLLMClient, LLMClient


class NemoDeployClient(LLMClient):
    """
    A wrapper around NemoQueryLLM for querying models in synthetic data generation
    """

    def __init__(self, nemo_deploy: NemoQueryLLM) -> None:
        self.client = nemo_deploy

    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: Optional[ConversationFormatter] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        if conversation_formatter is None:
            raise ValueError(
                "NemoDeployClient's query_model requires a conversation_formatter"
            )

        prompt = conversation_formatter.format_conversation(messages)
        self.client.model_name = model

        if n is not None:
            warnings.warn("n is not supported in NemoDeployClient")
        if stream:
            warnings.warn("streamming is not supported in NeMoDeployClient")

        if isinstance(stop, str):
            stop = [stop]

        response = self.client.query_llm(
            prompts=[prompt],
            max_output_len=max_tokens,
            random_seed=seed,
            stop_words_list=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )[0]

        return self._postprocess_response(response, stop)

    @staticmethod
    def _postprocess_response(responses: List[str], stop_words: List[str]) -> List[str]:
        processed_responses = []
        for response in responses:
            for stop in stop_words:
                if response.endswith(stop):
                    response = response[: -len(stop)]
            processed_responses.append(response.strip())
        return processed_responses

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
        raise NotImplementedError(
            "Reward model inference is not supported in NeMo Deploy Clients"
        )
