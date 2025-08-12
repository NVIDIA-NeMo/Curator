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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nemo_curator.services.conversation_formatter import ConversationFormatter

from .model_client import LLMClient


class NemoDeployClient(LLMClient):
    """
    A wrapper around NemoQueryLLM for querying models in synthetic data generation
    """

    def __init__(self, nemo_deploy: "NemoQueryLLM") -> None:  # noqa: UP037, F821
        self.client = nemo_deploy

    def query_model(  # noqa: PLR0913
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        seed: int | None = None,
        stop: str | None | list[str] = [],  # noqa: B006
        stream: bool = False,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[str]:
        if conversation_formatter is None:
            msg = "NemoDeployClient's query_model requires a conversation_formatter"
            raise ValueError(msg)

        prompt = conversation_formatter.format_conversation(messages)
        self.client.model_name = model

        if n is not None:
            warnings.warn("n is not supported in NemoDeployClient", stacklevel=2)
        if stream:
            warnings.warn("streamming is not supported in NeMoDeployClient", stacklevel=2)

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
    def _postprocess_response(responses: list[str], stop_words: list[str]) -> list[str]:
        processed_responses = []
        for response in responses:
            for stop in stop_words:
                response = response.removesuffix(stop)  # noqa: PLW2901
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
        msg = "Reward model inference is not supported in NeMo Deploy Clients"
        raise NotImplementedError(msg)
