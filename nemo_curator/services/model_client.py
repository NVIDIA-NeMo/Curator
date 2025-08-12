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
from abc import ABC, abstractmethod
from collections.abc import Iterable

from nemo_curator.services.conversation_formatter import ConversationFormatter


class LLMClient(ABC):
    """
    Interface representing a client connecting to an LLM inference server
    and making requests synchronously
    """

    @abstractmethod
    def query_model(  # noqa: PLR0913
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        max_tokens: int | None = None,
        n: int | None = 1,
        seed: int | None = None,
        stop: str | None | list[str] = None,
        stream: bool = False,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[str]:
        msg = "Subclass of LLMClient must implement 'query_model'"
        raise NotImplementedError(msg)

    @abstractmethod
    def query_reward_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
    ) -> dict:
        msg = "Subclass of LLMClient must implement 'query_reward_model'"
        raise NotImplementedError(msg)


class AsyncLLMClient(ABC):
    """
    Interface representing a client connecting to an LLM inference server
    and making requests asynchronously
    """

    @abstractmethod
    async def query_model(  # noqa: PLR0913
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        max_tokens: int | None = None,
        n: int | None = 1,
        seed: int | None = None,
        stop: str | None | list[str] = None,
        stream: bool = False,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[str]:
        msg = "Subclass of AsyncLLMClient must implement 'query_model'"
        raise NotImplementedError(msg)

    @abstractmethod
    async def query_reward_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
    ) -> dict:
        msg = "Subclass of LLMClient must implement 'query_reward_model'"
        raise NotImplementedError(msg)
