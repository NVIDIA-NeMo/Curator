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
from typing import Iterable, List, Optional, Union

from nemo_curator.services.conversation_formatter import ConversationFormatter


class LLMClient(ABC):
    """
    Interface representing a client connecting to an LLM inference server
    and making requests synchronously
    """

    @abstractmethod
    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: Optional[ConversationFormatter] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = 1,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        raise NotImplementedError("Subclass of LLMClient must implement 'query_model'")

    @abstractmethod
    def query_reward_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: Optional[ConversationFormatter] = None,
    ) -> dict:
        raise NotImplementedError(
            "Subclass of LLMClient must implement 'query_reward_model'"
        )


class AsyncLLMClient(ABC):
    """
    Interface representing a client connecting to an LLM inference server
    and making requests asynchronously
    """

    @abstractmethod
    async def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: Optional[ConversationFormatter] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = 1,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        raise NotImplementedError(
            "Subclass of AsyncLLMClient must implement 'query_model'"
        )

    @abstractmethod
    async def query_reward_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: Optional[ConversationFormatter] = None,
    ) -> dict:
        raise NotImplementedError(
            "Subclass of LLMClient must implement 'query_reward_model'"
        )
