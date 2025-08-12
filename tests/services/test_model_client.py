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

from collections.abc import Iterable

import pytest

from nemo_curator.services.conversation_formatter import ConversationFormatter
from nemo_curator.services.model_client import AsyncLLMClient, LLMClient


class TestLLMClient:
    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that LLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
            LLMClient()

    def test_must_implement_query_model(self) -> None:
        """Test that subclasses must implement query_model."""

        class IncompleteClient(LLMClient):
            def query_reward_model(
                self,
                *,
                messages: Iterable,  # noqa: ARG002
                model: str,  # noqa: ARG002
                conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
            ) -> dict:
                return {}

        with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
            IncompleteClient()

    def test_must_implement_query_reward_model(self) -> None:
        """Test that subclasses must implement query_reward_model."""

        class IncompleteClient(LLMClient):
            def query_model(  # noqa: PLR0913
                self,
                *,
                messages: Iterable,  # noqa: ARG002
                model: str,  # noqa: ARG002
                conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
                max_tokens: int | None = None,  # noqa: ARG002
                n: int | None = 1,  # noqa: ARG002
                seed: int | None = None,  # noqa: ARG002
                stop: str | None | list[str] = None,  # noqa: ARG002
                stream: bool = False,  # noqa: ARG002
                temperature: float | None = None,  # noqa: ARG002
                top_k: int | None = None,  # noqa: ARG002
                top_p: float | None = None,  # noqa: ARG002
            ) -> list[str]:
                return ["response"]

        with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
            IncompleteClient()

    def test_complete_implementation(self) -> None:
        """Test that a complete implementation can be instantiated."""

        class CompleteClient(LLMClient):
            def query_model(  # noqa: PLR0913
                self,
                *,
                messages: Iterable,  # noqa: ARG002
                model: str,  # noqa: ARG002
                conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
                max_tokens: int | None = None,  # noqa: ARG002
                n: int | None = 1,  # noqa: ARG002
                seed: int | None = None,  # noqa: ARG002
                stop: str | None | list[str] = None,  # noqa: ARG002
                stream: bool = False,  # noqa: ARG002
                temperature: float | None = None,  # noqa: ARG002
                top_k: int | None = None,  # noqa: ARG002
                top_p: float | None = None,  # noqa: ARG002
            ) -> list[str]:
                return ["response"]

            def query_reward_model(
                self,
                *,
                messages: Iterable,  # noqa: ARG002
                model: str,  # noqa: ARG002
                conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
            ) -> dict:
                return {"score": 0.5}

        client = CompleteClient()
        assert isinstance(client, LLMClient)


class TestAsyncLLMClient:
    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that AsyncLLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
            AsyncLLMClient()

    def test_must_implement_query_model(self) -> None:
        """Test that subclasses must implement query_model."""

        class IncompleteAsyncClient(AsyncLLMClient):
            async def query_reward_model(
                self,
                *,
                messages: Iterable,  # noqa: ARG002
                model: str,  # noqa: ARG002
                conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
            ) -> dict:
                return {}

        with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
            IncompleteAsyncClient()

    def test_must_implement_query_reward_model(self) -> None:
        """Test that subclasses must implement query_reward_model."""

        class IncompleteAsyncClient(AsyncLLMClient):
            async def query_model(  # noqa: PLR0913
                self,
                *,
                messages: Iterable,  # noqa: ARG002
                model: str,  # noqa: ARG002
                conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
                max_tokens: int | None = None,  # noqa: ARG002
                n: int | None = 1,  # noqa: ARG002
                seed: int | None = None,  # noqa: ARG002
                stop: str | None | list[str] = None,  # noqa: ARG002
                stream: bool = False,  # noqa: ARG002
                temperature: float | None = None,  # noqa: ARG002
                top_k: int | None = None,  # noqa: ARG002
                top_p: float | None = None,  # noqa: ARG002
            ) -> list[str]:
                return ["response"]

        with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
            IncompleteAsyncClient()

    def test_complete_implementation(self) -> None:
        """Test that a complete implementation can be instantiated."""

        class CompleteAsyncClient(AsyncLLMClient):
            async def query_model(  # noqa: PLR0913
                self,
                *,
                messages: Iterable,  # noqa: ARG002
                model: str,  # noqa: ARG002
                conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
                max_tokens: int | None = None,  # noqa: ARG002
                n: int | None = 1,  # noqa: ARG002
                seed: int | None = None,  # noqa: ARG002
                stop: str | None | list[str] = None,  # noqa: ARG002
                stream: bool = False,  # noqa: ARG002
                temperature: float | None = None,  # noqa: ARG002
                top_k: int | None = None,  # noqa: ARG002
                top_p: float | None = None,  # noqa: ARG002
            ) -> list[str]:
                return ["response"]

            async def query_reward_model(
                self,
                *,
                messages: Iterable,  # noqa: ARG002
                model: str,  # noqa: ARG002
                conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
            ) -> dict:
                return {"score": 0.5}

        client = CompleteAsyncClient()
        assert isinstance(client, AsyncLLMClient)


@pytest.mark.asyncio
async def test_async_implementation_called_correctly() -> None:
    """Test that an async implementation's methods are called with the right arguments."""

    class MockAsyncClient(AsyncLLMClient):
        def __init__(self):
            self.query_model_called = False
            self.query_reward_model_called = False
            self.last_model = None
            self.last_messages = None

        async def query_model(  # noqa: PLR0913
            self,
            *,
            messages: Iterable,
            model: str,
            conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
            max_tokens: int | None = None,  # noqa: ARG002
            n: int | None = 1,  # noqa: ARG002
            seed: int | None = None,  # noqa: ARG002
            stop: str | None | list[str] = None,  # noqa: ARG002
            stream: bool = False,  # noqa: ARG002
            temperature: float | None = None,  # noqa: ARG002
            top_k: int | None = None,  # noqa: ARG002
            top_p: float | None = None,  # noqa: ARG002
        ) -> list[str]:
            self.query_model_called = True
            self.last_model = model
            self.last_messages = messages
            return ["test response"]

        async def query_reward_model(
            self,
            *,
            messages: Iterable,
            model: str,
            conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
        ) -> dict:
            self.query_reward_model_called = True
            self.last_model = model
            self.last_messages = messages
            return {"GOOD": 0.8}

    client = MockAsyncClient()
    test_messages = [{"role": "user", "content": "test"}]

    # Test query_model
    result = await client.query_model(messages=test_messages, model="test-model")
    assert client.query_model_called
    assert client.last_model == "test-model"
    assert client.last_messages == test_messages
    assert result == ["test response"]

    # Test query_reward_model
    result = await client.query_reward_model(messages=test_messages, model="reward-model")
    assert client.query_reward_model_called
    assert client.last_model == "reward-model"
    assert client.last_messages == test_messages
    assert result == {"GOOD": 0.8}


def test_sync_implementation_called_correctly() -> None:
    """Test that a sync implementation's methods are called with the right arguments."""

    class MockClient(LLMClient):
        def __init__(self):
            self.query_model_called = False
            self.query_reward_model_called = False
            self.last_model = None
            self.last_messages = None

        def query_model(  # noqa: PLR0913
            self,
            *,
            messages: Iterable,
            model: str,
            conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
            max_tokens: int | None = None,  # noqa: ARG002
            n: int | None = 1,  # noqa: ARG002
            seed: int | None = None,  # noqa: ARG002
            stop: str | None | list[str] = None,  # noqa: ARG002
            stream: bool = False,  # noqa: ARG002
            temperature: float | None = None,  # noqa: ARG002
            top_k: int | None = None,  # noqa: ARG002
            top_p: float | None = None,  # noqa: ARG002
        ) -> list[str]:
            self.query_model_called = True
            self.last_model = model
            self.last_messages = messages
            return ["test response"]

        def query_reward_model(
            self,
            *,
            messages: Iterable,
            model: str,
            conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
        ) -> dict:
            self.query_reward_model_called = True
            self.last_model = model
            self.last_messages = messages
            return {"GOOD": 0.8}

    client = MockClient()
    test_messages = [{"role": "user", "content": "test"}]

    # Test query_model
    result = client.query_model(messages=test_messages, model="test-model")
    assert client.query_model_called
    assert client.last_model == "test-model"
    assert client.last_messages == test_messages
    assert result == ["test response"]

    # Test query_reward_model
    result = client.query_reward_model(messages=test_messages, model="reward-model")
    assert client.query_reward_model_called
    assert client.last_model == "reward-model"
    assert client.last_messages == test_messages
    assert result == {"GOOD": 0.8}
